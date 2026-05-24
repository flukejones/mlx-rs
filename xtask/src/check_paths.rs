//! Workspace lint: inline multi-segment path qualifiers.
//!
//! Catches paths like `crate::foo::Bar` (or `super::…`, `self::…`)
//! appearing inline in code — type annotations, fn signatures, struct
//! fields, generic args, callsites. The hard rule: at most ONE path
//! segment after `crate`/`super`/`self` is fine; anything deeper
//! belongs in a top-of-file `use`.
//!
//! Skips:
//! - `use ...;` statements
//! - line and block doc comments (`//`, `///`, `//!`, `/* … */`)
//! - string literals (caller burden — we don't parse strings)
//! - `#[…]` attributes
//! - `mlx-c` vendored upstream (`crates/mlxr-sys/src/mlx-c/`)
//!
//! Exit code: `0` clean, `1` violations (with file:line list on stderr).

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

/// Run the check against `roots` (typically `crates/`, `examples/`).
///
/// If `restrict_to` is `Some`, only lint files whose absolute path is
/// in that set. This is the pre-commit hook entry point: pass the
/// staged-file list so the hook only blocks on files actually being
/// committed.
pub fn run(repo_root: &Path, restrict_to: Option<&[PathBuf]>) -> Result<(), String> {
    let roots = [repo_root.join("crates"), repo_root.join("examples")];
    let mut violations: Vec<Violation> = Vec::new();

    for root in &roots {
        if !root.exists() {
            continue;
        }
        walk(root, &mut |path| {
            if let Some(allowed) = restrict_to {
                if !allowed.iter().any(|p| p == path) {
                    return;
                }
            }
            if let Err(e) = scan_file(path, &mut violations) {
                eprintln!("check-paths: {}: {e}", path.display());
            }
        });
    }

    if violations.is_empty() {
        return Ok(());
    }

    let n = violations.len();
    eprintln!(
        "\x1b[31mcheck-paths: {n} inline multi-segment qualifier(s) — see CODE_REVIEW.md \"Imports and paths\"\x1b[0m"
    );
    for v in &violations {
        eprintln!(
            "  {}:{}: {}",
            v.path.display(),
            v.line,
            v.snippet.trim_end()
        );
    }
    Err(format!("{n} violation(s)"))
}

struct Violation {
    path: PathBuf,
    line: usize,
    snippet: String,
}

fn walk(dir: &Path, on_file: &mut dyn FnMut(&Path)) {
    if dir.ends_with("mlx-c") {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(&path, on_file);
        } else if path.extension() == Some(OsStr::new("rs")) {
            on_file(&path);
        }
    }
}

fn scan_file(path: &Path, violations: &mut Vec<Violation>) -> std::io::Result<()> {
    let src = fs::read_to_string(path)?;
    let mut in_block_comment = false;

    for (idx, raw) in src.lines().enumerate() {
        let line_no = idx + 1;
        let (stripped, still_in_block) = strip_block_comments(raw, in_block_comment);
        in_block_comment = still_in_block;
        let trimmed = stripped.trim_start();

        // Skip whole-line doc / line comments / attributes / use stmts.
        if trimmed.starts_with("///")
            || trimmed.starts_with("//!")
            || trimmed.starts_with("//")
            || trimmed.starts_with("#[")
            || trimmed.starts_with("#![")
            || is_use_statement(trimmed)
        {
            continue;
        }

        if let Some(snippet) = find_qualifier(&stripped) {
            violations.push(Violation {
                path: path.to_path_buf(),
                line: line_no,
                snippet,
            });
        }
    }
    Ok(())
}

/// Trim a `/* ... */` block comment span. `in_block` carries open
/// state across lines. Returns `(stripped_text, still_in_block)`.
fn strip_block_comments(line: &str, mut in_block: bool) -> (String, bool) {
    let bytes = line.as_bytes();
    let mut out = String::with_capacity(line.len());
    let mut i = 0;
    while i < bytes.len() {
        if in_block {
            if i + 1 < bytes.len() && bytes[i] == b'*' && bytes[i + 1] == b'/' {
                in_block = false;
                i += 2;
            } else {
                i += 1;
            }
        } else if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            in_block = true;
            i += 2;
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    (out, in_block)
}

/// `use ...;` (possibly with leading `pub `). We accept any `use` at
/// indent zero or with arbitrary leading whitespace.
fn is_use_statement(trimmed: &str) -> bool {
    trimmed.starts_with("use ") || trimmed.starts_with("pub use ")
}

/// Hunt for a banned inline path qualifier. Returns the offending
/// substring (~30 chars context) on first hit.
///
/// Rule: at most one path segment after `crate`/`super`/`self`.
/// `::Method` / `::CONST` after a Type doesn't count (it's dispatch
/// on a resolved type, not another path hop).
///
/// Allowed shapes (NOT flagged):
/// - `crate::Foo`                 — 1 segment
/// - `super::Foo`                 — 1 segment
/// - `super::Foo::new()`          — 1 segment, `new` is assoc fn
/// - `crate::foo`                 — 1 segment (module ref)
///
/// Banned (flagged):
/// - `crate::foo::Bar`            — 2 segments
/// - `crate::config::ModelConfig` — 2 segments
/// - `super::super::config::Foo`  — 3 segments
/// - `crate::foo::bar()`          — 2 segments
///
/// The leading byte must not be `:`/`>`/`"`/`!`/`#` so we don't match
/// inside paths already part of a `use`, a turbofish, a string, or
/// an attribute.
fn find_qualifier(line: &str) -> Option<String> {
    const ROOTS: &[&str] = &["crate::", "super::", "self::"];
    let bytes = line.as_bytes();

    for root in ROOTS {
        let mut search_from = 0;
        while let Some(rel) = line[search_from..].find(root) {
            let start = search_from + rel;
            search_from = start + root.len();

            if start > 0 {
                let prev = bytes[start - 1];
                if matches!(prev, b':' | b'>' | b'"' | b'!' | b'#') {
                    continue;
                }
            }

            let rest = &line[start + root.len()..];
            if let Some(end_offset) = banned_segment_run(rest) {
                let end_col = (start + root.len() + end_offset).min(line.len());
                let snippet_start = start.saturating_sub(10);
                let snippet_end = (end_col + 10).min(line.len());
                return Some(line[snippet_start..snippet_end].to_string());
            }
        }
    }
    None
}

/// If the head of `s` is a banned path qualifier, return the byte
/// length consumed (for snippet end). Else `None`.
///
/// Rule: count path segments after the root. The first uppercase-led
/// segment terminates the path (the rest is method dispatch on a
/// resolved type, e.g. `super::Foo::new` — `Foo` is the target,
/// `new` is the assoc-fn). Banned iff segment count >= 2.
///
/// Examples:
/// - `crate::Foo` → 1 seg → allowed
/// - `super::Foo::new()` → 1 seg (Foo terminates) → allowed
/// - `crate::foo` → 1 seg → allowed
/// - `crate::foo::Bar` → 2 segs → banned
/// - `crate::config::ModelConfig` → 2 segs → banned
/// - `crate::foo::bar()` → 2 segs → banned
fn banned_segment_run(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut i = 0;
    let mut segments = 0;

    loop {
        let ident_start = i;
        while i < bytes.len() && is_ident_byte(bytes[i]) {
            i += 1;
        }
        if i == ident_start {
            return None;
        }
        segments += 1;
        let starts_uppercase = bytes[ident_start].is_ascii_uppercase();

        if starts_uppercase {
            // Terminal type segment — even if `::method` follows.
            return if segments >= 2 { Some(i) } else { None };
        }

        let has_more = i + 1 < bytes.len() && bytes[i] == b':' && bytes[i + 1] == b':';
        if !has_more {
            return if segments >= 2 { Some(i) } else { None };
        }
        i += 2;
    }
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finds(line: &str) -> bool {
        find_qualifier(line).is_some()
    }

    #[test]
    fn flags_crate_three_seg_type() {
        assert!(finds("    pub fn foo(x: &crate::config::ModelConfig) {"));
    }

    #[test]
    fn flags_super_chain() {
        assert!(finds("        let v: super::super::config::Foo = bar;"));
    }

    #[test]
    fn ignores_single_segment() {
        assert!(!finds("let x: crate::Foo = bar;"));
        assert!(!finds("    super::Foo::new()"));
    }

    #[test]
    fn ignores_use_statement() {
        // is_use_statement is checked first; find_qualifier itself
        // would match, but the line is skipped at the caller.
        // Sanity: bare `use` line should not be passed to us.
        assert!(is_use_statement("use crate::config::Family;"));
        assert!(is_use_statement("pub use crate::config::ModelConfig;"));
    }

    #[test]
    fn ignores_turbofish() {
        // `::<T>` is not an inline qualifier; we don't get false hits
        // because the leading char would be `>` for the path root.
        assert!(!finds("collect::<Vec<_>>()"));
    }

    #[test]
    fn flags_callsite() {
        assert!(finds(
            "        let cfg = crate::config::ModelConfig::from_dir(&dir);"
        ));
    }
}
