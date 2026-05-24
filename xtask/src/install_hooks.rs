//! Write `.git/hooks/pre-commit` invoking `cargo xtask check-paths
//! --staged`. Idempotent: overwrites the file in place.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

const HOOK: &str = r#"#!/usr/bin/env bash
# mlxr pre-commit hook — installed by `cargo run -q -p xtask -- install-hooks`.
#
# Runs the check-paths lint against the staged Rust files. Blocks the
# commit on any inline multi-segment path qualifier — see
# CODE_REVIEW.md "Imports and paths" for the rationale.
#
# To bypass for a single commit: `git commit --no-verify`.

set -euo pipefail

STAGED=$(git diff --cached --name-only --diff-filter=ACMR \
            | grep -E '\.rs$' || true)

if [ -z "$STAGED" ]; then
    exit 0
fi

# Pass the staged list to xtask so we only lint what's actually being
# committed.
exec cargo run --quiet -p xtask -- check-paths --staged $STAGED
"#;

pub fn run(repo_root: &Path) -> Result<(), String> {
    let hook_path = repo_root.join(".git/hooks/pre-commit");
    let hooks_dir = hook_path.parent().expect(".git/hooks parent");
    if !hooks_dir.exists() {
        return Err(format!(
            "{}: directory missing (is this a git repo?)",
            hooks_dir.display()
        ));
    }
    fs::write(&hook_path, HOOK).map_err(|e| format!("write {}: {e}", hook_path.display()))?;
    let mut perms = fs::metadata(&hook_path)
        .map_err(|e| format!("stat {}: {e}", hook_path.display()))?
        .permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&hook_path, perms)
        .map_err(|e| format!("chmod {}: {e}", hook_path.display()))?;
    println!("installed pre-commit hook at {}", hook_path.display());
    Ok(())
}
