//! Generate the steel-attention Metal preamble from the cached mlx
//! source tree. Produces a bare Metal-source `.txt` under `$OUT_DIR`,
//! exposed to Rust via `STEEL_ATTENTION_PREAMBLE_PATH`. mlx-lm's kernel
//! source uses `include_str!(env!(...))` to embed the preamble in
//! `KERNEL_HEADER` for `mlx_rs::fast::metal_kernel` to compile against.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// ===================== shared mlx fetch helper =====================
// Identical block in `mlx-sys/build.rs`. Keep them in sync when edited.

/// Resolve the upstream mlx source tree, fetching it into a shared
/// `target/mlx-rs-cache/<sha>/mlx/` if absent. Honours the
/// `MLX_RS_SRC_DIR` environment override (e.g. for air-gapped builds
/// or local mlx forks).
///
/// Cargo runs build scripts in parallel, so both `mlx-sys` and `mlx-lm`
/// may race here on a cold cache. The fetch is concurrency-safe: each
/// process stages into a unique `.staging.<pid>/` directory, then races
/// to win the atomic `rename` into the shared `mlx/` slot. Losers see
/// `mlx/` already exists and clean up their staging dir without error.
fn ensure_mlx_src() -> PathBuf {
    if let Some(dir) = env::var_os("MLX_RS_SRC_DIR") {
        let path = PathBuf::from(dir);
        if !path.is_dir() {
            panic!("MLX_RS_SRC_DIR={} is not a directory", path.display());
        }
        return path;
    }

    let (version, sha) = read_workspace_mlx_pin();
    let cache_root = workspace_root().join("target/mlx-rs-cache").join(&sha);
    let mlx_dir = cache_root.join("mlx");
    let marker = cache_root.join(".fetched");

    if marker.exists() && mlx_dir.is_dir() {
        return mlx_dir;
    }

    fs::create_dir_all(&cache_root)
        .unwrap_or_else(|e| panic!("mkdir {}: {e}", cache_root.display()));

    fetch_mlx_tarball(&sha, &cache_root)
        .unwrap_or_else(|e| panic!("fetch mlx {version} ({sha}): {e}"));

    // Best-effort marker write — if another concurrent build script already
    // wrote it, that's fine. The marker contents are identical (the sha).
    let _ = fs::write(&marker, format!("{sha}\n"));

    mlx_dir
}

/// Walk up from this crate's manifest until a `Cargo.toml` with a
/// `[workspace]` table is found. Panics if none.
fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    loop {
        let toml = dir.join("Cargo.toml");
        if toml.is_file() {
            let body = fs::read_to_string(&toml).unwrap_or_default();
            if body.contains("[workspace]") || body.contains("[workspace.metadata") {
                return dir;
            }
        }
        if !dir.pop() {
            panic!(
                "workspace root not found from {}",
                env::var("CARGO_MANIFEST_DIR").unwrap()
            );
        }
    }
}

/// Parse `[workspace.metadata.mlx]` `version` + `sha` from the
/// workspace root `Cargo.toml`. Tiny ad-hoc reader — avoids a build-
/// time TOML dependency.
fn read_workspace_mlx_pin() -> (String, String) {
    let toml = workspace_root().join("Cargo.toml");
    let body = fs::read_to_string(&toml)
        .unwrap_or_else(|e| panic!("read {}: {e}", toml.display()));
    let mut in_section = false;
    let mut version = None::<String>;
    let mut sha = None::<String>;
    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_section = trimmed == "[workspace.metadata.mlx]";
            continue;
        }
        if !in_section || trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("version") {
            version = Some(parse_quoted_value(rest));
        } else if let Some(rest) = trimmed.strip_prefix("sha") {
            sha = Some(parse_quoted_value(rest));
        }
    }
    let version = version.expect("[workspace.metadata.mlx] missing version");
    let sha = sha.expect("[workspace.metadata.mlx] missing sha");
    (version, sha)
}

/// Extract the inner text of the first `"..."`-quoted value on the
/// right of `=`. Survives trailing inline comments, e.g.
/// `sha = "abc" # last bumped 2026-05`.
fn parse_quoted_value(rest: &str) -> String {
    let after_eq = rest.split_once('=').map(|(_, v)| v).unwrap_or(rest).trim();
    let inner = after_eq.trim_start_matches('"');
    inner
        .split_once('"')
        .map(|(v, _)| v)
        .unwrap_or(inner)
        .to_string()
}

fn fetch_mlx_tarball(sha: &str, cache_root: &Path) -> Result<(), String> {
    let to = cache_root.join("mlx");
    // Re-check inside the function in case another process completed the
    // fetch between our marker check and the call.
    if to.is_dir() {
        return Ok(());
    }

    // Process-unique staging avoids concurrent curl/tar writing to the
    // same paths when two build scripts race on a cold cache.
    let pid = std::process::id();
    let staging = cache_root.join(format!(".staging.{pid}"));
    let _ = fs::remove_dir_all(&staging);
    fs::create_dir_all(&staging)
        .map_err(|e| format!("mkdir staging {}: {e}", staging.display()))?;

    let url = format!("https://codeload.github.com/ml-explore/mlx/tar.gz/{sha}");
    let tarball = staging.join("upstream.tar.gz");
    let curl = Command::new("curl")
        .arg("--fail")
        .arg("--silent")
        .arg("--show-error")
        .arg("--location")
        .arg("--output")
        .arg(&tarball)
        .arg(&url)
        .status()
        .map_err(|e| format!("spawn curl: {e}"))?;
    if !curl.success() {
        let _ = fs::remove_dir_all(&staging);
        return Err(format!("curl exited with {curl}"));
    }
    let tar = Command::new("tar")
        .arg("-xzf")
        .arg(&tarball)
        .arg("-C")
        .arg(&staging)
        .status()
        .map_err(|e| format!("spawn tar: {e}"))?;
    if !tar.success() {
        let _ = fs::remove_dir_all(&staging);
        return Err(format!("tar exited with {tar}"));
    }
    let _ = fs::remove_file(&tarball);

    let from = staging.join(format!("mlx-{sha}"));
    // Atomic rename within the same filesystem. If another process won
    // the race, our rename fails (ENOTEMPTY/EEXIST) — clean up + succeed.
    match fs::rename(&from, &to) {
        Ok(()) => {}
        Err(_) if to.is_dir() => {
            // Another process already placed `mlx/`. Their content is
            // identical (same SHA tarball). Accept their win.
        }
        Err(e) => {
            let _ = fs::remove_dir_all(&staging);
            return Err(format!(
                "rename {} -> {}: {e}",
                from.display(),
                to.display()
            ));
        }
    }
    let _ = fs::remove_dir_all(&staging);
    Ok(())
}

// =================== end shared mlx fetch helper ===================

const STEEL_SRC_REL: &str = "steel/attn/kernels/steel_attention";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=MLX_RS_SRC_DIR");

    let mlx_src = ensure_mlx_src();

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let preamble_path = generate_steel_preamble(&mlx_src, &out_dir);
    println!(
        "cargo:rustc-env=STEEL_ATTENTION_PREAMBLE_PATH={}",
        preamble_path.display()
    );
}

/// Run `make_compiled_preamble.sh` against `steel/attn/kernels/steel_attention.h`,
/// strip the `R"preamble(...)"` envelope, and write bare Metal source to
/// `<out>/steel_attention_preamble.metal`. Returns that path.
fn generate_steel_preamble(mlx_src: &Path, out_dir: &Path) -> PathBuf {
    let script = mlx_src.join("mlx/backend/metal/make_compiled_preamble.sh");
    if !script.is_file() {
        panic!("make_compiled_preamble.sh missing at {}", script.display());
    }

    // The script writes <stage>/steel_attention.cpp from steel/attn/.../steel_attention.h.
    let stage = out_dir.join("steel-preamble-stage");
    let _ = fs::remove_dir_all(&stage);
    fs::create_dir_all(&stage)
        .unwrap_or_else(|e| panic!("mkdir {}: {e}", stage.display()));

    let status = Command::new("bash")
        .arg(&script)
        .arg(&stage)
        .arg("clang") // CC — script only uses basename to compose filenames; xcrun does the real work.
        .arg(mlx_src) // PROJECT_SOURCE_DIR
        .arg(STEEL_SRC_REL)
        .status()
        .unwrap_or_else(|e| panic!("spawn {}: {e}", script.display()));
    if !status.success() {
        panic!(
            "{} exited with {} (need Xcode CLT + `xcrun metal` toolchain)",
            script.display(),
            status
        );
    }

    let cpp = stage.join("steel_attention.cpp");
    let cpp_body = fs::read_to_string(&cpp)
        .unwrap_or_else(|e| panic!("read {}: {e}", cpp.display()));
    let metal = strip_preamble_envelope(&cpp_body);

    let out_path = out_dir.join("steel_attention_preamble.metal");
    fs::write(&out_path, metal)
        .unwrap_or_else(|e| panic!("write {}: {e}", out_path.display()));
    out_path
}

/// Strip the `const char* steel_attention() { return R"preamble( ... )preamble"; }`
/// envelope. Returns the inner Metal source.
fn strip_preamble_envelope(cpp: &str) -> &str {
    let open = cpp
        .find("R\"preamble(")
        .expect("steel_attention.cpp missing R\"preamble(...) opening");
    let start = open + "R\"preamble(".len();
    let close = cpp[start..]
        .rfind(")preamble\"")
        .expect("steel_attention.cpp missing )preamble\" closing");
    cpp[start..start + close].trim_matches('\n').trim_start_matches('\n')
}
