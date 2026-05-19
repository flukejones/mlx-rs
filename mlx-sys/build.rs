#![allow(
    clippy::unwrap_used,
    reason = "build script: panic-on-error is idiomatic"
)]
#![allow(
    clippy::print_stdout,
    reason = "build script: cargo:* directives go to stdout"
)]

use cmake::Config;
use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

// ===================== shared mlx fetch helper =====================
// Identical block in `mlx-lm/build.rs`. Keep them in sync when edited.

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
        assert!(
            path.is_dir(),
            "MLX_RS_SRC_DIR={} is not a directory",
            path.display()
        );
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
        assert!(
            dir.pop(),
            "workspace root not found from {}",
            env::var("CARGO_MANIFEST_DIR").unwrap()
        );
    }
}

/// Parse `[workspace.metadata.mlx]` `version` + `sha` from the
/// workspace root `Cargo.toml`. Tiny ad-hoc reader — avoids a build-
/// time TOML dependency.
fn read_workspace_mlx_pin() -> (String, String) {
    let toml = workspace_root().join("Cargo.toml");
    let body = fs::read_to_string(&toml).unwrap_or_else(|e| panic!("read {}: {e}", toml.display()));
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
        .to_owned()
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

/// Find the clang runtime library path dynamically using xcrun
fn find_clang_rt_path() -> Option<String> {
    // Use xcrun to find the active toolchain path
    let output = Command::new("xcrun")
        .args(["--show-sdk-platform-path"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    // Get the developer directory which contains the toolchain
    let output = Command::new("xcode-select")
        .args(["--print-path"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let developer_dir = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let toolchain_base =
        format!("{developer_dir}/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang");

    // Find the clang version directory (it varies by Xcode version)
    let clang_dir = fs::read_dir(&toolchain_base).ok()?;
    for entry in clang_dir.flatten() {
        let darwin_path = entry.path().join("lib/darwin");
        let clang_rt_lib = darwin_path.join("libclang_rt.osx.a");
        if clang_rt_lib.exists() {
            return Some(darwin_path.to_string_lossy().to_string());
        }
    }

    None
}

fn build_and_link_mlx_c() {
    let mlx_src = ensure_mlx_src();

    let mut config = Config::new("src/mlx-c");
    config.very_verbose(true);
    config.define("CMAKE_INSTALL_PREFIX", ".");
    // Skip cmake's git-clone of mlx; reuse the shared cache.
    config.define(
        "FETCHCONTENT_SOURCE_DIR_MLX",
        mlx_src.to_string_lossy().as_ref(),
    );

    // Use Xcode's clang to ensure compatibility with the macOS SDK
    config.define("CMAKE_C_COMPILER", "/usr/bin/cc");
    config.define("CMAKE_CXX_COMPILER", "/usr/bin/c++");

    #[cfg(debug_assertions)]
    {
        config.define("CMAKE_BUILD_TYPE", "Debug");
    }

    #[cfg(not(debug_assertions))]
    {
        config.define("CMAKE_BUILD_TYPE", "Release");
    }

    config.define("MLX_BUILD_METAL", "OFF");
    config.define("MLX_BUILD_ACCELERATE", "OFF");

    #[cfg(feature = "metal")]
    {
        config.define("MLX_BUILD_METAL", "ON");
    }

    #[cfg(feature = "accelerate")]
    {
        config.define("MLX_BUILD_ACCELERATE", "ON");
    }

    // build the mlx-c project
    let dst = config.build();

    println!("cargo:rustc-link-search=native={}/build/lib", dst.display());
    println!("cargo:rustc-link-lib=static=mlx");
    println!("cargo:rustc-link-lib=static=mlxc");

    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=dylib=objc");
    println!("cargo:rustc-link-lib=framework=Foundation");

    #[cfg(feature = "metal")]
    {
        println!("cargo:rustc-link-lib=framework=Metal");
    }

    #[cfg(feature = "accelerate")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    // Link against Xcode's clang runtime for ___isPlatformVersionAtLeast symbol
    // This is needed on macOS 26+ where the bundled LLVM runtime may be outdated
    // See: https://github.com/conda-forge/llvmdev-feedstock/issues/244
    if let Some(clang_rt_path) = find_clang_rt_path() {
        println!("cargo:rustc-link-search={clang_rt_path}");
        println!("cargo:rustc-link-lib=static=clang_rt.osx");
    }
}

fn main() {
    build_and_link_mlx_c();

    // generate bindings
    let bindings = bindgen::Builder::default()
        .rust_target("1.73.0".parse().expect("rust-version"))
        .header("src/mlx-c/mlx/c/mlx.h")
        .header("src/mlx-c/mlx/c/linalg.h")
        .header("src/mlx-c/mlx/c/error.h")
        .header("src/mlx-c/mlx/c/transforms_impl.h")
        .clang_arg("-Isrc/mlx-c")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
