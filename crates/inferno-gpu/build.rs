//! Build the vendored ggml (llama.cpp compute core) into a static lib.
//!
//! Pinned upstream: ggml-org/llama.cpp v0.8.0, commit
//! 8642541b255bfb5dc39cd351a178be09013a55d6 (MIT) — see
//! third_party/ggml/LICENSE and third_party/VENDORED.md.
//!
//! File selection is by GLOB, not by name: upstream renames and splits
//! `.c`/`.cpp` between pins, and a hardcoded list broke the build exactly
//! that way once. Globbing makes the vendored build pin-agnostic.

use std::path::Path;

fn main() {
    // ggml-version.h is CMake-generated upstream (src/ggml-version.h.in);
    // generate it here from the pinned upstream so the vendored tree stays
    // pristine and the pin's identity is compiled in.
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let version_h = format!(
        "#pragma once\n#define GGML_VERSION \"{ver}\"\n#define GGML_COMMIT  \"{commit}\"\n",
        ver = "0.8.0",
        commit = "8642541b255bfb5dc39cd351a178be09013a55d6",
    );
    std::fs::write(Path::new(&out_dir).join("ggml-version.h"), version_h).unwrap();

    let ggml_dir = Path::new("../../third_party/ggml");
    let src = ggml_dir.join("src");
    let inc = ggml_dir.join("include");

    for p in [&src, &inc, &ggml_dir.join("LICENSE")] {
        assert!(p.exists(), "vendored ggml is missing: {}", p.display());
        println!("cargo:rerun-if-changed={}", p.display());
    }

    let sources = collect_sources(&src);
    assert!(
        sources.len() >= 20,
        "expected the vendored ggml sources to be present; found {}",
        sources.len()
    );

    let mut build = cc::Build::new();
    build
        .include(&out_dir)
        .include(&inc)
        .include(&src)
        // ggml-cpu's own headers (e.g. ggml-cpu-impl.h) are included by
        // relative name from its sources.
        .include(src.join("ggml-cpu"))
        .define("_GNU_SOURCE", None) // SCHED_BATCH & friends in <sched.h>
        // NOTE: GGML_USE_LLAMAFILE is deliberately NOT defined: it selects
        // the llamafile sgemm path, whose sources are not vendored. The
        // reference + arch-vectorized path is what this slice compiles;
        // llamafile is a later pre-registered optimization.
        .warnings(false)
        // gnu11 (not strict c11): ggml needs POSIX/GNU declarations
        // (CLOCK_MONOTONIC, M_PI).
        .flag_if_supported("-std=gnu11")
        .flag_if_supported("-O3");
    for f in &sources {
        build.file(f);
    }

    build.compile("ggml");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-search=native={out_dir}");
    // ggml's cpu backend is C++ (operator new, __cxa_guards) — the final
    // link needs the C++ runtime.
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

/// Every .c/.cpp in ggml/src, plus the cpu backend's own files and the
/// HOST-arch arch/ subdir (arch/x86 on x86_64, arch/arm64 on aarch64) —
/// upstream only builds the arch dirs for the arch it targets, and ARM-only
/// files (amx/, ggml-cpu-aarch64.cpp is #ifdef-guarded and harmless) break
/// an x86 compile. Backend trees are not vendored (VENDORED.md).
fn collect_sources(src: &Path) -> Vec<std::path::PathBuf> {
    fn add_dir(out: &mut Vec<std::path::PathBuf>, dir: &Path) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        let mut paths: Vec<std::path::PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.is_file()
                    && matches!(
                        p.extension().and_then(|e| e.to_str()),
                        Some("c") | Some("cpp")
                    )
            })
            .collect();
        paths.sort();
        out.extend(paths);
    }

    let mut out = Vec::new();
    add_dir(&mut out, src); // core files
    let cpu = src.join("ggml-cpu");
    add_dir(&mut out, &cpu); // cpu backend core
    // subdirs the classic CPU path needs: the tiled matmul (called
    // unconditionally by ggml-cpu.c) and the host-arch arch dir
    add_dir(&mut out, &cpu.join("tiled"));
    let host_arch = if cfg!(target_arch = "x86_64") {
        "x86"
    } else if cfg!(target_arch = "aarch64") {
        "arm64"
    } else {
        "x86"
    };
    let arch_dir = cpu.join("arch").join(host_arch);
    if arch_dir.exists() {
        add_dir(&mut out, &arch_dir);
    }
    out.sort();
    out
}
