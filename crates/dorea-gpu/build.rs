// Build script for dorea-gpu.
// Detects nvcc and, if found, compiles CUDA kernels and enables the "cuda" Cargo feature.
// If nvcc is not found, builds with CPU-only fallback.

use std::path::PathBuf;

fn find_nvcc() -> Option<PathBuf> {
    // 1. Check PATH — resolve to full path via `which` so parent-based include lookup works
    if let Ok(output) = std::process::Command::new("nvcc").arg("--version").output() {
        if output.status.success() {
            if let Ok(which_out) = std::process::Command::new("which").arg("nvcc").output() {
                if which_out.status.success() {
                    let resolved = PathBuf::from(
                        std::str::from_utf8(&which_out.stdout).unwrap_or("").trim()
                    );
                    if resolved.exists() {
                        return Some(resolved);
                    }
                }
            }
            // Fallback: nvcc is on PATH but `which` unavailable; return relative name
            return Some(PathBuf::from("nvcc"));
        }
    }

    // 2. Check CUDA_HOME/bin/nvcc
    if let Ok(cuda_home) = std::env::var("CUDA_HOME") {
        let nvcc = PathBuf::from(cuda_home).join("bin").join("nvcc");
        if nvcc.exists() {
            return Some(nvcc);
        }
    }

    // 3. Scan /usr/local/cuda-* (any version) — handles e.g. cuda-12.8, cuda-12.4
    if let Ok(entries) = std::fs::read_dir("/usr/local") {
        let mut cuda_dirs: Vec<_> = entries
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with("cuda-"))
            .map(|e| e.path().join("bin").join("nvcc"))
            .filter(|p| p.exists())
            .collect();
        // Prefer highest version (sort descending by path string)
        cuda_dirs.sort_by(|a, b| b.cmp(a));
        if let Some(p) = cuda_dirs.into_iter().next() {
            return Some(p);
        }
    }

    // 4. Remaining common install paths
    for candidate in &[
        "/usr/local/cuda/bin/nvcc",
        "/usr/bin/nvcc",
    ] {
        let p = PathBuf::from(candidate);
        if p.exists() {
            return Some(p);
        }
    }

    None
}

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernels_dir = manifest_dir.join("src").join("cuda").join("kernels");

    // Tell cargo to re-run if any .cu file changes or CUDA env vars change
    println!("cargo:rerun-if-changed=src/cuda/kernels/lut_apply.cu");
    println!("cargo:rerun-if-changed=src/cuda/kernels/hsl_correct.cu");
    println!("cargo:rerun-if-changed=src/cuda/kernels/clarity.cu");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=PATH");

    let Some(nvcc) = find_nvcc() else {
        println!("cargo:warning=nvcc not found — building dorea-gpu with CPU-only fallback");
        println!("cargo:warning=Rebuild the devcontainer (Dockerfile adds CUDA 12.4 toolkit) to enable CUDA kernels");
        return;
    };

    println!("cargo:warning=Found nvcc at {}", nvcc.display());

    // Compile each .cu file to an object file, then link.
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let kernel_names = ["lut_apply", "hsl_correct", "clarity"];
    let mut obj_files: Vec<PathBuf> = Vec::new();

    for name in &kernel_names {
        let src = kernels_dir.join(format!("{name}.cu"));
        let obj = out_dir.join(format!("{name}.o"));

        // Make CUDA includes take priority over system includes so that
        // CUDA's crt/math_functions.h declarations (e.g. cospi) don't conflict
        // with glibc 2.35+ mathcalls.h redeclaring them with noexcept(true).
        let cuda_include = nvcc.parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("targets/x86_64-linux/include"))
            .filter(|p| p.exists());
        let mut args = vec![
            "-c".to_string(),
            "-O2".to_string(),
            "-arch=sm_86".to_string(), // RTX 3060 is Ampere sm_86
            "--compiler-options=-fPIC".to_string(),
            "--allow-unsupported-compiler".to_string(), // GCC 14 > officially supported max (13)
            // Use gcc-12 from Debian bookworm as host compiler (GCC 14 is incompatible with CUDA 12.4)
            "--compiler-bindir".to_string(),
            "/usr/bin/gcc-12".to_string(),
        ];
        if let Some(ref inc) = cuda_include {
            args.push("-isystem".to_string());
            args.push(inc.to_str().unwrap().to_string());
        }
        args.push(src.to_str().unwrap().to_string());
        args.push("-o".to_string());
        args.push(obj.to_str().unwrap().to_string());

        let status = std::process::Command::new(&nvcc)
            .args(&args)
            .status()
            .expect("failed to run nvcc");

        if !status.success() {
            println!(
                "cargo:warning=nvcc failed to compile {name}.cu — \
                 falling back to CPU-only (GCC/CUDA version mismatch?). \
                 Install gcc-12 or gcc-13 and set CUDAHOSTCXX to enable CUDA kernels."
            );
            return;
        }
        obj_files.push(obj);
    }

    // Archive into a static lib
    let lib_path = out_dir.join("libdorea_cuda_kernels.a");
    let mut ar = std::process::Command::new("ar");
    ar.arg("rcs").arg(&lib_path);
    for obj in &obj_files {
        ar.arg(obj);
    }
    ar.status().expect("failed to run ar");

    // Add CUDA lib directory to linker search path.
    // nvcc lives at <cuda_root>/bin/nvcc; lib is at <cuda_root>/targets/x86_64-linux/lib/.
    if let Some(cuda_lib) = nvcc.parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("targets/x86_64-linux/lib"))
        .filter(|p| p.exists())
    {
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    }
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=dorea_cuda_kernels");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");

    // Signal to Cargo to enable the "cuda" feature
    println!("cargo:rustc-cfg=feature=\"cuda\"");
}
