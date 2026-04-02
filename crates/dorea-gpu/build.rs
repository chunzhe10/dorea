// Build script for dorea-gpu.
// Detects nvcc and, if found, compiles CUDA kernels and enables the "cuda" Cargo feature.
// If nvcc is not found, builds with CPU-only fallback.

use std::path::PathBuf;

fn find_nvcc() -> Option<PathBuf> {
    // 1. Check PATH
    if let Ok(output) = std::process::Command::new("nvcc").arg("--version").output() {
        if output.status.success() {
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

    // 3. Common install paths
    for candidate in &[
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-12.4/bin/nvcc",
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

        let status = std::process::Command::new(&nvcc)
            .args([
                "-c",
                "-O2",
                "-arch=native", // target the GPU actually present at compile time
                "--compiler-options=-fPIC",
                src.to_str().unwrap(),
                "-o",
                obj.to_str().unwrap(),
            ])
            .status()
            .expect("failed to run nvcc");

        if !status.success() {
            panic!("nvcc failed to compile {name}.cu");
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

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=dorea_cuda_kernels");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");

    // Signal to Cargo to enable the "cuda" feature
    println!("cargo:rustc-cfg=feature=\"cuda\"");
}
