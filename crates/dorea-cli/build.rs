// Build script for dorea-cli.
// Mirrors the nvcc detection in dorea-gpu/build.rs and emits the same
// `cfg(feature = "cuda")` flag so that `#[cfg(feature = "cuda")]` blocks in
// grade.rs are activated consistently with the dorea-gpu crate.

use std::path::PathBuf;

fn nvcc_available() -> bool {
    // 1. Check PATH
    if let Ok(output) = std::process::Command::new("nvcc").arg("--version").output() {
        if output.status.success() {
            return true;
        }
    }

    // 2. Check CUDA_HOME/bin/nvcc
    if let Ok(cuda_home) = std::env::var("CUDA_HOME") {
        let nvcc = PathBuf::from(cuda_home).join("bin").join("nvcc");
        if nvcc.exists() {
            return true;
        }
    }

    // 3. Scan /usr/local/cuda-*
    if let Ok(entries) = std::fs::read_dir("/usr/local") {
        let found = entries
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with("cuda-"))
            .any(|e| e.path().join("bin").join("nvcc").exists());
        if found {
            return true;
        }
    }

    // 4. Common fixed paths
    for candidate in &["/usr/local/cuda/bin/nvcc", "/usr/bin/nvcc"] {
        if PathBuf::from(candidate).exists() {
            return true;
        }
    }

    false
}

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=PATH");

    if nvcc_available() {
        println!("cargo:warning=Found nvcc — enabling CUDA grader reuse in dorea-cli");
        println!("cargo:rustc-cfg=feature=\"cuda\"");
    }
}
