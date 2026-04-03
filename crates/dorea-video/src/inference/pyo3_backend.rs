//! PyO3-based inference backend — runs Python models in-process via embedded Python.
//!
//! This module provides the same public API as `inference_subprocess` but avoids
//! the subprocess JSON-lines IPC overhead by calling Python directly through PyO3.
//! GPU tensors can be shared zero-copy via `DepthTensorGuard`.

use std::marker::PhantomData;
use std::path::PathBuf;
use std::time::Duration;
use thiserror::Error;

use pyo3::prelude::*;
use pyo3::conversion::ToPyObject;
use pyo3::types::PyModule;

/// Errors from the PyO3 inference backend.
#[derive(Debug, Error)]
pub enum InferenceError {
    #[error("failed to spawn inference server: {0}")]
    SpawnFailed(#[from] std::io::Error),
    #[error("IPC error: {0}")]
    Ipc(String),
    #[error("inference server error: {0}")]
    ServerError(String),
    #[error("timeout waiting for inference server")]
    Timeout,
    #[error("PNG encode/decode error: {0}")]
    ImageError(String),
    #[error("CUDA OOM during inference: {0}")]
    Oom(String),
    #[error("PyO3 initialization failed: {0}")]
    InitFailed(String),
}

/// Configuration for the inference backend (same fields as subprocess version).
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Python executable path (e.g. `/opt/dorea-venv/bin/python`).
    pub python_exe: PathBuf,
    /// Path to the RAUNE-Net weights .pth file.
    pub raune_weights: Option<PathBuf>,
    /// Path to the sea_thru_poc directory (contains models/raune_net.py).
    pub raune_models_dir: Option<PathBuf>,
    /// Skip RAUNE-Net entirely.
    pub skip_raune: bool,
    /// Path to Depth Anything V2 model directory.
    pub depth_model: Option<PathBuf>,
    /// Compute device: "cpu" or "cuda".
    pub device: Option<String>,
    /// Startup timeout (unused for PyO3, kept for API compat).
    pub startup_timeout: Duration,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            python_exe: PathBuf::from("/opt/dorea-venv/bin/python"),
            raune_weights: None,
            raune_models_dir: None,
            skip_raune: false,
            depth_model: None,
            device: None,
            startup_timeout: Duration::from_secs(120),
        }
    }
}

/// Guard that prevents Python's GC from reclaiming a GPU tensor while Rust
/// holds the device pointer. Drop this guard when Rust is done with the pointer.
pub struct DepthTensorGuard {
    py_guard: Py<PyAny>,
    /// Raw CUDA device pointer to the tensor data.
    pub data_ptr: usize,
    /// Number of elements in the tensor.
    pub numel: usize,
    /// Width of the depth map.
    pub width: usize,
    /// Height of the depth map.
    pub height: usize,
}

impl DepthTensorGuard {
    /// Explicitly release the underlying Python TensorGuard, freeing the GPU tensor.
    pub fn release(self) {
        Python::with_gil(|py| {
            let _ = self.py_guard.call_method0(py, "release");
        });
    }
}

/// In-process Python inference server using PyO3 embedding.
///
/// This struct is `!Send + !Sync` because Python's GIL requires single-threaded
/// access patterns. All calls must happen on the thread that created the server.
pub struct InferenceServer {
    _not_send: PhantomData<*const ()>,
}

impl InferenceServer {
    /// Initialize the embedded Python interpreter and load models.
    ///
    /// 1. Calls `pyo3::prepare_freethreaded_python()` to init Python.
    /// 2. Adds the `python/` dir to `sys.path`.
    /// 3. Imports `dorea_inference.bridge`.
    /// 4. Loads depth and (optionally) RAUNE models.
    pub fn spawn(config: &InferenceConfig) -> Result<Self, InferenceError> {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Add python/ dir to sys.path so `dorea_inference` is importable.
            let python_dir = Self::find_python_dir();
            if let Some(ref p) = python_dir {
                let sys = py.import_bound("sys")
                    .map_err(|e| InferenceError::InitFailed(format!("import sys: {e}")))?;
                let path = sys.getattr("path")
                    .map_err(|e| InferenceError::InitFailed(format!("sys.path: {e}")))?;
                let p_str = p.to_str().unwrap_or("");
                path.call_method1("insert", (0, p_str))
                    .map_err(|e| InferenceError::InitFailed(format!("sys.path.insert: {e}")))?;
            } else {
                log::debug!(
                    "Could not derive python/ dir from binary path; \
                     ensure dorea_inference is installed in the Python environment"
                );
            }

            // Import the bridge module.
            let bridge = py.import_bound("dorea_inference.bridge")
                .map_err(|e| InferenceError::InitFailed(format!("import dorea_inference.bridge: {e}")))?;

            // Load depth model.
            let device = config.device.as_deref().unwrap_or("cuda");
            let depth_path = config.depth_model.as_ref().map(|p| p.to_str().unwrap_or(""));

            Self::call_load_depth(py, &bridge, depth_path, device)?;

            // Load RAUNE model if not skipped.
            if !config.skip_raune {
                let weights = config.raune_weights.as_ref().map(|p| p.to_str().unwrap_or(""));
                let models_dir = config.raune_models_dir.as_ref().map(|p| p.to_str().unwrap_or(""));
                Self::call_load_raune(py, &bridge, weights, device, models_dir)?;
            }

            Ok(Self {
                _not_send: PhantomData,
            })
        })
    }

    /// Run Depth Anything V2 on an RGB image.
    ///
    /// `image_rgb`: HxWx3 u8 array flattened row-major.
    /// Returns `(depth_f32, out_width, out_height)` at inference resolution.
    pub fn run_depth(
        &mut self,
        _id: &str,
        image_rgb: &[u8],
        width: usize,
        height: usize,
        max_size: usize,
    ) -> Result<(Vec<f32>, usize, usize), InferenceError> {
        Python::with_gil(|py| {
            let bridge = py.import_bound("dorea_inference.bridge")
                .map_err(|e| InferenceError::Ipc(format!("import bridge: {e}")))?;

            // Create numpy array from raw RGB bytes.
            let np_flat = numpy::PyArray1::from_slice_bound(py, image_rgb);
            let np_reshaped = np_flat.call_method1("reshape", ((height, width, 3),))
                .map_err(|e| InferenceError::Ipc(format!("reshape: {e}")))?;

            // Call bridge.run_depth_cpu(reshaped, max_size).
            let result = bridge.call_method1("run_depth_cpu", (np_reshaped, max_size))
                .map_err(|e| Self::map_python_error(py, e))?;

            // Result is a numpy array of shape (h, w) with float32 values.
            let shape: Vec<usize> = result.getattr("shape")
                .map_err(|e| InferenceError::Ipc(format!("get shape: {e}")))?
                .extract()
                .map_err(|e| InferenceError::Ipc(format!("extract shape: {e}")))?;

            let out_h = shape[0];
            let out_w = shape[1];

            // Flatten and extract as Vec<f32>.
            let flat = result.call_method1("reshape", ((-1,),))
                .map_err(|e| InferenceError::Ipc(format!("flatten result: {e}")))?;
            let depth_data: Vec<f32> = flat.call_method0("tolist")
                .map_err(|e| InferenceError::Ipc(format!("tolist: {e}")))?
                .extract()
                .map_err(|e| InferenceError::Ipc(format!("extract depth: {e}")))?;

            Ok((depth_data, out_w, out_h))
        })
    }

    /// Run RAUNE-Net on an RGB image.
    ///
    /// `image_rgb`: HxWx3 u8 array flattened row-major.
    /// Returns the enhanced image as `(rgb_u8, out_width, out_height)`.
    pub fn run_raune(
        &mut self,
        _id: &str,
        image_rgb: &[u8],
        width: usize,
        height: usize,
        max_size: usize,
    ) -> Result<(Vec<u8>, usize, usize), InferenceError> {
        Python::with_gil(|py| {
            let bridge = py.import_bound("dorea_inference.bridge")
                .map_err(|e| InferenceError::Ipc(format!("import bridge: {e}")))?;

            // Create numpy array from raw RGB bytes.
            let np_flat = numpy::PyArray1::from_slice_bound(py, image_rgb);
            let np_reshaped = np_flat.call_method1("reshape", ((height, width, 3),))
                .map_err(|e| InferenceError::Ipc(format!("reshape: {e}")))?;

            // Call bridge.run_raune_cpu(reshaped, max_size).
            let result = bridge.call_method1("run_raune_cpu", (np_reshaped, max_size))
                .map_err(|e| Self::map_python_error(py, e))?;

            // Result is a numpy array of shape (h, w, 3) with uint8 values.
            let shape: Vec<usize> = result.getattr("shape")
                .map_err(|e| InferenceError::Ipc(format!("get shape: {e}")))?
                .extract()
                .map_err(|e| InferenceError::Ipc(format!("extract shape: {e}")))?;

            let out_h = shape[0];
            let out_w = shape[1];

            // Flatten and extract as Vec<u8>.
            let flat = result.call_method1("reshape", ((-1,),))
                .map_err(|e| InferenceError::Ipc(format!("flatten result: {e}")))?;
            let rgb_data: Vec<u8> = flat.call_method0("tolist")
                .map_err(|e| InferenceError::Ipc(format!("tolist: {e}")))?
                .extract()
                .map_err(|e| InferenceError::Ipc(format!("extract rgb: {e}")))?;

            Ok((rgb_data, out_w, out_h))
        })
    }

    /// Run depth inference on the GPU, returning a guard that holds the on-device tensor.
    ///
    /// The tensor stays on the GPU until the guard is released or dropped,
    /// enabling zero-copy sharing with CUDA kernels.
    pub fn run_depth_gpu(
        &mut self,
        _id: &str,
        image_rgb: &[u8],
        width: usize,
        height: usize,
        max_size: usize,
    ) -> Result<DepthTensorGuard, InferenceError> {
        Python::with_gil(|py| {
            let bridge = py.import_bound("dorea_inference.bridge")
                .map_err(|e| InferenceError::Ipc(format!("import bridge: {e}")))?;

            let np_flat = numpy::PyArray1::from_slice_bound(py, image_rgb);
            let np_reshaped = np_flat.call_method1("reshape", ((height, width, 3),))
                .map_err(|e| InferenceError::Ipc(format!("reshape: {e}")))?;

            let guard_py = bridge.call_method1("run_depth_gpu", (np_reshaped, max_size))
                .map_err(|e| Self::map_python_error(py, e))?;

            let data_ptr: usize = guard_py.getattr("data_ptr")
                .map_err(|e| InferenceError::Ipc(format!("get data_ptr: {e}")))?
                .extract()
                .map_err(|e| InferenceError::Ipc(format!("extract data_ptr: {e}")))?;
            let numel: usize = guard_py.getattr("numel")
                .map_err(|e| InferenceError::Ipc(format!("get numel: {e}")))?
                .extract()
                .map_err(|e| InferenceError::Ipc(format!("extract numel: {e}")))?;
            let shape: Vec<usize> = guard_py.getattr("shape")
                .map_err(|e| InferenceError::Ipc(format!("get shape: {e}")))?
                .extract()
                .map_err(|e| InferenceError::Ipc(format!("extract shape: {e}")))?;

            let guard_height = shape.first().copied().unwrap_or(0);
            let guard_width = shape.get(1).copied().unwrap_or(0);

            Ok(DepthTensorGuard {
                py_guard: guard_py.unbind(),
                data_ptr,
                numel,
                width: guard_width,
                height: guard_height,
            })
        })
    }

    /// Bilinearly upscale a depth map from (src_w, src_h) to (dst_w, dst_h).
    pub fn upscale_depth(
        depth: &[f32],
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0_f32; dst_w * dst_h];
        for dy in 0..dst_h {
            for dx in 0..dst_w {
                let sx = dx as f32 * (src_w as f32 - 1.0) / (dst_w as f32 - 1.0).max(1.0);
                let sy = dy as f32 * (src_h as f32 - 1.0) / (dst_h as f32 - 1.0).max(1.0);
                let x0 = sx.floor() as usize;
                let y0 = sy.floor() as usize;
                let x1 = (x0 + 1).min(src_w - 1);
                let y1 = (y0 + 1).min(src_h - 1);
                let fx = sx - x0 as f32;
                let fy = sy - y0 as f32;

                let v00 = depth[y0 * src_w + x0];
                let v10 = depth[y0 * src_w + x1];
                let v01 = depth[y1 * src_w + x0];
                let v11 = depth[y1 * src_w + x1];

                out[dy * dst_w + dx] = v00 * (1.0 - fx) * (1.0 - fy)
                    + v10 * fx * (1.0 - fy)
                    + v01 * (1.0 - fx) * fy
                    + v11 * fx * fy;
            }
        }
        out
    }

    /// Query free VRAM in bytes (calls Python bridge.vram_free_bytes()).
    pub fn vram_free_bytes(&self) -> Result<usize, InferenceError> {
        Python::with_gil(|py| {
            let bridge = py.import_bound("dorea_inference.bridge")
                .map_err(|e| InferenceError::Ipc(format!("import bridge: {e}")))?;
            let result = bridge.call_method0("vram_free_bytes")
                .map_err(|e| InferenceError::Ipc(format!("vram_free_bytes: {e}")))?;
            let bytes: usize = result.extract()
                .map_err(|e| InferenceError::Ipc(format!("extract vram_free_bytes: {e}")))?;
            Ok(bytes)
        })
    }

    /// Graceful shutdown — no subprocess to kill, just return Ok.
    pub fn shutdown(self) -> Result<(), InferenceError> {
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Find the `python/` directory adjacent to the binary (same logic as subprocess version).
    fn find_python_dir() -> Option<PathBuf> {
        std::env::current_exe().ok().and_then(|exe| {
            // .parent() = target/debug|release, .parent() = target, .parent() = workspace root
            exe.parent()
                .and_then(|p| p.parent())
                .and_then(|p| p.parent())
                .map(|root| root.join("python"))
        }).filter(|p| p.exists())
    }

    /// Load the depth model via the Python bridge.
    fn call_load_depth(
        py: Python<'_>,
        bridge: &Bound<'_, PyModule>,
        model_path: Option<&str>,
        device: &str,
    ) -> Result<(), InferenceError> {
        let py_model_path = match model_path {
            Some(p) => p.to_object(py),
            None => py.None(),
        };
        bridge.call_method1("load_depth_model", (py_model_path, device))
            .map_err(|e| InferenceError::InitFailed(format!("load_depth_model: {e}")))?;
        Ok(())
    }

    /// Load the RAUNE model via the Python bridge.
    fn call_load_raune(
        py: Python<'_>,
        bridge: &Bound<'_, PyModule>,
        weights_path: Option<&str>,
        device: &str,
        models_dir: Option<&str>,
    ) -> Result<(), InferenceError> {
        let py_weights = match weights_path {
            Some(p) => p.to_object(py),
            None => py.None(),
        };
        let py_models_dir = match models_dir {
            Some(p) => p.to_object(py),
            None => py.None(),
        };
        bridge.call_method1("load_raune_model", (py_weights, device, py_models_dir))
            .map_err(|e| InferenceError::InitFailed(format!("load_raune_model: {e}")))?;
        Ok(())
    }

    /// Map a Python exception to the appropriate InferenceError variant.
    /// Detects CUDA OOM specifically for better error reporting.
    fn map_python_error(py: Python<'_>, err: PyErr) -> InferenceError {
        // Try to detect CUDA OOM errors.
        // torch.cuda.OutOfMemoryError is a subclass of RuntimeError.
        let msg = format!("{err}");
        if msg.contains("OutOfMemoryError") || msg.contains("CUDA out of memory") {
            return InferenceError::Oom(msg);
        }

        // Check if it's a torch.cuda.OutOfMemoryError by trying to import and check.
        let is_oom = (|| -> Result<bool, PyErr> {
            let torch = py.import_bound("torch")?;
            let cuda = torch.getattr("cuda")?;
            let oom_type = cuda.getattr("OutOfMemoryError")?;
            let err_value = err.value_bound(py);
            Ok(err_value.is_instance(&oom_type)?)
        })();

        if matches!(is_oom, Ok(true)) {
            return InferenceError::Oom(msg);
        }

        InferenceError::ServerError(msg)
    }
}
