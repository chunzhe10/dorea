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
use pyo3::types::PyModule;

/// Errors from the PyO3 inference backend.
#[derive(Debug, Error)]
pub enum InferenceError {
    #[error("IPC error: {0}")]
    Ipc(String),
    #[error("inference server error: {0}")]
    ServerError(String),
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
    /// Enable Maxine enhancement.
    pub maxine: bool,
    /// Maxine super-resolution upscale factor (default 2).
    pub maxine_upscale_factor: u32,
    /// Skip Depth Anything at spawn (load on demand via load_depth()).
    pub skip_depth: bool,
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
            maxine: false,
            maxine_upscale_factor: 2,
            skip_depth: false,
        }
    }
}

/// A single image item for fused RAUNE + Depth batch inference.
pub struct RauneDepthBatchItem {
    pub id: String,
    /// Raw RGB24 pixels, row-major.
    pub pixels: Vec<u8>,
    pub width: usize,
    pub height: usize,
    /// Max long-edge for RAUNE resize (pixels).
    pub raune_max_size: usize,
    /// Max long-edge for depth resize (pixels, must be ≤518 for Depth Anything).
    pub depth_max_size: usize,
}

/// A single image item for batch depth inference.
pub struct DepthBatchItem {
    /// Unique identifier (e.g. "kf_000042") returned in the response for correlation.
    pub id: String,
    /// Raw RGB24 pixels, row-major.
    pub pixels: Vec<u8>,
    pub width: usize,
    pub height: usize,
    /// Max dimension for depth model resize (passed to Python as `max_size`).
    pub max_size: usize,
}

/// A single image item for YOLO-seg batch inference.
pub struct YoloSegBatchItem {
    pub id: String,
    pub pixels: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

/// Guard that prevents Python's GC from reclaiming a GPU tensor while Rust
/// holds the device pointer. Drop this guard when Rust is done with the pointer.
pub struct DepthTensorGuard {
    pub(crate) py_guard: Py<PyAny>,
    /// Raw CUDA device pointer to the tensor data.
    pub data_ptr: usize,
    /// Number of elements in the tensor.
    pub numel: usize,
    /// Width of the depth map.
    pub width: usize,
    /// Height of the depth map.
    pub height: usize,
    _not_send: std::marker::PhantomData<*const ()>,
}

impl DepthTensorGuard {
    /// Explicitly release the underlying Python TensorGuard, freeing the GPU tensor early.
    /// Dropping the guard also calls release automatically via the `Drop` impl.
    pub fn release(self) {
        // Drop triggers the Drop impl which calls release().
        drop(self);
    }
}

impl Drop for DepthTensorGuard {
    fn drop(&mut self) {
        Python::with_gil(|py| {
            if let Err(e) = self.py_guard.call_method0(py, "release") {
                log::warn!("TensorGuard.release() failed during drop: {e}");
            }
        });
    }
}

/// In-process Python inference server using PyO3 embedding.
///
/// This struct is `!Send + !Sync` because Python's GIL requires single-threaded
/// access patterns. All calls must happen on the thread that created the server.
pub struct InferenceServer {
    /// Cached bridge module — avoids re-importing on every inference call.
    bridge: Py<PyModule>,
    /// Device string captured at spawn time (e.g. "cuda" or "cpu").
    device: String,
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
                let already_present = path
                    .call_method1("__contains__", (p_str,))
                    .map_err(|e| InferenceError::InitFailed(format!("sys.path.__contains__: {e}")))?
                    .is_truthy()
                    .map_err(|e| InferenceError::InitFailed(format!("sys.path.__contains__ truthy: {e}")))?;
                if !already_present {
                    path.call_method1("insert", (0, p_str))
                        .map_err(|e| InferenceError::InitFailed(format!("sys.path.insert: {e}")))?;
                }
            } else {
                log::debug!(
                    "Could not derive python/ dir from binary path; \
                     ensure dorea_inference is installed in the Python environment"
                );
            }

            // Import the bridge module.
            let bridge = py.import_bound("dorea_inference.bridge")
                .map_err(|e| InferenceError::InitFailed(format!("import dorea_inference.bridge: {e}")))?;

            // Load depth model (unless skipped).
            let device = config.device.as_deref().unwrap_or("cuda");
            let depth_path = config.depth_model.as_ref().map(|p| p.to_str().unwrap_or(""));

            if !config.skip_depth {
                Self::call_load_depth(py, &bridge, depth_path, device)?;
            }

            // Load RAUNE model if not skipped.
            if !config.skip_raune {
                let weights = config.raune_weights.as_ref().map(|p| p.to_str().unwrap_or(""));
                let models_dir = config.raune_models_dir.as_ref().map(|p| p.to_str().unwrap_or(""));
                Self::call_load_raune(py, &bridge, weights, device, models_dir)?;
            }

            // Load Maxine if requested.
            if config.maxine {
                Self::call_load_maxine(py, &bridge, config.maxine_upscale_factor, device)?;
            }

            Ok(Self {
                bridge: bridge.unbind(),
                device: device.to_string(),
                _not_send: PhantomData,
            })
        })
    }

    /// Run Depth Anything V2 on an RGB image.
    ///
    /// `image_rgb`: HxWx3 u8 array flattened row-major.
    /// Returns `(depth_f32, out_width, out_height)` at inference resolution.
    pub fn run_depth(
        &self,
        _id: &str,
        image_rgb: &[u8],
        width: usize,
        height: usize,
        max_size: usize,
    ) -> Result<(Vec<f32>, usize, usize), InferenceError> {
        Python::with_gil(|py| {
            let bridge = self.bridge.bind(py);

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
        &self,
        _id: &str,
        image_rgb: &[u8],
        width: usize,
        height: usize,
        max_size: usize,
    ) -> Result<(Vec<u8>, usize, usize), InferenceError> {
        Python::with_gil(|py| {
            let bridge = self.bridge.bind(py);

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
        &self,
        _id: &str,
        image_rgb: &[u8],
        width: usize,
        height: usize,
        max_size: usize,
    ) -> Result<DepthTensorGuard, InferenceError> {
        Python::with_gil(|py| {
            let bridge = self.bridge.bind(py);

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
                _not_send: std::marker::PhantomData,
            })
        })
    }

    /// Run Depth Anything V2 on a batch of images in a single Python call.
    ///
    /// Returns `Vec<(id, depth_f32, out_width, out_height)>` in the same order as `items`.
    #[allow(clippy::type_complexity)]
    pub fn run_depth_batch(
        &self,
        items: &[DepthBatchItem],
    ) -> Result<Vec<(String, Vec<f32>, usize, usize)>, InferenceError> {
        if items.is_empty() {
            return Ok(vec![]);
        }

        let max_size = items[0].max_size;

        Python::with_gil(|py| {
            let bridge = self.bridge.bind(py);

            // Build Python list of (H, W, 3) numpy arrays
            let py_list = pyo3::types::PyList::empty_bound(py);
            for item in items {
                let np_flat = numpy::PyArray1::from_slice_bound(py, &item.pixels);
                let np_reshaped = np_flat
                    .call_method1("reshape", ((item.height, item.width, 3),))
                    .map_err(|e| InferenceError::Ipc(format!("reshape for id={}: {e}", item.id)))?;
                py_list
                    .append(np_reshaped)
                    .map_err(|e| InferenceError::Ipc(format!("list append for id={}: {e}", item.id)))?;
            }

            // Call bridge.run_depth_batch_cpu(imgs, max_size)
            let result = bridge
                .call_method1("run_depth_batch_cpu", (py_list, max_size))
                .map_err(|e| Self::map_python_error(py, e))?;

            // Result is a Python list of (H, W) float32 numpy arrays
            let result_list = result
                .downcast::<pyo3::types::PyList>()
                .map_err(|e| InferenceError::Ipc(format!("run_depth_batch_cpu result not a list: {e}")))?;

            let mut out = Vec::with_capacity(items.len());
            for (i, item) in items.iter().enumerate() {
                let arr = result_list
                    .get_item(i)
                    .map_err(|e| InferenceError::Ipc(format!("get result[{i}] for id={}: {e}", item.id)))?;

                let shape: Vec<usize> = arr
                    .getattr("shape")
                    .map_err(|e| InferenceError::Ipc(format!("get shape[{i}]: {e}")))?
                    .extract()
                    .map_err(|e| InferenceError::Ipc(format!("extract shape[{i}]: {e}")))?;

                let out_h = shape[0];
                let out_w = shape[1];

                let flat = arr
                    .call_method1("reshape", ((-1_i32,),))
                    .map_err(|e| InferenceError::Ipc(format!("flatten result[{i}]: {e}")))?;
                let depth_data: Vec<f32> = flat
                    .call_method0("tolist")
                    .map_err(|e| InferenceError::Ipc(format!("tolist[{i}]: {e}")))?
                    .extract()
                    .map_err(|e| InferenceError::Ipc(format!("extract depth[{i}]: {e}")))?;

                out.push((item.id.clone(), depth_data, out_w, out_h));
            }

            Ok(out)
        })
    }

    /// Run fused RAUNE + Depth inference on a batch of images.
    ///
    /// For each item: RAUNE enhancement → (optional Maxine upscale) → depth estimation.
    /// Returns `Vec<(id, enhanced_rgb_u8, enh_w, enh_h, depth_f32, depth_w, depth_h)>`.
    ///
    /// `enable_maxine`: if true, Maxine super-resolution runs between RAUNE and Depth.
    #[allow(clippy::type_complexity)]
    pub fn run_raune_depth_batch(
        &mut self,
        items: &[RauneDepthBatchItem],
        enable_maxine: bool,
    ) -> Result<Vec<(String, Vec<u8>, usize, usize, Vec<f32>, usize, usize)>, InferenceError> {
        if items.is_empty() {
            return Ok(vec![]);
        }

        Python::with_gil(|py| {
            let bridge = self.bridge.bind(py);

            // Build Python list of (H, W, 3) numpy arrays
            let py_list = pyo3::types::PyList::empty_bound(py);
            for item in items {
                let np_flat = numpy::PyArray1::from_slice_bound(py, &item.pixels);
                let np_reshaped = np_flat
                    .call_method1("reshape", ((item.height, item.width, 3),))
                    .map_err(|e| InferenceError::Ipc(format!("reshape for id={}: {e}", item.id)))?;
                py_list
                    .append(np_reshaped)
                    .map_err(|e| InferenceError::Ipc(format!("list append for id={}: {e}", item.id)))?;
            }

            let raune_max = items[0].raune_max_size;
            let depth_max = items[0].depth_max_size;

            // Call bridge.run_raune_depth_batch_cpu(imgs, raune_max, depth_max, enable_maxine)
            let result = bridge
                .call_method1(
                    "run_raune_depth_batch_cpu",
                    (py_list, raune_max, depth_max, enable_maxine),
                )
                .map_err(|e| Self::map_python_error(py, e))?;

            // Result is a Python list of (enhanced_np, depth_np) tuples
            let result_list = result
                .downcast::<pyo3::types::PyList>()
                .map_err(|e| InferenceError::Ipc(format!(
                    "run_raune_depth_batch_cpu result not a list: {e}"
                )))?;

            let mut out = Vec::with_capacity(items.len());
            for (i, item) in items.iter().enumerate() {
                let pair = result_list
                    .get_item(i)
                    .map_err(|e| InferenceError::Ipc(format!("get result[{i}]: {e}")))?;

                let pair_tuple = pair
                    .downcast::<pyo3::types::PyTuple>()
                    .map_err(|e| InferenceError::Ipc(format!("result[{i}] not a tuple: {e}")))?;

                // Enhanced RGB u8 array (H, W, 3)
                let enhanced_arr = pair_tuple.get_item(0)
                    .map_err(|e| InferenceError::Ipc(format!("get enhanced[{i}]: {e}")))?;
                let enh_shape: Vec<usize> = enhanced_arr.getattr("shape")
                    .map_err(|e| InferenceError::Ipc(format!("enhanced shape[{i}]: {e}")))?
                    .extract()
                    .map_err(|e| InferenceError::Ipc(format!("extract enhanced shape[{i}]: {e}")))?;
                let enh_h = enh_shape[0];
                let enh_w = enh_shape[1];
                let enhanced_flat = enhanced_arr
                    .call_method1("reshape", ((-1_i32,),))
                    .map_err(|e| InferenceError::Ipc(format!("flatten enhanced[{i}]: {e}")))?;
                let enhanced_data: Vec<u8> = enhanced_flat
                    .call_method0("tolist")
                    .map_err(|e| InferenceError::Ipc(format!("tolist enhanced[{i}]: {e}")))?
                    .extract()
                    .map_err(|e| InferenceError::Ipc(format!("extract enhanced[{i}]: {e}")))?;

                // Depth f32 array (H, W)
                let depth_arr = pair_tuple.get_item(1)
                    .map_err(|e| InferenceError::Ipc(format!("get depth[{i}]: {e}")))?;
                let depth_shape: Vec<usize> = depth_arr.getattr("shape")
                    .map_err(|e| InferenceError::Ipc(format!("depth shape[{i}]: {e}")))?
                    .extract()
                    .map_err(|e| InferenceError::Ipc(format!("extract depth shape[{i}]: {e}")))?;
                let depth_h = depth_shape[0];
                let depth_w = depth_shape[1];
                let depth_flat = depth_arr
                    .call_method1("reshape", ((-1_i32,),))
                    .map_err(|e| InferenceError::Ipc(format!("flatten depth[{i}]: {e}")))?;
                let depth_data: Vec<f32> = depth_flat
                    .call_method0("tolist")
                    .map_err(|e| InferenceError::Ipc(format!("tolist depth[{i}]: {e}")))?
                    .extract()
                    .map_err(|e| InferenceError::Ipc(format!("extract depth[{i}]: {e}")))?;

                out.push((
                    item.id.clone(),
                    enhanced_data, enh_w, enh_h,
                    depth_data, depth_w, depth_h,
                ));
            }

            Ok(out)
        })
    }

    /// Run YOLO-seg binary segmentation on a batch of frames via PyO3.
    pub fn run_yolo_seg_batch(
        &mut self,
        items: &[YoloSegBatchItem],
    ) -> Result<Vec<(String, Vec<u8>, usize, usize)>, InferenceError> {
        if items.is_empty() {
            return Ok(vec![]);
        }

        Python::with_gil(|py| {
            let bridge = self.bridge.bind(py);

            let py_list = pyo3::types::PyList::empty_bound(py);
            for item in items {
                let np_flat = numpy::PyArray1::from_slice_bound(py, &item.pixels);
                let np_reshaped = np_flat
                    .call_method1("reshape", ((item.height, item.width, 3),))
                    .map_err(|e| InferenceError::Ipc(format!("reshape for id={}: {e}", item.id)))?;
                py_list
                    .append(np_reshaped)
                    .map_err(|e| InferenceError::Ipc(format!("list append for id={}: {e}", item.id)))?;
            }

            let result = bridge
                .call_method1("run_yolo_seg_batch_cpu", (py_list,))
                .map_err(|e| Self::map_python_error(py, e))?;

            let result_list = result
                .downcast::<pyo3::types::PyList>()
                .map_err(|e| InferenceError::Ipc(format!("result not a list: {e}")))?;

            let mut out = Vec::with_capacity(items.len());
            for (i, item) in items.iter().enumerate() {
                let arr = result_list.get_item(i)
                    .map_err(|e| InferenceError::Ipc(format!("get result[{i}]: {e}")))?;
                let shape: Vec<usize> = arr.getattr("shape")
                    .map_err(|e| InferenceError::Ipc(format!("shape[{i}]: {e}")))?
                    .extract()
                    .map_err(|e| InferenceError::Ipc(format!("extract shape[{i}]: {e}")))?;
                let h = shape[0];
                let w = shape[1];
                let flat = arr.call_method0("tobytes")
                    .map_err(|e| InferenceError::Ipc(format!("tobytes[{i}]: {e}")))?;
                let mask_data: Vec<u8> = flat.extract()
                    .map_err(|e| InferenceError::Ipc(format!("extract mask[{i}]: {e}")))?;
                out.push((item.id.clone(), mask_data, w, h));
            }

            Ok(out)
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
            let bridge = self.bridge.bind(py);
            let result = bridge.call_method0("vram_free_bytes")
                .map_err(|e| InferenceError::Ipc(format!("vram_free_bytes: {e}")))?;
            let bytes: usize = result.extract()
                .map_err(|e| InferenceError::Ipc(format!("extract vram_free_bytes: {e}")))?;
            Ok(bytes)
        })
    }

    /// Run Maxine enhancement in-process via the PyO3 bridge.
    ///
    /// Returns enhanced RGB u8 at the same resolution as input (`width × height × 3` bytes).
    /// Maxine always preserves input dimensions — no shape extraction needed.
    /// Returns `ServerError` if the Python bridge returns a different size.
    pub fn enhance(
        &self,
        _id: &str,
        image_rgb: &[u8],
        width: usize,
        height: usize,
        artifact_reduce: bool,
    ) -> Result<Vec<u8>, InferenceError> {
        Python::with_gil(|py| {
            let bridge = self.bridge.bind(py);
            let np_flat = numpy::PyArray1::from_slice_bound(py, image_rgb);
            let np_reshaped = np_flat
                .call_method1("reshape", ((height, width, 3),))
                .map_err(|e| InferenceError::Ipc(format!("reshape: {e}")))?;
            let result = bridge
                .call_method1("run_maxine", (np_reshaped, artifact_reduce))
                .map_err(|e| Self::map_python_error(py, e))?;
            let flat = result
                .call_method1("reshape", ((-1_i32,),))
                .map_err(|e| InferenceError::Ipc(format!("flatten result: {e}")))?;
            let rgb_data: Vec<u8> = flat
                .call_method0("tolist")
                .map_err(|e| InferenceError::Ipc(format!("tolist: {e}")))?
                .extract()
                .map_err(|e| InferenceError::Ipc(format!("extract rgb: {e}")))?;
            let expected = width * height * 3;
            if rgb_data.len() != expected {
                return Err(InferenceError::ServerError(format!(
                    "enhance: expected {} bytes ({}x{}x3), got {}",
                    expected, width, height, rgb_data.len()
                )));
            }
            Ok(rgb_data)
        })
    }

    /// Unload Maxine model and free its VRAM without stopping the server.
    pub fn unload_maxine(&self) -> Result<(), InferenceError> {
        Python::with_gil(|py| {
            let bridge = self.bridge.bind(py);
            bridge
                .call_method0("unload_maxine")
                .map_err(|e| InferenceError::InitFailed(format!("unload_maxine: {e}")))?;
            Ok(())
        })
    }

    /// Load RAUNE-Net into the running server (after it was started without it).
    pub fn load_raune(
        &self,
        weights: Option<&std::path::Path>,
        models_dir: Option<&std::path::Path>,
    ) -> Result<(), InferenceError> {
        Python::with_gil(|py| {
            let bridge = self.bridge.bind(py);
            let py_weights = match weights.and_then(|p| p.to_str()) {
                Some(p) => p.into_py(py),
                None => py.None(),
            };
            let py_models_dir = match models_dir.and_then(|p| p.to_str()) {
                Some(p) => p.into_py(py),
                None => py.None(),
            };
            bridge
                .call_method1("load_raune_model", (py_weights, self.device.as_str(), py_models_dir))
                .map_err(|e| InferenceError::InitFailed(format!("load_raune_model: {e}")))?;
            Ok(())
        })
    }

    /// Load Depth Anything into the running server (after it was started without it).
    pub fn load_depth(
        &self,
        model_path: Option<&std::path::Path>,
    ) -> Result<(), InferenceError> {
        Python::with_gil(|py| {
            let bridge = self.bridge.bind(py);
            let py_model_path = match model_path.and_then(|p| p.to_str()) {
                Some(p) => p.into_py(py),
                None => py.None(),
            };
            bridge
                .call_method1("load_depth_model", (py_model_path, self.device.as_str()))
                .map_err(|e| InferenceError::InitFailed(format!("load_depth_model: {e}")))?;
            Ok(())
        })
    }

    /// Graceful shutdown — unload Python models and release VRAM.
    pub fn shutdown(self) -> Result<(), InferenceError> {
        Python::with_gil(|py| {
            let bridge = self.bridge.bind(py);
            bridge.call_method0("unload_models")
                .map_err(|e| InferenceError::InitFailed(format!("unload_models: {e}")))?;
            Ok(())
        })
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
            Some(p) => p.into_py(py),
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
            Some(p) => p.into_py(py),
            None => py.None(),
        };
        let py_models_dir = match models_dir {
            Some(p) => p.into_py(py),
            None => py.None(),
        };
        bridge.call_method1("load_raune_model", (py_weights, device, py_models_dir))
            .map_err(|e| InferenceError::InitFailed(format!("load_raune_model: {e}")))?;
        Ok(())
    }

    /// Load the Maxine model via the Python bridge.
    fn call_load_maxine(
        py: Python<'_>,
        bridge: &Bound<'_, PyModule>,
        upscale_factor: u32,
        _device: &str,  // Reserved for future multi-GPU support; Python bridge is CUDA-only today
    ) -> Result<(), InferenceError> {
        bridge
            .call_method1("load_maxine_model", (upscale_factor,))
            .map_err(|e| InferenceError::InitFailed(format!("load_maxine_model: {e}")))?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_config_default_has_maxine_fields() {
        let cfg = InferenceConfig::default();
        assert!(!cfg.maxine, "maxine should default to false");
        assert_eq!(cfg.maxine_upscale_factor, 2);
        assert!(!cfg.skip_depth, "skip_depth should default to false");
    }

    #[test]
    fn upscale_depth_identity() {
        let src = vec![0.1f32, 0.2, 0.3, 0.4];
        let out = InferenceServer::upscale_depth(&src, 2, 2, 2, 2);
        assert_eq!(out, src);
    }

    #[test]
    fn upscale_depth_bilinear() {
        let src = vec![0.0f32, 1.0, 0.0, 0.0];
        let out = InferenceServer::upscale_depth(&src, 2, 2, 4, 4);
        assert_eq!(out.len(), 16);
        assert!(out[0] >= 0.0 && out[0] <= 1.0,
                "output should be in [0,1], got {}", out[0]);
        // Top-left of output should be close to top-left of input (0.0)
        assert!(out[0] < 0.01, "expected ~0.0, got {}", out[0]);
        // Top-right of output should be close to top-right of input (1.0)
        assert!(out[3] > 0.9, "expected ~1.0, got {}", out[3]);
    }
}
