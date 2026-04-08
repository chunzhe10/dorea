//! CUDA-accelerated guided filter for depth map edge snapping.
//!
//! Snaps warped depth maps to class boundaries from YOLO-seg masks.
//! The guided filter preserves depth gradients within class regions but
//! enforces sharp edges at class transitions (diver vs water).

#[cfg(feature = "cuda")]
use std::sync::Arc;
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::Ptx;
#[cfg(feature = "cuda")]
use crate::GpuError;

#[cfg(feature = "cuda")]
use super::map_cudarc_error;

/// Embedded PTX for the guided filter kernel.
#[cfg(feature = "cuda")]
const GUIDED_FILTER_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/guided_filter.ptx"));

/// Parameters for the guided filter.
#[derive(Debug, Clone)]
pub struct GuidedFilterParams {
    /// Filter radius (window = 2*radius+1). Default: 4 (9x9 window at proxy res).
    pub radius: i32,
    /// Epsilon for water regions (larger = smoother). Default: 0.01.
    pub epsilon_water: f32,
    /// Epsilon for diver regions (smaller = sharper edges). Default: 0.001.
    pub epsilon_diver: f32,
}

impl Default for GuidedFilterParams {
    fn default() -> Self {
        Self {
            radius: 4,
            epsilon_water: 0.01,
            epsilon_diver: 0.001,
        }
    }
}

/// Apply guided filter to snap a warped depth map to class mask boundaries.
///
/// `depth`: f32 depth map at proxy resolution (width * height).
/// `class_mask`: u8 class mask (0=water, 1=diver) at the same resolution.
/// Returns the filtered depth map.
#[cfg(feature = "cuda")]
pub fn guided_filter_cuda(
    device: &Arc<CudaDevice>,
    depth: &[f32],
    class_mask: &[u8],
    width: usize,
    height: usize,
    params: &GuidedFilterParams,
) -> Result<Vec<f32>, GpuError> {
    let n = width * height;
    if depth.len() != n {
        return Err(GpuError::InvalidInput(format!(
            "depth len {} != width*height {}", depth.len(), n
        )));
    }
    if class_mask.len() != n {
        return Err(GpuError::InvalidInput(format!(
            "class_mask len {} != width*height {}", class_mask.len(), n
        )));
    }

    // Load PTX (idempotent — cudarc caches loaded modules)
    device.load_ptx(
        Ptx::from_src(GUIDED_FILTER_PTX),
        "guided_filter",
        &["guided_filter_kernel"],
    ).map_err(|e| GpuError::ModuleLoad(format!("load guided_filter PTX: {e}")))?;

    // Upload inputs
    let d_depth = device.htod_sync_copy(depth).map_err(map_cudarc_error)?;
    let d_guide = device.htod_sync_copy(class_mask).map_err(map_cudarc_error)?;
    let d_out: CudaSlice<f32> = device.alloc_zeros(n).map_err(map_cudarc_error)?;

    // Launch kernel (2D grid)
    let func = device.get_func("guided_filter", "guided_filter_kernel")
        .ok_or_else(|| GpuError::ModuleLoad("guided_filter_kernel not found".into()))?;
    let block_dim = (16u32, 16u32, 1u32);
    let grid_dim = (
        (width as u32 + block_dim.0 - 1) / block_dim.0,
        (height as u32 + block_dim.1 - 1) / block_dim.1,
        1u32,
    );
    let cfg = LaunchConfig {
        grid_dim,
        block_dim,
        shared_mem_bytes: 0,
    };

    unsafe {
        func.launch(cfg, (
            &d_depth,
            &d_guide,
            &d_out,
            width as i32,
            height as i32,
            params.radius,
            params.epsilon_water,
            params.epsilon_diver,
        ))
    }.map_err(map_cudarc_error)?;

    // Download result
    let result = device.dtoh_sync_copy(&d_out).map_err(map_cudarc_error)?;
    Ok(result)
}
