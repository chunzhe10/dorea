//! CombinedLut — manages CUDA 3D texture arrays for the precomputed combined LUT.
//!
//! One `CUarray` + `CUtexObject` per depth zone.
//! Created once at `CudaGrader::new()`; reused for all frames.

use std::sync::Arc;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DeviceRepr, LaunchAsync, LaunchConfig};
use cudarc::driver::sys::{self, *};
use cudarc::nvrtc::Ptx;
use crate::{GradeParams, GpuError};
use dorea_cal::Calibration;

use super::map_cudarc_error;

/// Grid size for the combined LUT (N³ lattice points per zone).
/// Bump to 129 if 10-bit empirical testing reveals banding.
pub const COMBINED_LUT_GRID: usize = 97;

/// Embedded PTX for the GPU LUT build kernel.
const BUILD_COMBINED_LUT_PTX: &str =
    include_str!(concat!(env!("OUT_DIR"), "/build_combined_lut.ptx"));

pub(crate) struct CombinedLut {
    /// One CUarray per zone (float4, N×N×N).
    arrays:          Vec<CUarray>,
    /// One texture object per zone.
    pub textures:    Vec<CUtexObject>,
    pub n_zones:     usize,
    pub grid_size:   usize,
    /// Zone boundary values [n_zones + 1], kept for the per-frame kernel.
    pub zone_boundaries: Vec<f32>,
}

// Safety: CombinedLut is !Send by design (CUDA context is thread-local).
// We do not implement Send/Sync; callers hold it inside CudaGrader which is also !Send.
impl Drop for CombinedLut {
    fn drop(&mut self) {
        unsafe {
            for &tex in &self.textures {
                let _ = sys::lib().cuTexObjectDestroy(tex);
            }
            for &arr in &self.arrays {
                let _ = sys::lib().cuArrayDestroy(arr);
            }
        }
    }
}

impl CombinedLut {
    /// Build the combined LUT textures from calibration data.
    ///
    /// Call once at grader construction. This:
    ///   1. Uploads calibration to device
    ///   2. Launches build_combined_lut_kernel (fills float4 device buffer)
    ///   3. Copies each zone's slice into a CUarray (device-to-device via cuMemcpy3D)
    ///   4. Creates an unnormalized-coordinate linear texture object per zone
    ///   5. Frees the build buffer (73MB freed after this fn returns)
    pub(crate) fn build(
        device: &Arc<CudaDevice>,
        calibration: &Calibration,
        params: &GradeParams,
    ) -> Result<Self, GpuError> {
        let n_zones = calibration.depth_luts.n_zones();
        if n_zones == 0 {
            return Err(GpuError::InvalidInput("n_zones must be >= 1".into()));
        }
        let n = COMBINED_LUT_GRID;
        let lut_size = calibration.depth_luts.luts[0].size;

        // --- Upload calibration data ---
        let luts_flat: Vec<f32> = calibration.depth_luts.luts.iter()
            .flat_map(|lg| lg.data.iter().copied())
            .collect();
        let d_luts = device.htod_sync_copy(&luts_flat).map_err(map_cudarc_error)?;
        let d_boundaries = device.htod_sync_copy(&calibration.depth_luts.zone_boundaries)
            .map_err(map_cudarc_error)?;

        let mut h_offsets = [0.0f32; 6];
        let mut s_ratios  = [1.0f32; 6];
        let mut v_offsets = [0.0f32; 6];
        let mut weights   = [0.0f32; 6];
        for (i, q) in calibration.hsl_corrections.0.iter().enumerate().take(6) {
            h_offsets[i] = q.h_offset;
            s_ratios[i]  = q.s_ratio;
            v_offsets[i] = q.v_offset;
            weights[i]   = q.weight;
        }
        let d_h_offsets = device.htod_sync_copy(&h_offsets).map_err(map_cudarc_error)?;
        let d_s_ratios  = device.htod_sync_copy(&s_ratios).map_err(map_cudarc_error)?;
        let d_v_offsets = device.htod_sync_copy(&v_offsets).map_err(map_cudarc_error)?;
        let d_weights   = device.htod_sync_copy(&weights).map_err(map_cudarc_error)?;

        // --- Allocate build buffer: n_zones × N³ × float4 (4 floats) ---
        let build_count = n_zones * n * n * n * 4;
        let d_build: CudaSlice<f32> = device.alloc_zeros(build_count).map_err(map_cudarc_error)?;

        // --- Launch build kernel ---
        device.load_ptx(
            Ptx::from_src(BUILD_COMBINED_LUT_PTX),
            "build_combined_lut",
            &["build_combined_lut_kernel"],
        ).map_err(|e| GpuError::ModuleLoad(format!("load build_combined_lut PTX: {e}")))?;

        let total_threads = (n_zones * n * n * n) as u32;
        let block = 256u32;
        let grid  = total_threads.div_ceil(block);

        {
            let func = device.get_func("build_combined_lut", "build_combined_lut_kernel")
                .ok_or_else(|| GpuError::ModuleLoad("build_combined_lut_kernel not found".into()))?;
            let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 };
            // cudarc 0.12 LaunchAsync supports tuples up to 12 elements.
            // We have 14 args, so we use the raw *mut c_void params slice instead.
            let warmth   = params.warmth;
            let strength = params.strength;
            let contrast = params.contrast;
            let n_i32        = n as i32;
            let lut_size_i32 = lut_size as i32;
            let n_zones_i32  = n_zones as i32;
            let total_i32    = total_threads as i32;
            let mut args: [*mut std::ffi::c_void; 14] = [
                (&d_build)   .as_kernel_param(),
                (&d_luts)    .as_kernel_param(),
                (&d_boundaries).as_kernel_param(),
                (&d_h_offsets) .as_kernel_param(),
                (&d_s_ratios)  .as_kernel_param(),
                (&d_v_offsets) .as_kernel_param(),
                (&d_weights)   .as_kernel_param(),
                warmth   .as_kernel_param(),
                strength .as_kernel_param(),
                contrast .as_kernel_param(),
                n_i32       .as_kernel_param(),
                lut_size_i32.as_kernel_param(),
                n_zones_i32 .as_kernel_param(),
                total_i32   .as_kernel_param(),
            ];
            unsafe {
                func.launch(cfg, &mut args[..])
            }.map_err(map_cudarc_error)?;
        }

        // Synchronise: ensure build kernel completed before cuMemcpy3D
        device.synchronize().map_err(map_cudarc_error)?;

        // --- For each zone: copy device buffer slice → CUarray, create texture ---
        let mut arrays:   Vec<CUarray>     = Vec::with_capacity(n_zones);
        let mut textures: Vec<CUtexObject> = Vec::with_capacity(n_zones);

        // Raw device pointer to the build buffer
        let build_base: CUdeviceptr = *d_build.device_ptr();
        // Bytes per zone: N*N*N float4 = N*N*N * 4 * sizeof(f32)
        let zone_bytes = (n * n * n * 4 * std::mem::size_of::<f32>()) as u64;
        // Width in bytes for one row: N texels × float4 = N × 16 bytes
        let width_in_bytes = n * 4 * std::mem::size_of::<f32>();

        for zone in 0..n_zones {
            // --- Create CUarray ---
            let arr_desc = CUDA_ARRAY3D_DESCRIPTOR {
                Width:       n,
                Height:      n,
                Depth:       n,
                Format:      CUarray_format_enum::CU_AD_FORMAT_FLOAT,
                NumChannels: 4,
                Flags:       0,
            };
            let mut arr: CUarray = std::ptr::null_mut();
            unsafe {
                let rc = sys::lib().cuArray3DCreate_v2(&mut arr, &arr_desc);
                if rc != cudaError_enum::CUDA_SUCCESS {
                    return Err(GpuError::CudaFail(format!("cuArray3DCreate zone {zone}: {rc:?}")));
                }
            }
            arrays.push(arr);

            // --- cuMemcpy3D: linear device buffer → CUarray (device-to-device) ---
            let zone_ptr = build_base + zone as u64 * zone_bytes;
            // Use MaybeUninit::zeroed() to avoid Rust's debug-mode zero-init validity
            // checks on enums (CUmemorytype has no zero variant).
            let mut cpy: CUDA_MEMCPY3D =
                unsafe { std::mem::MaybeUninit::<CUDA_MEMCPY3D>::zeroed().assume_init() };
            cpy.srcMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_DEVICE;
            cpy.srcDevice     = zone_ptr;
            cpy.srcPitch      = width_in_bytes;
            cpy.srcHeight     = n;
            cpy.dstMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_ARRAY;
            cpy.dstArray      = arr;
            cpy.WidthInBytes  = width_in_bytes;
            cpy.Height        = n;
            cpy.Depth         = n;

            unsafe {
                let rc = sys::lib().cuMemcpy3D_v2(&cpy);
                if rc != cudaError_enum::CUDA_SUCCESS {
                    // arr was pushed; destroy it along with all previously created resources.
                    for &tex in &textures { let _ = sys::lib().cuTexObjectDestroy(tex); }
                    for &a in &arrays    { let _ = sys::lib().cuArrayDestroy(a); }
                    return Err(GpuError::CudaFail(format!("cuMemcpy3D zone {zone}: {rc:?}")));
                }
            }

            // --- Create texture object (unnormalized coords, linear filter) ---
            let mut res_desc: CUDA_RESOURCE_DESC = unsafe { std::mem::zeroed() };
            res_desc.resType = CUresourcetype_enum::CU_RESOURCE_TYPE_ARRAY;
            res_desc.res.array.hArray = arr;

            let mut tex_desc: CUDA_TEXTURE_DESC = unsafe { std::mem::zeroed() };
            tex_desc.addressMode[0] = CUaddress_mode_enum::CU_TR_ADDRESS_MODE_CLAMP;
            tex_desc.addressMode[1] = CUaddress_mode_enum::CU_TR_ADDRESS_MODE_CLAMP;
            tex_desc.addressMode[2] = CUaddress_mode_enum::CU_TR_ADDRESS_MODE_CLAMP;
            tex_desc.filterMode     = CUfilter_mode_enum::CU_TR_FILTER_MODE_LINEAR;
            tex_desc.flags          = 0;    // unnormalized coords

            let mut tex: CUtexObject = 0;
            unsafe {
                let rc = sys::lib().cuTexObjectCreate(&mut tex, &res_desc, &tex_desc, std::ptr::null());
                if rc != cudaError_enum::CUDA_SUCCESS {
                    // arr was pushed; destroy it and all previously created resources.
                    for &tex in &textures { let _ = sys::lib().cuTexObjectDestroy(tex); }
                    for &a in &arrays    { let _ = sys::lib().cuArrayDestroy(a); }
                    return Err(GpuError::CudaFail(format!("cuTexObjectCreate zone {zone}: {rc:?}")));
                }
            }
            textures.push(tex);
        }

        // d_build drops here — build buffer freed
        Ok(Self {
            arrays,
            textures,
            n_zones,
            grid_size: n,
            zone_boundaries: calibration.depth_luts.zone_boundaries.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// TextureSet — one set of per-zone 3D textures (CUarray + CUtexObject per zone)
// ---------------------------------------------------------------------------

/// One set of per-zone 3D textures (CUarray + CUtexObject per zone).
struct TextureSet {
    arrays: Vec<CUarray>,
    textures: Vec<CUtexObject>,
    n_zones: usize,
    zone_boundaries: Vec<f32>,
}

impl TextureSet {
    /// Allocate CUarrays and texture objects for `n_zones` zones.
    fn allocate(n_zones: usize, grid_size: usize) -> Result<Self, GpuError> {
        let n = grid_size;
        let mut arrays: Vec<CUarray>     = Vec::with_capacity(n_zones);
        let mut textures: Vec<CUtexObject> = Vec::with_capacity(n_zones);

        for _ in 0..n_zones {
            let mut arr: CUarray = std::ptr::null_mut();
            let desc = CUDA_ARRAY3D_DESCRIPTOR {
                Width:       n,
                Height:      n,
                Depth:       n,
                Format:      CUarray_format_enum::CU_AD_FORMAT_FLOAT,
                NumChannels: 4,
                Flags:       0,
            };
            unsafe {
                let rc = sys::lib().cuArray3DCreate_v2(&mut arr, &desc);
                if rc != cudaError_enum::CUDA_SUCCESS {
                    // clean up already-created resources
                    for &tex in &textures { let _ = sys::lib().cuTexObjectDestroy(tex); }
                    for &a in &arrays    { let _ = sys::lib().cuArrayDestroy(a); }
                    return Err(GpuError::CudaFail(format!("cuArray3DCreate: {rc:?}")));
                }
            }
            arrays.push(arr);

            let tex = Self::create_texture(arr)?;
            textures.push(tex);
        }

        Ok(Self {
            arrays,
            textures,
            n_zones,
            zone_boundaries: vec![0.0; n_zones + 1],
        })
    }

    fn create_texture(arr: CUarray) -> Result<CUtexObject, GpuError> {
        let mut res_desc: CUDA_RESOURCE_DESC = unsafe { std::mem::zeroed() };
        res_desc.resType = CUresourcetype_enum::CU_RESOURCE_TYPE_ARRAY;
        res_desc.res.array.hArray = arr;

        let mut tex_desc: CUDA_TEXTURE_DESC = unsafe { std::mem::zeroed() };
        tex_desc.addressMode[0] = CUaddress_mode_enum::CU_TR_ADDRESS_MODE_CLAMP;
        tex_desc.addressMode[1] = CUaddress_mode_enum::CU_TR_ADDRESS_MODE_CLAMP;
        tex_desc.addressMode[2] = CUaddress_mode_enum::CU_TR_ADDRESS_MODE_CLAMP;
        tex_desc.filterMode     = CUfilter_mode_enum::CU_TR_FILTER_MODE_LINEAR;
        tex_desc.flags          = 0; // unnormalized coords

        let mut tex: CUtexObject = 0;
        unsafe {
            let rc = sys::lib().cuTexObjectCreate(&mut tex, &res_desc, &tex_desc, std::ptr::null());
            if rc != cudaError_enum::CUDA_SUCCESS {
                return Err(GpuError::CudaFail(format!("cuTexObjectCreate: {rc:?}")));
            }
        }
        Ok(tex)
    }
}

impl Drop for TextureSet {
    fn drop(&mut self) {
        unsafe {
            for &tex in &self.textures { let _ = sys::lib().cuTexObjectDestroy(tex); }
            for &arr in &self.arrays   { let _ = sys::lib().cuArrayDestroy(arr); }
        }
    }
}

// ---------------------------------------------------------------------------
// AdaptiveLut — double-buffered adaptive LUT with per-keyframe texture rebuilds
// ---------------------------------------------------------------------------

/// Double-buffered adaptive LUT with per-keyframe texture rebuilds.
///
/// Holds two `TextureSet`s. One is "active" (used for grading), the other can
/// be rebuilt for the next keyframe. Swapping is a pointer toggle — no copies.
pub(crate) struct AdaptiveLut {
    sets: [TextureSet; 2],
    /// Index of the set currently used for grading (0 or 1).
    pub active: usize,
    pub grid_size: usize,
    pub runtime_n_zones: usize,

    // Base LUT data on device (per-segment, rarely changes)
    d_base_luts: CudaSlice<f32>,
    d_base_boundaries: CudaSlice<f32>,
    base_n_zones: usize,

    // HSL + grade params on device (constant per clip)
    d_h_offsets: CudaSlice<f32>,
    d_s_ratios: CudaSlice<f32>,
    d_v_offsets: CudaSlice<f32>,
    d_weights: CudaSlice<f32>,
    warmth: f32,
    strength: f32,
    contrast: f32,
    lut_size: usize,
}

impl AdaptiveLut {
    /// Create with two pre-allocated texture sets.
    ///
    /// `base_luts_flat`: flattened [base_n_zones][lut_size^3][3]
    /// `base_boundaries`: [base_n_zones + 1]
    /// `hsl_data`: (h_offsets, s_ratios, v_offsets, weights) each &[f32] of len 6
    /// `params`: (warmth, strength, contrast)
    pub(crate) fn new(
        device: &Arc<CudaDevice>,
        base_luts_flat: &[f32],
        base_boundaries: &[f32],
        base_n_zones: usize,
        hsl_data: (&[f32], &[f32], &[f32], &[f32]),
        params: (f32, f32, f32),
        lut_size: usize,
        runtime_n_zones: usize,
    ) -> Result<Self, GpuError> {
        let grid_size = COMBINED_LUT_GRID;

        let d_base_luts = device.htod_sync_copy(base_luts_flat).map_err(map_cudarc_error)?;
        let d_base_boundaries = device.htod_sync_copy(base_boundaries).map_err(map_cudarc_error)?;

        let (ho, sr, vo, wt) = hsl_data;
        let d_h_offsets = device.htod_sync_copy(ho).map_err(map_cudarc_error)?;
        let d_s_ratios  = device.htod_sync_copy(sr).map_err(map_cudarc_error)?;
        let d_v_offsets = device.htod_sync_copy(vo).map_err(map_cudarc_error)?;
        let d_weights   = device.htod_sync_copy(wt).map_err(map_cudarc_error)?;

        let set_a = TextureSet::allocate(runtime_n_zones, grid_size)?;
        let set_b = TextureSet::allocate(runtime_n_zones, grid_size)?;

        Ok(Self {
            sets: [set_a, set_b],
            active: 0,
            grid_size,
            runtime_n_zones,
            d_base_luts,
            d_base_boundaries,
            base_n_zones,
            d_h_offsets,
            d_s_ratios,
            d_v_offsets,
            d_weights,
            warmth: params.0,
            strength: params.1,
            contrast: params.2,
            lut_size,
        })
    }

    /// Rebuild one texture set with new runtime zone boundaries.
    ///
    /// Launches `build_combined_lut_kernel` to fill a temp buffer, then
    /// copies each zone's slice into the set's CUarrays via cuMemcpy3D.
    pub(crate) fn rebuild_set(
        &mut self,
        device: &Arc<CudaDevice>,
        set_index: usize,
        runtime_boundaries: &[f32],
    ) -> Result<(), GpuError> {
        let n = self.grid_size;
        let nz = self.runtime_n_zones;
        let total = nz * n * n * n;

        let d_runtime_bounds = device.htod_sync_copy(runtime_boundaries).map_err(map_cudarc_error)?;
        let d_build: CudaSlice<f32> = device.alloc_zeros(total * 4).map_err(map_cudarc_error)?;

        let func = device.get_func("build_combined_lut", "build_combined_lut_kernel")
            .ok_or_else(|| GpuError::ModuleLoad("build_combined_lut_kernel not found".into()))?;
        let cfg = LaunchConfig {
            grid_dim: (((total as u32) + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let warmth   = self.warmth;
        let strength = self.strength;
        let contrast = self.contrast;
        let n_i32        = n as i32;
        let lut_size_i32 = self.lut_size as i32;
        let base_n_zones_i32 = self.base_n_zones as i32;
        let nz_i32       = nz as i32;
        let total_i32    = total as i32;
        let mut args: [*mut std::ffi::c_void; 16] = [
            (&d_build)            .as_kernel_param(),
            (&self.d_base_luts)   .as_kernel_param(),
            (&self.d_base_boundaries).as_kernel_param(),
            base_n_zones_i32      .as_kernel_param(),
            (&d_runtime_bounds)   .as_kernel_param(),
            nz_i32                .as_kernel_param(),
            (&self.d_h_offsets)   .as_kernel_param(),
            (&self.d_s_ratios)    .as_kernel_param(),
            (&self.d_v_offsets)   .as_kernel_param(),
            (&self.d_weights)     .as_kernel_param(),
            warmth                .as_kernel_param(),
            strength              .as_kernel_param(),
            contrast              .as_kernel_param(),
            n_i32                 .as_kernel_param(),
            lut_size_i32          .as_kernel_param(),
            total_i32             .as_kernel_param(),
        ];
        unsafe {
            func.launch(cfg, &mut args[..])
        }.map_err(map_cudarc_error)?;

        device.synchronize().map_err(map_cudarc_error)?;

        // Copy build buffer → CUarrays (one per runtime zone)
        let zone_bytes = (n * n * n * 4 * std::mem::size_of::<f32>()) as u64;
        let width_in_bytes = n * 4 * std::mem::size_of::<f32>();
        let build_base: CUdeviceptr = *d_build.device_ptr();

        for z in 0..nz {
            let zone_ptr = build_base + z as u64 * zone_bytes;
            // Use MaybeUninit to avoid debug-mode validity checks on CUmemorytype enum
            let mut cpy: CUDA_MEMCPY3D =
                unsafe { std::mem::MaybeUninit::<CUDA_MEMCPY3D>::zeroed().assume_init() };
            cpy.srcMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_DEVICE;
            cpy.srcDevice     = zone_ptr;
            cpy.srcPitch      = width_in_bytes;
            cpy.srcHeight     = n;
            cpy.dstMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_ARRAY;
            cpy.dstArray      = self.sets[set_index].arrays[z];
            cpy.WidthInBytes  = width_in_bytes;
            cpy.Height        = n;
            cpy.Depth         = n;
            unsafe {
                let rc = sys::lib().cuMemcpy3D_v2(&cpy);
                if rc != cudaError_enum::CUDA_SUCCESS {
                    return Err(GpuError::CudaFail(format!("cuMemcpy3D zone {z}: {rc:?}")));
                }
            }
        }

        self.sets[set_index].zone_boundaries = runtime_boundaries.to_vec();
        Ok(())
    }

    /// Load new segment base LUT + HSL data (called at scene segment boundaries).
    pub(crate) fn load_segment(
        &mut self,
        device: &Arc<CudaDevice>,
        base_luts_flat: &[f32],
        base_boundaries: &[f32],
        hsl_data: (&[f32], &[f32], &[f32], &[f32]),
    ) -> Result<(), GpuError> {
        self.d_base_luts = device.htod_sync_copy(base_luts_flat).map_err(map_cudarc_error)?;
        self.d_base_boundaries = device.htod_sync_copy(base_boundaries).map_err(map_cudarc_error)?;
        self.base_n_zones = base_boundaries.len() - 1;
        let (ho, sr, vo, wt) = hsl_data;
        self.d_h_offsets = device.htod_sync_copy(ho).map_err(map_cudarc_error)?;
        self.d_s_ratios  = device.htod_sync_copy(sr).map_err(map_cudarc_error)?;
        self.d_v_offsets = device.htod_sync_copy(vo).map_err(map_cudarc_error)?;
        self.d_weights   = device.htod_sync_copy(wt).map_err(map_cudarc_error)?;
        Ok(())
    }

    /// Swap active/inactive sets. Returns the new active index.
    pub(crate) fn swap(&mut self) -> usize {
        self.active = 1 - self.active;
        self.active
    }

    pub(crate) fn active_textures(&self) -> &[CUtexObject] {
        &self.sets[self.active].textures
    }

    pub(crate) fn active_boundaries(&self) -> &[f32] {
        &self.sets[self.active].zone_boundaries
    }

    pub(crate) fn inactive_textures(&self) -> &[CUtexObject] {
        &self.sets[1 - self.active].textures
    }

    pub(crate) fn inactive_boundaries(&self) -> &[f32] {
        &self.sets[1 - self.active].zone_boundaries
    }
}
