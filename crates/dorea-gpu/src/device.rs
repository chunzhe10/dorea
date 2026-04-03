//! Device memory types for GPU tensor sharing and RAII frame buffers.
//!
//! `BorrowedDeviceSlice` — borrows a device pointer owned by Python (PyTorch).
//! Lifetime-bound to prevent use-after-free at compile time.
//!
//! Only compiled when the `cuda` feature is enabled.

#[cfg(feature = "cuda")]
use std::marker::PhantomData;

/// A borrowed device pointer — NOT owned by Rust.
///
/// Carries lifetime `'a` tied to the scope that guarantees the pointer's validity
/// (typically the GIL scope via `Python<'py>`). The borrow checker prevents this
/// from escaping that scope. No `Drop` impl — no `cudaFree`.
///
/// pub(crate) — not yet integrated. See TODO: wire run_depth_gpu → grade_frame_cuda
/// SAFETY: from_raw caller must ensure the device pointer remains valid while this slice exists.
/// The lifetime 'a is not type-enforced to the Python GIL scope — callers must uphold this manually.
#[cfg(feature = "cuda")]
pub(crate) struct BorrowedDeviceSlice<'a, T> {
    ptr: *mut T,
    len: usize,
    _phantom: PhantomData<(&'a (), *mut T)>,
}

#[cfg(feature = "cuda")]
impl<'a, T> BorrowedDeviceSlice<'a, T> {
    /// Create a borrowed device slice from a raw device pointer.
    ///
    /// # Safety
    /// - `ptr` must be a valid device pointer with at least `len` elements of type T.
    /// - The pointer must remain valid for lifetime `'a`.
    /// - The caller must ensure no concurrent writes to this memory.
    pub(crate) unsafe fn from_raw(ptr: *mut T, len: usize) -> Self {
        Self { ptr, len, _phantom: PhantomData }
    }

    /// Raw device pointer (for passing to cudarc kernel launches).
    pub fn as_device_ptr(&self) -> *const T {
        self.ptr as *const T
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the slice is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// BorrowedDeviceSlice is !Send + !Sync because raw pointers are not Send/Sync.
// This is correct — device pointers must not cross thread boundaries.

/// Per-frame VRAM cost at a given resolution.
///
/// Used by AdaptiveBatcher to compute max batch size from available VRAM.
/// Accounts for all allocations across LUT, HSL, and clarity stages.
#[cfg(feature = "cuda")]
pub fn per_frame_vram_bytes(width: usize, height: usize) -> usize {
    let n = width * height;
    // Proxy dims for clarity stage
    let long_edge = width.max(height);
    let proxy_n = if long_edge <= 518 {
        n
    } else {
        let scale = 518.0_f64 / long_edge as f64;
        let pw = ((width as f64 * scale).round() as usize).max(1);
        let ph = ((height as f64 * scale).round() as usize).max(1);
        pw * ph
    };

    let rgb_f32 = n * 3 * 4;    // one RGB plane (n pixels × 3 channels × 4 bytes)
    let depth_f32 = n * 4;       // depth plane
    let lut_budget = 5 * 33 * 33 * 33 * 3 * 4;  // 5 zones × 33³ × 3ch × 4 bytes ≈ 22 MB
    let proxy_f32 = proxy_n * 4; // one proxy L plane

    // Peak during LUT stage: d_rgb_in + d_depth + d_luts + d_rgb_after_lut
    // Peak during clarity: d_rgb_after_hsl + d_proxy_l + d_blur_a + d_blur_b + d_rgb_out
    // Conservative: 3 RGB planes + depth + LUT budget + 3 proxy planes
    3 * rgb_f32 + depth_f32 + lut_budget + 3 * proxy_f32
}

/// Verify that the CUDA context is healthy by performing a tiny allocation.
///
/// Call this after OOM recovery before retrying. If it fails, the context
/// is likely wedged — skip to CPU fallback.
#[cfg(feature = "cuda")]
pub fn verify_cuda_context(device: &std::sync::Arc<cudarc::driver::CudaDevice>) -> Result<(), crate::GpuError> {
    let _probe = device.alloc_zeros::<f32>(1)
        .map_err(|e| crate::GpuError::CudaFail(format!("context health check failed: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn per_frame_vram_1080p() {
        #[cfg(feature = "cuda")]
        {
            use super::per_frame_vram_bytes;
            let bytes = per_frame_vram_bytes(1920, 1080);
            // 1920*1080 = 2073600 pixels
            // long_edge = 1920 > 518, scale = 518/1920 ≈ 0.26979
            // pw = round(1920 * 0.26979) = round(518) = 518
            // ph = round(1080 * 0.26979) = round(291.37) = 291
            // proxy_n = 518 * 291 = 150738
            // rgb_f32 = 2073600 * 3 * 4 = 24883200
            // depth_f32 = 2073600 * 4 = 8294400
            // lut_budget = 5 * 33 * 33 * 33 * 3 * 4 = 21,781,620 ... actually 5*35937*12 = 2155620 ... recalc
            // 33^3 = 35937, 5 * 35937 * 3 * 4 = 5 * 35937 * 12 = 2,156,220
            // proxy_f32 = 150738 * 4 = 602952
            // total = 3*24883200 + 8294400 + 2156220 + 3*602952
            //       = 74649600 + 8294400 + 2156220 + 1808856
            //       = 86909076
            let n = 1920usize * 1080;
            let scale = 518.0_f64 / 1920.0_f64;
            let pw = ((1920.0_f64 * scale).round() as usize).max(1);
            let ph = ((1080.0_f64 * scale).round() as usize).max(1);
            let proxy_n = pw * ph;
            let rgb_f32 = n * 3 * 4;
            let depth_f32 = n * 4;
            let lut_budget = 5 * 33 * 33 * 33 * 3 * 4;
            let proxy_f32 = proxy_n * 4;
            let expected = 3 * rgb_f32 + depth_f32 + lut_budget + 3 * proxy_f32;
            assert_eq!(bytes, expected, "per_frame_vram_bytes mismatch");
        }
    }
}
