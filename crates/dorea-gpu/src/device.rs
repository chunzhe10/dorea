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
#[cfg(feature = "cuda")]
pub struct BorrowedDeviceSlice<'a, T> {
    ptr: *mut T,
    len: usize,
    _phantom: PhantomData<&'a T>,
}

#[cfg(feature = "cuda")]
impl<'a, T> BorrowedDeviceSlice<'a, T> {
    /// Create a borrowed device slice from a raw device pointer.
    ///
    /// # Safety
    /// - `ptr` must be a valid device pointer with at least `len` elements of type T.
    /// - The pointer must remain valid for lifetime `'a`.
    /// - The caller must ensure no concurrent writes to this memory.
    pub unsafe fn from_raw(ptr: *mut T, len: usize) -> Self {
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
#[cfg(feature = "cuda")]
pub fn per_frame_vram_bytes(width: usize, height: usize) -> usize {
    let n_pixels = width * height;
    let rgb_f32 = n_pixels * 3 * std::mem::size_of::<f32>();
    let depth_f32 = n_pixels * std::mem::size_of::<f32>();
    // Ping-pong buffers: 2x RGB + 1x depth
    rgb_f32 * 2 + depth_f32
}

/// Verify that the CUDA context is healthy by querying the free device memory.
///
/// Call this after OOM recovery before retrying. If it fails, the context
/// is likely wedged — skip to CPU fallback.
///
/// Uses the raw CUDA driver API (cudaMemGetInfo) — no cudarc dependency needed.
#[cfg(feature = "cuda")]
pub fn verify_cuda_context() -> Result<(), crate::GpuError> {
    extern "C" {
        /// cudaMemGetInfo(size_t *free, size_t *total) → cudaError_t
        fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    }
    let mut free: usize = 0;
    let mut total: usize = 0;
    let status = unsafe { cudaMemGetInfo(&mut free, &mut total) };
    if status != 0 {
        return Err(crate::GpuError::CudaFail(
            format!("context health check failed: cudaMemGetInfo returned {status}")
        ));
    }
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
            // 2 * 2073600 * 3 * 4 (RGB f32 ping-pong) + 2073600 * 4 (depth f32)
            // = 49766400 + 8294400 = 58060800 ≈ 55.4 MB
            assert_eq!(bytes, 58_060_800);
        }
    }
}
