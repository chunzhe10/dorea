//! Inference integration — PyO3 in-process embedding or subprocess fallback.
//!
//! When the `python` feature is enabled, inference runs in-process via PyO3.
//! Otherwise, the subprocess JSON-lines IPC implementation is used.

#[cfg(feature = "python")]
mod pyo3_backend;

#[cfg(feature = "python")]
pub use pyo3_backend::*;

#[cfg(not(feature = "python"))]
pub use crate::inference_subprocess::*;
