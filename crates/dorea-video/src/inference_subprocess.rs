// Rust-side manager for the Python inference subprocess.
// Spawns `python -m dorea_inference.server` with model paths,
// communicates via JSON lines over stdin/stdout.

use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;
use thiserror::Error;

use base64::{engine::general_purpose::STANDARD as B64, Engine};

#[derive(Debug, Error)]
pub enum InferenceError {
    #[error("failed to spawn inference server: {0}")]
    SpawnFailed(#[from] io::Error),
    #[error("IPC error: {0}")]
    Ipc(String),
    #[error("inference server error: {0}")]
    ServerError(String),
    #[error("timeout waiting for inference server")]
    Timeout,
    #[error("PNG encode/decode error: {0}")]
    ImageError(String),
}

/// Configuration for spawning the inference server.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Python executable path (e.g. `/opt/dorea-venv/bin/python`).
    pub python_exe: PathBuf,
    /// Path to the RAUNE-Net weights .pth file.
    /// `None` = let Python use its built-in default path.
    /// Set `skip_raune = true` to disable RAUNE entirely.
    pub raune_weights: Option<PathBuf>,
    /// Path to the sea_thru_poc directory (contains models/raune_net.py).
    pub raune_models_dir: Option<PathBuf>,
    /// Skip RAUNE-Net entirely (pass `--no-raune` to the server).
    /// Use this for inference phases that only need depth (e.g. per-frame grading).
    pub skip_raune: bool,
    /// Path to Depth Anything V2 model directory. None = use Python default / HF.
    pub depth_model: Option<PathBuf>,
    /// Compute device: "cpu" or "cuda". None = auto-detect.
    pub device: Option<String>,
    /// Startup timeout.
    pub startup_timeout: Duration,
    /// Enable Maxine enhancement in the inference subprocess.
    pub maxine: bool,
    /// Maxine super-resolution upscale factor (default 2).
    pub maxine_upscale_factor: u32,
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
        }
    }
}

impl InferenceConfig {
    /// Build the CLI argument list for the Python inference server.
    pub fn build_args(&self) -> Vec<String> {
        let mut args = vec!["-m".to_string(), "dorea_inference.server".to_string()];

        if self.skip_raune {
            args.push("--no-raune".to_string());
        } else {
            if let Some(p) = &self.raune_weights {
                args.push("--raune-weights".to_string());
                args.push(p.to_str().unwrap_or("").to_string());
            }
        }

        if let Some(p) = &self.raune_models_dir {
            args.push("--raune-models-dir".to_string());
            args.push(p.to_str().unwrap_or("").to_string());
        }

        if let Some(p) = &self.depth_model {
            args.push("--depth-model".to_string());
            args.push(p.to_str().unwrap_or("").to_string());
        }

        if let Some(d) = &self.device {
            args.push("--device".to_string());
            args.push(d.clone());
        }

        if self.maxine {
            args.push("--maxine".to_string());
            args.push("--maxine-upscale-factor".to_string());
            args.push(self.maxine_upscale_factor.to_string());
        }

        args
    }
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

/// A single image for the fused raune_depth_batch IPC call.
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

/// Active connection to the Python inference subprocess.
pub struct InferenceServer {
    child: Child,
    stdin: ChildStdin,
    stdout_reader: BufReader<std::process::ChildStdout>,
}

impl InferenceServer {
    /// Spawn the inference server and wait for it to signal readiness.
    ///
    /// Enforces `config.startup_timeout`: returns `InferenceError::Timeout` if the
    /// Python process does not respond to the initial ping within the deadline.
    pub fn spawn(config: &InferenceConfig) -> Result<Self, InferenceError> {
        let mut cmd = Command::new(&config.python_exe);
        cmd.args(config.build_args());

        // Set PYTHONPATH to the python/ dir adjacent to the target/ build directory.
        // Binary lives at <workspace>/target/{debug,release}/dorea; python/ is a sibling of target/.
        let pythonpath = std::env::current_exe().ok()
            .and_then(|exe| {
                // .parent() = target/debug|release, .parent() = target, .parent() = workspace root
                exe.parent().and_then(|p| p.parent()).and_then(|p| p.parent())
                    .map(|root| root.join("python"))
            })
            .filter(|p| p.exists());
        if let Some(ref p) = pythonpath {
            cmd.env("PYTHONPATH", p);
        } else {
            log::debug!(
                "PYTHONPATH not set: could not derive python/ dir from binary path; \
                 ensure dorea_inference is installed in the Python environment"
            );
        }

        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()); // let stderr pass through for diagnostics

        let mut child = cmd.spawn()?;
        let mut stdin = child.stdin.take().ok_or_else(|| {
            InferenceError::Ipc("could not open inference server stdin".to_string())
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            InferenceError::Ipc("could not open inference server stdout".to_string())
        })?;

        // Send ping immediately so the server starts loading models ASAP.
        stdin.write_all(b"{\"type\":\"ping\"}\n")
            .and_then(|_| stdin.flush())
            .map_err(|e| InferenceError::Ipc(format!("write ping: {e}")))?;

        // Read pong in a background thread so we can enforce the startup timeout.
        // BufReader<ChildStdout> is Send, so this is safe.
        let timeout = config.startup_timeout;
        let (tx, rx) = mpsc::channel::<io::Result<(String, BufReader<std::process::ChildStdout>)>>();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            let result = reader.read_line(&mut line).map(|_| (line, reader));
            let _ = tx.send(result);
        });

        let (pong_line, stdout_reader) = match rx.recv_timeout(timeout) {
            Ok(Ok((line, reader))) => (line, reader),
            Ok(Err(e)) => {
                let _ = child.kill();
                return Err(InferenceError::Ipc(format!("read from server: {e}")));
            }
            Err(_) => {
                log::warn!("Inference server startup timed out after {timeout:?}");
                let _ = child.kill();
                return Err(InferenceError::Timeout);
            }
        };

        let trimmed = pong_line.trim_end_matches('\n').trim_end_matches('\r');
        if trimmed.is_empty() {
            let _ = child.kill();
            return Err(InferenceError::Ipc(
                "server sent empty response during startup (may have crashed)".to_string(),
            ));
        }
        let v: serde_json::Value = serde_json::from_str(trimmed)
            .map_err(|e| InferenceError::Ipc(format!("ping response parse error: {e}")))?;
        if v["type"].as_str() != Some("pong") {
            let _ = child.kill();
            return Err(InferenceError::Ipc(format!("expected pong, got: {trimmed}")));
        }

        Ok(Self { child, stdin, stdout_reader })
    }

    /// Send a ping and verify the server responds.
    pub fn ping(&mut self) -> Result<(), InferenceError> {
        self.send_line(r#"{"type":"ping"}"#)?;
        let resp = self.recv_line()?;
        let v: serde_json::Value = serde_json::from_str(&resp)
            .map_err(|e| InferenceError::Ipc(format!("ping response parse error: {e}")))?;
        if v["type"].as_str() == Some("pong") {
            Ok(())
        } else {
            Err(InferenceError::Ipc(format!("expected pong, got: {resp}")))
        }
    }

    /// Run RAUNE-Net on an RGB image.
    ///
    /// `image_rgb`: HxWx3 u8 array flattened row-major.
    /// Returns the enhanced image as HxWx3 u8.
    pub fn run_raune(
        &mut self,
        id: &str,
        image_rgb: &[u8],
        width: usize,
        height: usize,
        max_size: usize,
    ) -> Result<(Vec<u8>, usize, usize), InferenceError> {
        let b64 = B64.encode(image_rgb);

        let req = serde_json::json!({
            "type": "raune",
            "id": id,
            "image_b64": b64,
            "format": "raw_rgb",
            "width": width,
            "height": height,
            "max_size": max_size
        });
        self.send_line(&req.to_string())?;

        let resp = self.recv_line()?;
        let v: serde_json::Value = serde_json::from_str(&resp)
            .map_err(|e| InferenceError::Ipc(format!("raune response parse: {e}")))?;

        if v["type"].as_str() == Some("error") {
            return Err(InferenceError::ServerError(
                v["message"].as_str().unwrap_or("unknown error").to_string(),
            ));
        }
        if v["type"].as_str() != Some("raune_result") {
            return Err(InferenceError::Ipc(format!("unexpected response type: {resp}")));
        }

        let w = v["width"].as_u64().unwrap_or(0) as usize;
        let h = v["height"].as_u64().unwrap_or(0) as usize;
        let b64_out = v["image_b64"].as_str()
            .ok_or_else(|| InferenceError::Ipc("missing image_b64 in raune_result".to_string()))?;

        let png_bytes = B64.decode(b64_out)
            .map_err(|e| InferenceError::Ipc(format!("base64 decode: {e}")))?;
        let rgb = decode_png_bytes(&png_bytes)?;

        Ok((rgb, w, h))
    }

    /// Run Depth Anything V2 on an RGB image.
    ///
    /// Returns (depth_f32, out_width, out_height) at inference resolution.
    pub fn run_depth(
        &mut self,
        id: &str,
        image_rgb: &[u8],
        width: usize,
        height: usize,
        max_size: usize,
    ) -> Result<(Vec<f32>, usize, usize), InferenceError> {
        let b64 = B64.encode(image_rgb);

        let req = serde_json::json!({
            "type": "depth",
            "id": id,
            "image_b64": b64,
            "format": "raw_rgb",
            "width": width,
            "height": height,
            "max_size": max_size
        });
        self.send_line(&req.to_string())?;

        let resp = self.recv_line()?;
        let v: serde_json::Value = serde_json::from_str(&resp)
            .map_err(|e| InferenceError::Ipc(format!("depth response parse: {e}")))?;

        if v["type"].as_str() == Some("error") {
            return Err(InferenceError::ServerError(
                v["message"].as_str().unwrap_or("unknown error").to_string(),
            ));
        }
        if v["type"].as_str() != Some("depth_result") {
            return Err(InferenceError::Ipc(format!("unexpected response type: {resp}")));
        }

        let w = v["width"].as_u64().unwrap_or(0) as usize;
        let h = v["height"].as_u64().unwrap_or(0) as usize;
        let b64_out = v["depth_f32_b64"].as_str()
            .ok_or_else(|| InferenceError::Ipc("missing depth_f32_b64".to_string()))?;

        let raw = B64.decode(b64_out)
            .map_err(|e| InferenceError::Ipc(format!("base64 decode: {e}")))?;

        if raw.len() != w * h * 4 {
            return Err(InferenceError::Ipc(format!(
                "depth buffer size mismatch: got {} bytes, expected {}",
                raw.len(), w * h * 4
            )));
        }

        let depth: Vec<f32> = raw.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        Ok((depth, w, h))
    }

    /// Run Depth Anything V2 on a batch of images in a single IPC round-trip.
    ///
    /// Returns `Vec<(id, depth_f32, out_width, out_height)>` in the same order as `items`.
    #[allow(clippy::type_complexity)]
    pub fn run_depth_batch(
        &mut self,
        items: &[DepthBatchItem],
    ) -> Result<Vec<(String, Vec<f32>, usize, usize)>, InferenceError> {
        if items.is_empty() {
            return Ok(vec![]);
        }

        let json_items: Vec<serde_json::Value> = items.iter().map(|item| {
            let b64 = B64.encode(&item.pixels);
            serde_json::json!({
                "id": item.id,
                "image_b64": b64,
                "format": "raw_rgb",
                "width": item.width,
                "height": item.height,
                "max_size": item.max_size
            })
        }).collect();

        let req = serde_json::json!({
            "type": "depth_batch",
            "items": json_items
        });
        self.send_line(&req.to_string())?;

        let resp = self.recv_line()?;
        let v: serde_json::Value = serde_json::from_str(&resp)
            .map_err(|e| InferenceError::Ipc(format!("depth_batch response parse: {e}")))?;

        if v["type"].as_str() == Some("error") {
            return Err(InferenceError::ServerError(
                v["message"].as_str().unwrap_or("unknown error").to_string(),
            ));
        }
        if v["type"].as_str() != Some("depth_batch_result") {
            return Err(InferenceError::Ipc(format!("unexpected response type: {resp}")));
        }

        let results = v["results"].as_array()
            .ok_or_else(|| InferenceError::Ipc("missing results array in depth_batch_result".to_string()))?;

        let mut out = Vec::with_capacity(results.len());
        for r in results {
            let id = r["id"].as_str().unwrap_or("").to_string();
            let w = r["width"].as_u64().unwrap_or(0) as usize;
            let h = r["height"].as_u64().unwrap_or(0) as usize;
            let b64_out = r["depth_f32_b64"].as_str()
                .ok_or_else(|| InferenceError::Ipc(format!("missing depth_f32_b64 for id={id}")))?;

            let raw = B64.decode(b64_out)
                .map_err(|e| InferenceError::Ipc(format!("base64 decode for id={id}: {e}")))?;

            if raw.len() != w * h * 4 {
                return Err(InferenceError::Ipc(format!(
                    "depth buffer size mismatch for id={id}: got {} bytes, expected {}",
                    raw.len(), w * h * 4
                )));
            }

            let depth: Vec<f32> = raw.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();

            out.push((id, depth, w, h));
        }

        Ok(out)
    }

    /// Run RAUNE then Depth Anything on a batch, with the enhanced tensor staying on GPU
    /// between the two models. Returns enhanced RGB and depth map for each item.
    ///
    /// Returns `Vec<(id, enhanced_rgb_u8, enh_w, enh_h, depth_f32, depth_w, depth_h)>`.
    #[allow(clippy::type_complexity)]
    pub fn run_raune_depth_batch(
        &mut self,
        items: &[RauneDepthBatchItem],
    ) -> Result<Vec<(String, Vec<u8>, usize, usize, Vec<f32>, usize, usize)>, InferenceError> {
        if items.is_empty() {
            return Ok(vec![]);
        }

        let json_items: Vec<serde_json::Value> = items.iter().map(|item| {
            serde_json::json!({
                "id": item.id,
                "image_b64": B64.encode(&item.pixels),
                "format": "raw_rgb",
                "width": item.width,
                "height": item.height,
                "raune_max_size": item.raune_max_size,
                "depth_max_size": item.depth_max_size,
            })
        }).collect();

        let req = serde_json::json!({ "type": "raune_depth_batch", "items": json_items });
        self.send_line(&req.to_string())?;

        let resp = self.recv_line()?;
        let v: serde_json::Value = serde_json::from_str(&resp)
            .map_err(|e| InferenceError::Ipc(format!("raune_depth_batch parse: {e}")))?;

        if v["type"].as_str() == Some("error") {
            return Err(InferenceError::ServerError(
                v["message"].as_str().unwrap_or("unknown error").to_string(),
            ));
        }
        if v["type"].as_str() != Some("raune_depth_batch_result") {
            return Err(InferenceError::Ipc(format!("unexpected response type: {resp}")));
        }

        let results = v["results"].as_array()
            .ok_or_else(|| InferenceError::Ipc("missing results array in raune_depth_batch_result".to_string()))?;

        let mut out = Vec::with_capacity(results.len());
        for r in results {
            let id = r["id"].as_str().unwrap_or("").to_string();

            // Enhanced image (PNG-encoded)
            let enh_w = r["enhanced_width"].as_u64().unwrap_or(0) as usize;
            let enh_h = r["enhanced_height"].as_u64().unwrap_or(0) as usize;
            let enh_b64 = r["image_b64"].as_str()
                .ok_or_else(|| InferenceError::Ipc(format!("missing image_b64 for {id}")))?;
            let enh_png = B64.decode(enh_b64)
                .map_err(|e| InferenceError::Ipc(format!("base64 enh for {id}: {e}")))?;
            let enh_rgb = decode_png_bytes(&enh_png)?;
            if enh_rgb.len() != enh_w * enh_h * 3 {
                return Err(InferenceError::Ipc(format!(
                    "enhanced image size mismatch for {id}: got {} want {}",
                    enh_rgb.len(), enh_w * enh_h * 3
                )));
            }

            // Depth map (raw f32 LE)
            let depth_w = r["depth_width"].as_u64().unwrap_or(0) as usize;
            let depth_h = r["depth_height"].as_u64().unwrap_or(0) as usize;
            let depth_b64 = r["depth_f32_b64"].as_str()
                .ok_or_else(|| InferenceError::Ipc(format!("missing depth_f32_b64 for {id}")))?;
            let raw = B64.decode(depth_b64)
                .map_err(|e| InferenceError::Ipc(format!("base64 depth for {id}: {e}")))?;
            if raw.len() != depth_w * depth_h * 4 {
                return Err(InferenceError::Ipc(format!(
                    "depth size mismatch for {id}: got {} want {}",
                    raw.len(), depth_w * depth_h * 4
                )));
            }
            let depth: Vec<f32> = raw.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();

            out.push((id, enh_rgb, enh_w, enh_h, depth, depth_w, depth_h));
        }
        Ok(out)
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

    /// Send a graceful shutdown request and wait for the process to exit.
    pub fn shutdown(mut self) -> Result<(), InferenceError> {
        let _ = self.send_line(r#"{"type":"shutdown"}"#);
        let _ = self.child.wait();
        Ok(())
    }

    fn send_line(&mut self, line: &str) -> Result<(), InferenceError> {
        self.stdin.write_all(line.as_bytes())
            .and_then(|_| self.stdin.write_all(b"\n"))
            .and_then(|_| self.stdin.flush())
            .map_err(|e| InferenceError::Ipc(format!("write to server: {e}")))
    }

    fn recv_line(&mut self) -> Result<String, InferenceError> {
        let mut line = String::new();
        self.stdout_reader.read_line(&mut line)
            .map_err(|e| InferenceError::Ipc(format!("read from server: {e}")))?;
        let trimmed = line.trim_end_matches('\n').trim_end_matches('\r').to_string();
        if trimmed.is_empty() {
            return Err(InferenceError::Ipc("server sent empty response (may have crashed)".to_string()));
        }
        Ok(trimmed)
    }
}

impl Drop for InferenceServer {
    fn drop(&mut self) {
        // Best-effort shutdown
        let _ = self.stdin.write_all(b"{\"type\":\"shutdown\"}\n");
        let _ = self.child.wait();
    }
}

/// Encode RGB24 pixels to PNG bytes in memory.
/// Only used in tests (IPC switched to raw RGB in production).
#[cfg(test)]
fn encode_png_bytes(rgb: &[u8], width: usize, height: usize) -> Result<Vec<u8>, InferenceError> {
    // Minimal uncompressed PNG (for speed; compression is the Python server's concern)
    // Use the `image` crate idiom via raw bytes.
    // Since we don't have image crate here, build a minimal PNG manually.
    // Actually: use a simple raw RGB PNG via the lodepng-compatible format.
    // Simplest: use the `miniz_oxide` approach or just write raw PNG header.
    //
    // For now, produce a valid PNG using raw inflate + PNG chunk construction.
    // This avoids adding the `image` dep to dorea-video.

    let expected = width * height * 3;
    if rgb.len() < expected {
        return Err(InferenceError::ImageError(format!(
            "expected {} bytes, got {}", expected, rgb.len()
        )));
    }

    // Build PNG using miniz_oxide (already available transitively) or manual construction.
    // We'll use a dead-simple approach: prepend filter bytes and zlib-compress.
    let mut filtered = Vec::with_capacity(height * (1 + width * 3));
    for row in rgb.chunks_exact(width * 3) {
        filtered.push(0u8); // filter type = None
        filtered.extend_from_slice(row);
    }

    let compressed = deflate_zlib(&filtered);

    let mut out = Vec::new();
    // PNG signature
    out.extend_from_slice(b"\x89PNG\r\n\x1a\n");
    // IHDR
    write_png_chunk(&mut out, b"IHDR", &{
        let mut h = Vec::new();
        h.extend_from_slice(&(width as u32).to_be_bytes());
        h.extend_from_slice(&(height as u32).to_be_bytes());
        h.extend_from_slice(&[8u8, 2, 0, 0, 0]); // bit depth=8, color type=RGB, compression, filter, interlace
        h
    });
    // IDAT
    write_png_chunk(&mut out, b"IDAT", &compressed);
    // IEND
    write_png_chunk(&mut out, b"IEND", &[]);

    Ok(out)
}

#[cfg(test)]
fn deflate_zlib(data: &[u8]) -> Vec<u8> {
    // Use a simple store-mode zlib (no compression) for speed.
    // Format: zlib header (0x78 0x01) + DEFLATE stored blocks + adler32 checksum.
    let mut out = Vec::new();
    out.push(0x78); // zlib CMF: deflate, window size 32K
    out.push(0x01); // zlib FLG: no dict, check bits

    // DEFLATE store blocks (each up to 65535 bytes)
    let max_block = 65535_usize;
    let chunks: Vec<&[u8]> = data.chunks(max_block).collect();
    for (i, chunk) in chunks.iter().enumerate() {
        let last = i == chunks.len() - 1;
        out.push(if last { 1 } else { 0 }); // BFINAL + BTYPE=00 (store)
        let len = chunk.len() as u16;
        let nlen = !len;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&nlen.to_le_bytes());
        out.extend_from_slice(chunk);
    }

    // Adler-32 checksum
    let (mut s1, mut s2) = (1u32, 0u32);
    for &b in data {
        s1 = (s1 + b as u32) % 65521;
        s2 = (s2 + s1) % 65521;
    }
    out.extend_from_slice(&((s2 << 16) | s1).to_be_bytes());
    out
}

#[cfg(test)]
fn write_png_chunk(out: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(chunk_type);
    out.extend_from_slice(data);
    let crc = crc32(chunk_type, data);
    out.extend_from_slice(&crc.to_be_bytes());
}

#[cfg(test)]
fn crc32(chunk_type: &[u8], data: &[u8]) -> u32 {
    // CRC-32 (IEEE 802.3)
    let table: [u32; 256] = {
        let mut t = [0u32; 256];
        for i in 0..256u32 {
            let mut c = i;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            }
            t[i as usize] = c;
        }
        t
    };
    let mut crc = 0xFFFF_FFFFu32;
    for &b in chunk_type.iter().chain(data.iter()) {
        crc = table[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

/// Decode a PNG byte stream to RGB24 pixels.
fn decode_png_bytes(png: &[u8]) -> Result<Vec<u8>, InferenceError> {
    // Parse enough of the PNG to extract raw pixel data.
    // For simplicity, defer to a dependency-free approach: require 8-bit RGB PNG.
    // We'll rely on the Python side always producing valid 8-bit RGB PNGs.
    //
    // Since we don't have the `image` crate in dorea-video, we implement a
    // minimal PNG reader supporting only 8-bit RGB non-interlaced PNGs.
    parse_png_rgb(png)
        .map_err(InferenceError::ImageError)
}

fn parse_png_rgb(data: &[u8]) -> Result<Vec<u8>, String> {
    if data.len() < 8 || &data[..8] != b"\x89PNG\r\n\x1a\n" {
        return Err("not a PNG file".to_string());
    }

    let mut pos = 8;
    let mut width = 0usize;
    let mut height = 0usize;
    let mut idat = Vec::new();

    while pos + 12 <= data.len() {
        let length = u32::from_be_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
        let chunk_type = &data[pos+4..pos+8];
        if pos + 8 + length > data.len() {
            return Err(format!("truncated PNG chunk at offset {pos}"));
        }
        let chunk_data = &data[pos+8..pos+8+length];
        pos += 12 + length;

        match chunk_type {
            b"IHDR" if chunk_data.len() >= 13 => {
                width  = u32::from_be_bytes([chunk_data[0],chunk_data[1],chunk_data[2],chunk_data[3]]) as usize;
                height = u32::from_be_bytes([chunk_data[4],chunk_data[5],chunk_data[6],chunk_data[7]]) as usize;
                let bit_depth  = chunk_data[8];
                let color_type = chunk_data[9];
                let interlace  = chunk_data[12];
                if bit_depth != 8 || color_type != 2 || interlace != 0 {
                    return Err(format!(
                        "unsupported PNG: bit_depth={bit_depth} color_type={color_type} interlace={interlace}"
                    ));
                }
            }
            b"IDAT" => idat.extend_from_slice(chunk_data),
            b"IEND" => break,
            _ => {}
        }
    }

    if width == 0 || height == 0 {
        return Err("could not parse PNG dimensions".to_string());
    }

    let decompressed = inflate_zlib(&idat)
        .map_err(|e| format!("zlib inflate failed: {e}"))?;

    // Each row: 1 filter byte + stride pixel bytes
    let stride = width * 3;
    let expected_decompressed = height * (stride + 1);
    if decompressed.len() < expected_decompressed {
        return Err(format!(
            "decompressed PNG data too short: {} < {expected_decompressed}",
            decompressed.len()
        ));
    }

    let mut rgb = vec![0u8; width * height * 3];

    for row in 0..height {
        let src_off = row * (stride + 1);
        let filter = decompressed[src_off];
        let src = &decompressed[src_off+1..src_off+1+stride];
        let dst_start = row * stride;

        // Copy prior row before writing current row (avoids borrow-checker conflict between
        // reading the prior row and writing the current row within the same `rgb` slice).
        let prior: Vec<u8> = if row > 0 {
            rgb[(row - 1) * stride..row * stride].to_vec()
        } else {
            vec![0u8; stride]
        };

        match filter {
            0 => { // None
                rgb[dst_start..dst_start+stride].copy_from_slice(src);
            }
            1 => { // Sub
                for i in 0..stride {
                    let a = if i >= 3 { rgb[dst_start + i - 3] } else { 0 };
                    rgb[dst_start + i] = src[i].wrapping_add(a);
                }
            }
            2 => { // Up
                for i in 0..stride {
                    rgb[dst_start + i] = src[i].wrapping_add(prior[i]);
                }
            }
            3 => { // Average
                for i in 0..stride {
                    let a = if i >= 3 { rgb[dst_start + i - 3] as u16 } else { 0u16 };
                    let b = prior[i] as u16;
                    rgb[dst_start + i] = src[i].wrapping_add(((a + b) / 2) as u8);
                }
            }
            4 => { // Paeth
                for i in 0..stride {
                    let a = if i >= 3 { rgb[dst_start + i - 3] } else { 0 };
                    let b = prior[i];
                    let c = if i >= 3 { prior[i - 3] } else { 0 };
                    rgb[dst_start + i] = src[i].wrapping_add(paeth_predictor(a, b, c));
                }
            }
            _ => return Err(format!("unsupported PNG filter type {filter} at row {row}")),
        }
    }

    Ok(rgb)
}

/// PNG Paeth predictor function (PNG spec section 9.4).
fn paeth_predictor(a: u8, b: u8, c: u8) -> u8 {
    let a = a as i32;
    let b = b as i32;
    let c = c as i32;
    let p = a + b - c;
    let pa = (p - a).abs();
    let pb = (p - b).abs();
    let pc = (p - c).abs();
    if pa <= pb && pa <= pc { a as u8 }
    else if pb <= pc { b as u8 }
    else { c as u8 }
}

fn inflate_zlib(data: &[u8]) -> Result<Vec<u8>, String> {
    use flate2::read::ZlibDecoder;
    use std::io::Read;
    let mut out = Vec::new();
    ZlibDecoder::new(data)
        .read_to_end(&mut out)
        .map_err(|e| e.to_string())?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raune_depth_batch_item_fields_compile() {
        let item = RauneDepthBatchItem {
            id: "kf0000".to_string(),
            pixels: vec![0u8; 60 * 80 * 3],
            width: 80,
            height: 60,
            raune_max_size: 80,
            depth_max_size: 56,
        };
        assert_eq!(item.id, "kf0000");
    }

    #[test]
    fn upscale_depth_identity() {
        let depth: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
        let out = InferenceServer::upscale_depth(&depth, 2, 2, 2, 2);
        assert_eq!(out, depth, "2x2→2x2 should be identity");
    }

    #[test]
    fn upscale_depth_bilinear() {
        let depth: Vec<f32> = vec![0.0, 1.0, 0.0, 0.0];
        // 2x2 → 3x3
        let out = InferenceServer::upscale_depth(&depth, 2, 2, 3, 3);
        assert_eq!(out.len(), 9);
        // Top-left corner = 0.0
        assert!((out[0] - 0.0).abs() < 1e-5, "top-left should be 0.0");
        // Top-right corner = 1.0
        assert!((out[2] - 1.0).abs() < 1e-5, "top-right should be 1.0");
    }

    #[test]
    fn png_encode_roundtrip_store_mode() {
        let rgb: Vec<u8> = (0..12u8).collect(); // 2x2 RGB
        let png = encode_png_bytes(&rgb, 2, 2).expect("encode failed");
        // Should have valid PNG signature
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n", "PNG signature mismatch");
        // Decode should work (store-mode deflate)
        let decoded = decode_png_bytes(&png).expect("decode failed");
        assert_eq!(decoded, rgb, "roundtrip mismatch");
    }

    #[test]
    fn spawn_command_includes_maxine_flags() {
        let config = InferenceConfig {
            maxine: true,
            maxine_upscale_factor: 2,
            ..InferenceConfig::default()
        };
        let args = config.build_args();
        assert!(args.contains(&"--maxine".to_string()), "missing --maxine");
        assert!(args.contains(&"--maxine-upscale-factor".to_string()), "missing --maxine-upscale-factor");
        assert!(args.contains(&"2".to_string()), "missing upscale factor value");
    }

    #[test]
    fn spawn_command_omits_maxine_when_disabled() {
        let config = InferenceConfig {
            maxine: false,
            ..InferenceConfig::default()
        };
        let args = config.build_args();
        assert!(!args.contains(&"--maxine".to_string()), "--maxine should be absent");
    }
}
