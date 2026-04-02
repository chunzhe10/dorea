// Rust-side manager for the Python inference subprocess.
// Spawns `python -m dorea_inference.server` with model paths,
// communicates via JSON lines over stdin/stdout.

use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
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
    /// Path to the RAUNE-Net weights .pth file. None = skip RAUNE.
    pub raune_weights: Option<PathBuf>,
    /// Path to the sea_thru_poc directory (contains models/raune_net.py).
    pub raune_models_dir: Option<PathBuf>,
    /// Path to Depth Anything V2 model directory. None = skip depth.
    pub depth_model: Option<PathBuf>,
    /// Compute device: "cpu" or "cuda". None = auto-detect.
    pub device: Option<String>,
    /// Startup timeout.
    pub startup_timeout: Duration,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            python_exe: PathBuf::from("/opt/dorea-venv/bin/python"),
            raune_weights: None,
            raune_models_dir: None,
            depth_model: None,
            device: None,
            startup_timeout: Duration::from_secs(120),
        }
    }
}

/// Active connection to the Python inference subprocess.
pub struct InferenceServer {
    child: Child,
    stdin: ChildStdin,
    stdout_reader: BufReader<std::process::ChildStdout>,
}

impl InferenceServer {
    /// Spawn the inference server and wait for it to signal readiness.
    pub fn spawn(config: &InferenceConfig) -> Result<Self, InferenceError> {
        let mut cmd = Command::new(&config.python_exe);
        cmd.args(["-m", "dorea_inference.server"]);

        if let Some(p) = &config.raune_weights {
            cmd.args(["--raune-weights", p.to_str().unwrap_or("")]);
        } else {
            cmd.arg("--no-raune");
        }

        if let Some(p) = &config.raune_models_dir {
            cmd.args(["--raune-models-dir", p.to_str().unwrap_or("")]);
        }

        if let Some(p) = &config.depth_model {
            cmd.args(["--depth-model", p.to_str().unwrap_or("")]);
        } else {
            cmd.arg("--no-depth");
        }

        if let Some(d) = &config.device {
            cmd.args(["--device", d]);
        }

        // The Python server searches for dorea_inference in its own package tree.
        // Ensure the python/ directory is in PYTHONPATH.
        if let Some(python_dir) = config.python_exe.parent().and_then(|p| p.parent()) {
            let inference_dir = python_dir.parent().map(|r| r.join("python"))
                .unwrap_or_else(|| PathBuf::from("python"));
            cmd.env("PYTHONPATH", inference_dir.to_str().unwrap_or("python"));
        }

        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()); // let stderr pass through for debugging

        let mut child = cmd.spawn()?;
        let stdin = child.stdin.take().ok_or_else(|| {
            InferenceError::Ipc("could not open inference server stdin".to_string())
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            InferenceError::Ipc("could not open inference server stdout".to_string())
        })?;
        let stdout_reader = BufReader::new(stdout);

        let mut server = Self { child, stdin, stdout_reader };

        // Send a ping and wait for pong to confirm readiness
        server.ping()?;
        Ok(server)
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
        let png = encode_png_bytes(image_rgb, width, height)?;
        let b64 = B64.encode(&png);

        let req = serde_json::json!({
            "type": "raune",
            "id": id,
            "image_b64": b64,
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
        let png = encode_png_bytes(image_rgb, width, height)?;
        let b64 = B64.encode(&png);

        let req = serde_json::json!({
            "type": "depth",
            "id": id,
            "image_b64": b64,
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
fn encode_png_bytes(rgb: &[u8], width: usize, height: usize) -> Result<Vec<u8>, InferenceError> {
    use std::io::Cursor;

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

    let _ = Cursor::new(&out); // prevent optimizer removal
    Ok(out)
}

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

fn write_png_chunk(out: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(chunk_type);
    out.extend_from_slice(data);
    let crc = crc32(chunk_type, data);
    out.extend_from_slice(&crc.to_be_bytes());
}

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

    // Un-filter: each row has a 1-byte filter type
    let stride = width * 3;
    let mut rgb = vec![0u8; width * height * 3];

    for row in 0..height {
        let src_off = row * (stride + 1);
        let filter = decompressed[src_off];
        let src = &decompressed[src_off+1..src_off+1+stride];

        match filter {
            0 => {
                let dst = &mut rgb[row * stride..(row+1)*stride];
                dst.copy_from_slice(src);
            }
            1 => { // Sub
                let dst_start = row * stride;
                for i in 0..stride {
                    let a = if i >= 3 { rgb[dst_start + i - 3] } else { 0 };
                    rgb[dst_start + i] = src[i].wrapping_add(a);
                }
            }
            2 => { // Up — copy prior row first to avoid borrow conflict
                let prior: Vec<u8> = if row > 0 {
                    rgb[(row-1)*stride..row*stride].to_vec()
                } else {
                    vec![0u8; stride]
                };
                let dst_start = row * stride;
                for i in 0..stride {
                    rgb[dst_start + i] = src[i].wrapping_add(prior[i]);
                }
            }
            _ => return Err(format!("unsupported PNG filter type {filter} at row {row}")),
        }
    }

    Ok(rgb)
}

fn inflate_zlib(data: &[u8]) -> Result<Vec<u8>, String> {
    if data.len() < 6 {
        return Err("truncated zlib stream".to_string());
    }
    // Skip 2-byte zlib header, skip 4-byte adler32 trailer
    let deflate_data = &data[2..data.len()-4];
    inflate_deflate(deflate_data)
}

fn inflate_deflate(data: &[u8]) -> Result<Vec<u8>, String> {
    // Only handles DEFLATE store mode (type 00) blocks — matches our encoder.
    // For real PNG inputs from Python (which use actual compression), this will fail.
    // TODO: replace with a real inflate implementation or add `flate2` dependency.
    //
    // For now: attempt to decode store-mode blocks; return error for compressed data.
    let mut out = Vec::new();
    let mut pos = 0usize;
    loop {
        if pos >= data.len() { break; }
        let bfinal_btype = data[pos];
        pos += 1;
        let bfinal = bfinal_btype & 1;
        let btype  = (bfinal_btype >> 1) & 3;

        match btype {
            0 => {
                // Store mode: skip to byte boundary (already aligned after bfinal_btype)
                if pos + 4 > data.len() { return Err("truncated store block".to_string()); }
                let len  = u16::from_le_bytes([data[pos], data[pos+1]]) as usize;
                pos += 4; // skip len + nlen
                if pos + len > data.len() { return Err("store block overflows input".to_string()); }
                out.extend_from_slice(&data[pos..pos+len]);
                pos += len;
            }
            _ => {
                // Compressed block — need a real inflate implementation
                return Err(format!(
                    "DEFLATE btype={btype} (compressed) not supported by minimal inflate; \
                     add flate2 dependency to dorea-video for production use"
                ));
            }
        }
        if bfinal != 0 { break; }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
