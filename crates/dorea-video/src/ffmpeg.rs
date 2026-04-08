// ffmpeg subprocess integration: probe, decode, encode, single-frame extraction.

use std::io::{self, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStderr, ChildStdin, Command, Stdio};
use std::sync::OnceLock;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FfmpegError {
    #[error("ffmpeg not found in PATH: {0}")]
    NotFound(#[from] io::Error),
    #[error("ffprobe failed: {0}")]
    ProbeFailed(String),
    #[error("ffmpeg decode failed: {0}")]
    DecodeFailed(String),
    #[error("ffmpeg encode failed: {0}")]
    EncodeFailed(String),
    #[error("invalid video metadata: {0}")]
    InvalidMetadata(String),
}

/// Video metadata returned by `probe`.
#[derive(Debug, Clone)]
pub struct VideoInfo {
    pub width: usize,
    pub height: usize,
    /// Frames per second (rational rounded to f64).
    pub fps: f64,
    /// Duration in seconds.
    pub duration_secs: f64,
    /// Estimated frame count (may be 0 if unknown).
    pub frame_count: u64,
    /// Has audio stream.
    pub has_audio: bool,
    /// Codec name as reported by ffprobe, e.g. `"hevc"`, `"prores"`, `"h264"`.
    pub codec_name: String,
    /// Pixel format as reported by ffprobe, e.g. `"yuv420p10le"`, `"yuv422p10le"`.
    pub pix_fmt: String,
    /// Bits per colour component (8, 10, 12). Falls back to 0 if unknown.
    pub bits_per_component: u8,
}

/// Probe a video file for metadata.
pub fn probe(input: &Path) -> Result<VideoInfo, FfmpegError> {
    let output = Command::new("ffprobe")
        .args([
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            input.to_str().unwrap_or(""),
        ])
        .output()
        .map_err(FfmpegError::NotFound)?;

    if !output.status.success() {
        return Err(FfmpegError::ProbeFailed(
            String::from_utf8_lossy(&output.stderr).to_string(),
        ));
    }

    let json: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|e| FfmpegError::ProbeFailed(e.to_string()))?;

    let streams = json["streams"].as_array().ok_or_else(|| {
        FfmpegError::InvalidMetadata("no streams found".to_string())
    })?;

    let video = streams
        .iter()
        .find(|s| s["codec_type"].as_str() == Some("video"))
        .ok_or_else(|| FfmpegError::InvalidMetadata("no video stream".to_string()))?;

    let width = video["width"]
        .as_u64()
        .ok_or_else(|| FfmpegError::InvalidMetadata("missing width".to_string()))? as usize;
    let height = video["height"]
        .as_u64()
        .ok_or_else(|| FfmpegError::InvalidMetadata("missing height".to_string()))? as usize;

    // fps as rational string "30000/1001" or plain "30"
    let fps = parse_rational(video["r_frame_rate"].as_str().unwrap_or("30/1"));

    let duration_secs = json["format"]["duration"]
        .as_str()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);

    let frame_count = video["nb_frames"]
        .as_str()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or_else(|| (duration_secs * fps).round() as u64);

    let has_audio = streams
        .iter()
        .any(|s| s["codec_type"].as_str() == Some("audio"));

    let codec_name = video["codec_name"]
        .as_str()
        .unwrap_or("unknown")
        .to_string();
    let pix_fmt = video["pix_fmt"]
        .as_str()
        .unwrap_or("unknown")
        .to_string();
    let bits_per_component = video["bits_per_raw_sample"]
        .as_str()
        .and_then(|s| s.parse::<u8>().ok())
        .unwrap_or_else(|| {
            if pix_fmt.contains("10") {
                10
            } else if pix_fmt.contains("12") {
                12
            } else {
                8
            }
        });

    Ok(VideoInfo {
        width,
        height,
        fps,
        duration_secs,
        frame_count,
        has_audio,
        codec_name,
        pix_fmt,
        bits_per_component,
    })
}

fn parse_rational(s: &str) -> f64 {
    if let Some((num, den)) = s.split_once('/') {
        let n = num.trim().parse::<f64>().unwrap_or(30.0);
        let d = den.trim().parse::<f64>().unwrap_or(1.0);
        if d == 0.0 { 30.0 } else { n / d }
    } else {
        s.parse::<f64>().unwrap_or(30.0)
    }
}

/// A decoded video frame.
#[derive(Debug)]
pub struct Frame {
    pub index: u64,
    pub pixels: Vec<u8>, // RGB24, width * height * 3
    pub width: usize,
    pub height: usize,
}

/// Decode all frames from a video file.
///
/// Yields frames in display order. Uses NVDEC if available, falls back to software.
pub fn decode_frames(
    input: &Path,
    info: &VideoInfo,
) -> Result<impl Iterator<Item = Result<Frame, FfmpegError>>, FfmpegError> {
    let child = spawn_decoder(input, info)?;
    Ok(FrameReader::new(child, info.width, info.height))
}

fn spawn_decoder(input: &Path, info: &VideoInfo) -> Result<Child, FfmpegError> {
    let input_str = input.to_str().unwrap_or("");
    let size_str = format!("{}x{}", info.width, info.height);

    // Try hardware decode first, fall back to software
    let hw_result = Command::new("ffmpeg")
        .args([
            "-hwaccel", "nvdec",
            "-i", input_str,
            "-vf", &format!("scale={size_str}"),
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "pipe:1",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn();

    if let Ok(child) = hw_result {
        return Ok(child);
    } else if let Err(ref e) = hw_result {
        log::debug!("nvdec spawn failed ({e}), falling back to software decode");
    }

    // Software fallback
    Command::new("ffmpeg")
        .args([
            "-i", input_str,
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "pipe:1",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(FfmpegError::NotFound)
}

struct FrameReader {
    child: Child,
    frame_index: u64,
    width: usize,
    height: usize,
    frame_bytes: usize,
    done: bool,
}

impl FrameReader {
    fn new(child: Child, width: usize, height: usize) -> Self {
        Self {
            child,
            frame_index: 0,
            width,
            height,
            frame_bytes: width * height * 3,
            done: false,
        }
    }
}

impl Drop for FrameReader {
    fn drop(&mut self) {
        // Kill the child so it doesn't block waiting for a reader that's gone.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Iterator for FrameReader {
    type Item = Result<Frame, FfmpegError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let mut buf = vec![0u8; self.frame_bytes];
        let stdout = self.child.stdout.as_mut()?;

        match read_exact(stdout, &mut buf) {
            Ok(0) | Err(_) => {
                self.done = true;
                return None;
            }
            Ok(_) => {}
        }

        let frame = Frame {
            index: self.frame_index,
            pixels: buf,
            width: self.width,
            height: self.height,
        };
        self.frame_index += 1;
        Some(Ok(frame))
    }
}

/// Read exactly `buf.len()` bytes.
///
/// Returns `Ok(0)` only on a clean EOF before any bytes are read (end of stream).
/// Returns `Err(UnexpectedEof)` if the stream ends mid-frame (partial frame).
fn read_exact(reader: &mut dyn Read, buf: &mut [u8]) -> io::Result<usize> {
    let mut pos = 0;
    while pos < buf.len() {
        match reader.read(&mut buf[pos..]) {
            Ok(0) if pos == 0 => return Ok(0), // clean EOF before frame starts
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("partial frame: got {pos}/{} bytes", buf.len()),
                ));
            }
            Ok(n) => pos += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(pos)
}

/// Decode all frames from a video file at a custom (scaled) resolution.
///
/// Uses ffmpeg `-vf scale=WxH`. If the requested size matches `info`, delegates
/// to the existing `decode_frames` path to avoid an extra scale filter.
pub fn decode_frames_scaled(
    input: &Path,
    info: &VideoInfo,
    width: usize,
    height: usize,
) -> Result<impl Iterator<Item = Result<Frame, FfmpegError>>, FfmpegError> {
    let child = if width == info.width && height == info.height {
        spawn_decoder(input, info)?
    } else {
        spawn_decoder_at(input, width, height)?
    };
    Ok(FrameReader::new(child, width, height))
}

fn spawn_decoder_at(input: &Path, width: usize, height: usize) -> Result<Child, FfmpegError> {
    let input_str = input.to_str().unwrap_or("");
    let scale_str = format!("scale={width}x{height}");

    let hw_result = Command::new("ffmpeg")
        .args([
            "-hwaccel", "nvdec",
            "-i", input_str,
            "-vf", &scale_str,
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "pipe:1",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn();

    if let Ok(child) = hw_result {
        return Ok(child);
    } else if let Err(ref e) = hw_result {
        log::debug!("nvdec scaled spawn failed ({e}), falling back to software decode");
    }

    Command::new("ffmpeg")
        .args([
            "-i", input_str,
            "-vf", &scale_str,
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "pipe:1",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(FfmpegError::NotFound)
}

/// Extract a single frame at a given timestamp (seconds).
pub fn extract_frame_at(
    input: &Path,
    timestamp_secs: f64,
    width: usize,
    height: usize,
) -> Result<Vec<u8>, FfmpegError> {
    let ts_str = format!("{timestamp_secs:.3}");
    let scale_str = format!("scale={width}x{height}");

    let output = Command::new("ffmpeg")
        .args([
            "-ss", &ts_str,
            "-i", input.to_str().unwrap_or(""),
            "-vframes", "1",
            "-vf", &scale_str,
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "pipe:1",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .map_err(FfmpegError::NotFound)?;

    let expected = width * height * 3;
    if output.stdout.len() < expected {
        return Err(FfmpegError::DecodeFailed(format!(
            "expected {expected} bytes, got {} at ts={timestamp_secs:.3}",
            output.stdout.len()
        )));
    }
    Ok(output.stdout[..expected].to_vec())
}

/// Encoder for streaming frames to an output video file.
pub struct FrameEncoder {
    child: Child,
    stdin: ChildStdin,
    stderr: Option<ChildStderr>,
    frame_bytes: usize,
}

impl FrameEncoder {
    /// Spawn an encoder subprocess.
    ///
    /// `input_video` is used only for audio passthrough (`-i <input>` for the audio stream).
    /// Pass `None` to skip audio.
    pub fn new(
        output: &Path,
        width: usize,
        height: usize,
        fps: f64,
        input_for_audio: Option<&Path>,
    ) -> Result<Self, FfmpegError> {
        // Validate output directory exists before doing any expensive work.
        if let Some(parent) = output.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                return Err(FfmpegError::EncodeFailed(format!(
                    "output directory does not exist: {}",
                    parent.display()
                )));
            }
        }

        let w_s = width.to_string();
        let h_s = height.to_string();
        let fps_s = format!("{fps:.3}");
        let size_s = format!("{w_s}x{h_s}");
        let out_s = output.to_str().unwrap_or("output.mp4");

        let mut cmd = Command::new("ffmpeg");
        cmd.args([
            "-y",
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-s", &size_s,
            "-r", &fps_s,
            "-i", "pipe:0",
        ]);

        if let Some(audio_src) = input_for_audio {
            cmd.args(["-i", audio_src.to_str().unwrap_or("")]);
            cmd.args(["-map", "0:v", "-map", "1:a", "-c:a", "copy"]);
        } else {
            cmd.args(["-map", "0:v"]);
        }

        // Try NVENC first
        let nvenc_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "18"];
        let sw_args = ["-c:v", "libx264", "-crf", "18", "-preset", "fast"];

        // Probe whether nvenc is available (cached across FrameEncoder instances).
        static NVENC_AVAILABLE: OnceLock<bool> = OnceLock::new();
        let nvenc_available = *NVENC_AVAILABLE.get_or_init(|| {
            Command::new("ffmpeg")
                .args(["-hide_banner", "-encoders"])
                .output()
                .map(|o| String::from_utf8_lossy(&o.stdout).contains("h264_nvenc"))
                .unwrap_or(false)
        });

        let codec_args: &[&str] = if nvenc_available { &nvenc_args } else { &sw_args };
        cmd.args(codec_args);
        cmd.arg(out_s);
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped()); // capture stderr so we can report ffmpeg errors

        let mut child = cmd.spawn().map_err(FfmpegError::NotFound)?;
        let stdin = child.stdin.take().ok_or_else(|| {
            FfmpegError::EncodeFailed("could not open encoder stdin".to_string())
        })?;
        let stderr = child.stderr.take();

        // Check immediately if ffmpeg exited (e.g. bad codec args, permissions, etc.)
        if let Ok(Some(status)) = child.try_wait() {
            let stderr_msg = stderr
                .map(|mut s| {
                    let mut buf = String::new();
                    let _ = s.read_to_string(&mut buf);
                    buf
                })
                .unwrap_or_default();
            return Err(FfmpegError::EncodeFailed(format!(
                "ffmpeg encoder exited immediately (code {:?}): {}",
                status.code(),
                stderr_msg.trim()
            )));
        }

        Ok(Self {
            child,
            stdin,
            stderr,
            frame_bytes: width * height * 3,
        })
    }

    /// Create a lossless ffv1/mkv encoder for temporary intermediate storage.
    ///
    /// Output is a lossless Matroska video (no audio, no NVENC attempt).
    /// Intended for the Maxine-enhanced frame temp file between Pass 1 and Pass 2.
    pub fn new_lossless_temp(
        output: &Path,
        width: usize,
        height: usize,
        fps: f64,
    ) -> Result<Self, FfmpegError> {
        let w_s = width.to_string();
        let h_s = height.to_string();
        let fps_s = format!("{fps:.3}");
        let size_s = format!("{w_s}x{h_s}");
        let out_s = output.to_str().unwrap_or("temp.mkv");

        let mut cmd = Command::new("ffmpeg");
        cmd.args([
            "-y",
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-s", &size_s,
            "-r", &fps_s,
            "-i", "pipe:0",
            "-map", "0:v",
            "-c:v", "ffv1",
            "-level", "3",
            out_s,
        ]);
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(FfmpegError::NotFound)?;
        let stdin = child.stdin.take().ok_or_else(|| {
            FfmpegError::EncodeFailed("could not open lossless encoder stdin".to_string())
        })?;
        let stderr = child.stderr.take();

        if let Ok(Some(status)) = child.try_wait() {
            let msg = stderr
                .map(|mut s| {
                    let mut buf = String::new();
                    let _ = s.read_to_string(&mut buf);
                    buf
                })
                .unwrap_or_default();
            return Err(FfmpegError::EncodeFailed(format!(
                "ffv1 encoder exited immediately (code {:?}): {}",
                status.code(),
                msg.trim()
            )));
        }

        Ok(Self {
            child,
            stdin,
            stderr,
            frame_bytes: width * height * 3,
        })
    }

    /// Write one RGB24 frame to the encoder.
    pub fn write_frame(&mut self, pixels: &[u8]) -> Result<(), FfmpegError> {
        if pixels.len() != self.frame_bytes {
            return Err(FfmpegError::EncodeFailed(format!(
                "frame size mismatch: expected {} bytes, got {}",
                self.frame_bytes,
                pixels.len()
            )));
        }
        if let Err(e) = self.stdin.write_all(pixels) {
            // Broken pipe means ffmpeg exited — check its stderr for a real error message.
            let stderr_msg = self.stderr.as_mut().map(|s| {
                let mut buf = String::new();
                let _ = s.read_to_string(&mut buf);
                buf
            }).unwrap_or_default();
            let exit_code = self.child.try_wait().ok().flatten()
                .and_then(|s| s.code())
                .map(|c| format!(" (exit code {c})"))
                .unwrap_or_default();
            let detail = if stderr_msg.trim().is_empty() {
                format!("{e}{exit_code}")
            } else {
                format!("{}{exit_code}: {}", e, stderr_msg.trim())
            };
            return Err(FfmpegError::EncodeFailed(format!("ffmpeg encoder died: {detail}")));
        }
        Ok(())
    }

    /// Finalize encoding. Must be called after all frames are written.
    pub fn finish(mut self) -> Result<(), FfmpegError> {
        drop(self.stdin); // close stdin → signal EOF to ffmpeg

        // Drain stderr in a background thread to prevent pipe-buffer deadlock:
        // if ffmpeg writes enough progress output to fill the 64KB pipe buffer
        // it will block, and wait() below would deadlock.
        let stderr_thread = self.stderr.take().map(|mut s| {
            std::thread::spawn(move || {
                let mut buf = String::new();
                let _ = s.read_to_string(&mut buf);
                buf
            })
        });

        let status = self.child.wait()
            .map_err(|e| FfmpegError::EncodeFailed(format!("failed to wait for encoder: {e}")))?;

        let stderr_msg = stderr_thread
            .and_then(|t| t.join().ok())
            .unwrap_or_default();

        if !status.success() {
            return Err(FfmpegError::EncodeFailed(format!(
                "ffmpeg encoder exited with code {:?}: {}",
                status.code(),
                stderr_msg.trim()
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rational_parse() {
        assert!((parse_rational("30000/1001") - 29.97).abs() < 0.01);
        assert!((parse_rational("25/1") - 25.0).abs() < 0.001);
        assert!((parse_rational("24") - 24.0).abs() < 0.001);
        assert!((parse_rational("0/0") - 30.0).abs() < 0.001); // div-by-zero fallback
    }

    #[test]
    fn video_info_has_codec_fields() {
        let info = VideoInfo {
            width: 3840,
            height: 2160,
            fps: 29.97,
            duration_secs: 10.0,
            frame_count: 300,
            has_audio: true,
            codec_name: "hevc".to_string(),
            pix_fmt: "yuv420p10le".to_string(),
            bits_per_component: 10,
        };
        assert_eq!(info.bits_per_component, 10);
        assert_eq!(info.codec_name, "hevc");
        assert_eq!(info.pix_fmt, "yuv420p10le");
    }

    #[test]
    fn lossless_temp_encoder_creates_decodable_file() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("dorea_test_ffv1_{}.mkv", std::process::id()));
        let width = 8usize;
        let height = 4usize;
        let fps = 30.0f64;

        // Write 3 frames
        let mut enc = FrameEncoder::new_lossless_temp(&path, width, height, fps)
            .expect("failed to create lossless encoder");
        let frame: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();
        for _ in 0..3 {
            enc.write_frame(&frame).expect("write_frame failed");
        }
        enc.finish().expect("finish failed");

        // File must exist and be non-empty
        assert!(path.exists(), "output file does not exist");
        assert!(path.metadata().unwrap().len() > 0, "output file is empty");

        // Must be decodable as a video
        let probe_result = probe(&path);
        assert!(probe_result.is_ok(), "ffprobe failed: {:?}", probe_result);
        let info = probe_result.unwrap();
        assert_eq!(info.width, width);
        assert_eq!(info.height, height);

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }
}
