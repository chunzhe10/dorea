// `dorea preview` — before/after contact sheet (5-10 frames).

use std::path::PathBuf;
use std::time::Duration;
use clap::Args;
use anyhow::{Context, Result};
use image::{Rgb, RgbImage};

use dorea_cal::Calibration;
use dorea_gpu::{grade_frame, GradeParams};
use dorea_video::ffmpeg;
use dorea_video::inference::{InferenceConfig, InferenceServer};

#[derive(Args, Debug)]
pub struct PreviewArgs {
    /// Input video file
    #[arg(long)]
    pub input: PathBuf,

    /// Pre-computed .dorea-cal file (optional — auto-derives from sampled frames)
    #[arg(long)]
    pub calibration: Option<PathBuf>,

    /// Output contact sheet PNG [default: preview.png]
    #[arg(long, default_value = "preview.png")]
    pub output: PathBuf,

    /// Number of frames to sample [default: 8]
    #[arg(long, default_value = "8")]
    pub frames: usize,

    /// Warmth multiplier
    #[arg(long, default_value = "1.0")]
    pub warmth: f32,

    /// Blend strength
    #[arg(long, default_value = "0.8")]
    pub strength: f32,

    /// Contrast multiplier
    #[arg(long, default_value = "1.0")]
    pub contrast: f32,

    /// Path to RAUNE-Net weights .pth
    #[arg(long)]
    pub raune_weights: Option<PathBuf>,

    /// Path to RAUNE-Net checkout directory (contains models/raune_net.py).
    /// Also accepts the parent sea_thru_poc dir — auto-descends to RAUNE-Net/.
    #[arg(long)]
    pub raune_models_dir: Option<PathBuf>,

    /// Path to Depth Anything V2 model directory
    #[arg(long)]
    pub depth_model: Option<PathBuf>,

    /// Python executable
    #[arg(long, default_value = "/opt/dorea-venv/bin/python")]
    pub python: PathBuf,

    /// Force CPU-only mode
    #[arg(long)]
    pub cpu_only: bool,

    /// Enable verbose logging
    #[arg(short, long)]
    pub verbose: bool,
}

pub fn run(args: PreviewArgs) -> Result<()> {
    log::info!("Generating preview for {}", args.input.display());

    let info = ffmpeg::probe(&args.input)
        .context("ffprobe failed")?;
    log::info!("Input: {}x{} @ {:.2}fps, {:.1}s", info.width, info.height, info.fps, info.duration_secs);

    let n = args.frames.clamp(1, 20);
    let params = GradeParams {
        warmth: args.warmth,
        strength: args.strength,
        contrast: args.contrast,
    };

    // Decide preview tile size (max 512px wide per tile)
    let tile_w = info.width.min(512);
    let tile_h = (info.height as f64 * tile_w as f64 / info.width as f64).round() as usize;

    // Sample evenly-spaced timestamps
    let timestamps: Vec<f64> = (0..n)
        .map(|i| (i as f64 + 0.5) * info.duration_secs / n as f64)
        .collect();

    log::info!("Sampling {n} frames at {tile_w}x{tile_h} tile size...");

    // Spawn inference server
    let inf_cfg = InferenceConfig {
        python_exe: args.python.clone(),
        raune_weights: args.raune_weights.clone(),
        raune_models_dir: args.raune_models_dir.clone(),
        depth_model: args.depth_model.clone(),
        device: if args.cpu_only { Some("cpu".to_string()) } else { None },
        startup_timeout: Duration::from_secs(180),
    };
    let mut inf_server = InferenceServer::spawn(&inf_cfg)
        .context("failed to spawn inference server")?;

    // Load or auto-derive calibration
    let calibration = if let Some(cal_path) = &args.calibration {
        Calibration::load(cal_path).context("failed to load .dorea-cal")?
    } else {
        log::info!("Auto-deriving calibration from sampled frames...");
        use crate::calibrate::{run_calibration_from_frames, CalibrationInput};
        let mut inputs = Vec::new();
        for (i, &ts) in timestamps.iter().enumerate() {
            let pixels = ffmpeg::extract_frame_at(&args.input, ts, tile_w, tile_h)
                .with_context(|| format!("extract frame at {ts:.2}s failed"))?;
            let id = format!("p{i:02}");
            let target = inf_server.run_raune(&id, &pixels, tile_w, tile_h, 1024)
                .map(|(t, _, _)| t)
                .unwrap_or_else(|_| pixels.clone());
            let (depth_proxy, dw, dh) = inf_server.run_depth(&id, &pixels, tile_w, tile_h, 518)
                .unwrap_or_else(|_| (vec![0.5f32; tile_w * tile_h], tile_w, tile_h));
            let depth = InferenceServer::upscale_depth(&depth_proxy, dw, dh, tile_w, tile_h);
            inputs.push(CalibrationInput { pixels, target, depth, width: tile_w, height: tile_h });
        }
        run_calibration_from_frames(&inputs).context("auto-calibration failed")?
    };

    // Build contact sheet: 2 rows (original / graded), n columns
    let sheet_w = tile_w * n;
    let sheet_h = tile_h * 2;
    let mut sheet = RgbImage::new(sheet_w as u32, sheet_h as u32);

    for (i, &ts) in timestamps.iter().enumerate() {
        let orig = ffmpeg::extract_frame_at(&args.input, ts, tile_w, tile_h)
            .with_context(|| format!("extract frame at {ts:.2}s failed"))?;

        let id = format!("prev{i:02}");
        let (depth_proxy, dw, dh) = inf_server.run_depth(&id, &orig, tile_w, tile_h, 518)
            .unwrap_or_else(|_| (vec![0.5f32; tile_w * tile_h], tile_w, tile_h));
        let depth = InferenceServer::upscale_depth(&depth_proxy, dw, dh, tile_w, tile_h);

        let graded = grade_frame(&orig, &depth, tile_w, tile_h, &calibration, &params)
            .unwrap_or_else(|_| orig.clone());

        // Copy original into top row
        blit_row(&mut sheet, &orig, tile_w, tile_h, i * tile_w, 0);
        // Copy graded into bottom row
        blit_row(&mut sheet, &graded, tile_w, tile_h, i * tile_w, tile_h);
    }

    let _ = inf_server.shutdown();

    sheet.save(&args.output)
        .with_context(|| format!("failed to save contact sheet to {}", args.output.display()))?;

    log::info!("Preview saved to {}", args.output.display());
    Ok(())
}

fn blit_row(sheet: &mut RgbImage, pixels: &[u8], width: usize, height: usize, x_off: usize, y_off: usize) {
    for row in 0..height {
        for col in 0..width {
            let src = row * width * 3 + col * 3;
            let r = pixels[src];
            let g = pixels[src + 1];
            let b = pixels[src + 2];
            sheet.put_pixel((x_off + col) as u32, (y_off + row) as u32, Rgb([r, g, b]));
        }
    }
}
