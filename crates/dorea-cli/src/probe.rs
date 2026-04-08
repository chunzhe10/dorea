//! `dorea probe` — detect container, codec, bit depth, and suggest flags.

use std::path::PathBuf;
use clap::Args;
use anyhow::{Context, Result};
use dorea_video::ffmpeg::{self, InputEncoding};

#[derive(Args, Debug)]
pub struct ProbeArgs {
    /// Input video file
    #[arg(long)]
    pub input: PathBuf,
}

pub fn run(args: ProbeArgs) -> Result<()> {
    let info = ffmpeg::probe(&args.input)
        .context("ffprobe failed — is ffmpeg installed?")?;

    let encoding = InputEncoding::auto_detect(&info, &args.input);

    println!("File:       {}", args.input.display());
    println!("Resolution: {}x{}", info.width, info.height);
    println!("FPS:        {:.3}", info.fps);
    println!("Duration:   {:.1}s ({} frames)", info.duration_secs, info.frame_count);
    println!("Codec:      {}", info.codec_name);
    println!("Pixel fmt:  {}", info.pix_fmt);
    println!("Bit depth:  {}-bit", info.bits_per_component);
    println!("Audio:      {}", if info.has_audio { "yes" } else { "no" });
    println!();
    println!("Detected encoding: {encoding}");

    if encoding.is_10bit() {
        println!();
        println!("Suggested command:");
        println!("  dorea grade --input {} --input-encoding {} --output-codec prores",
            args.input.display(), encoding);
    } else {
        println!();
        println!("Suggested command:");
        println!("  dorea grade --input {}", args.input.display());
    }

    Ok(())
}
