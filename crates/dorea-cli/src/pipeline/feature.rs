//! Feature stage: fused RAUNE+Maxine+Depth inference → paged calibration store.

use std::collections::HashMap;
use anyhow::{Context, Result};

use dorea_video::inference::InferenceServer;

use super::{PipelineConfig, KeyframeStageOutput, FeatureStageOutput};

/// Temporary on-disk store for calibration frames.
///
/// Two packed binary files, both memory-mapped after the inference phase:
///   depths.bin  — all depth maps concatenated (f32 LE), one region per frame
///   pixtar.bin  — all [pixels | target] pairs concatenated (u8 RGB), one pair per frame
pub struct PagedCalibrationStore {
    dir: std::path::PathBuf,
    widths: Vec<usize>,
    heights: Vec<usize>,
    depth_offsets: Vec<usize>,
    pixtar_offsets: Vec<usize>,
    depth_writer: Option<std::io::BufWriter<std::fs::File>>,
    pixtar_writer: Option<std::io::BufWriter<std::fs::File>>,
    depth_mmap: Option<memmap2::Mmap>,
    pixtar_mmap: Option<memmap2::Mmap>,
}

impl PagedCalibrationStore {
    pub fn new() -> Result<Self> {
        use std::io::BufWriter;
        let dir = std::env::temp_dir()
            .join(format!("dorea_cal_{}", std::process::id()));
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("failed to create calibration temp dir {:?}", dir))?;
        let dw = BufWriter::new(
            std::fs::File::create(dir.join("depths.bin")).context("create depths.bin")?
        );
        let pw = BufWriter::new(
            std::fs::File::create(dir.join("pixtar.bin")).context("create pixtar.bin")?
        );
        Ok(Self {
            dir,
            widths: Vec::new(),
            heights: Vec::new(),
            depth_offsets: Vec::new(),
            pixtar_offsets: Vec::new(),
            depth_writer: Some(dw),
            pixtar_writer: Some(pw),
            depth_mmap: None,
            pixtar_mmap: None,
        })
    }

    pub fn push(&mut self, pixels: &[u8], target: &[u8], depth: &[f32], width: usize, height: usize) -> Result<()> {
        use std::io::Write;
        let dw = self.depth_writer.as_mut().context("store already sealed")?;
        let pw = self.pixtar_writer.as_mut().context("store already sealed")?;

        let depth_off = self.depth_offsets.last().copied().unwrap_or(0)
            + self.widths.last().copied().unwrap_or(0)
            * self.heights.last().copied().unwrap_or(0) * 4;
        let pixtar_off = self.pixtar_offsets.last().copied().unwrap_or(0)
            + self.widths.last().copied().unwrap_or(0)
            * self.heights.last().copied().unwrap_or(0) * 6;

        for f in depth {
            dw.write_all(&f.to_le_bytes()).context("write depth")?;
        }
        pw.write_all(pixels).context("write pixels")?;
        pw.write_all(target).context("write target")?;

        self.depth_offsets.push(depth_off);
        self.pixtar_offsets.push(pixtar_off);
        self.widths.push(width);
        self.heights.push(height);
        Ok(())
    }

    pub fn seal(&mut self) -> Result<()> {
        use std::io::Write;
        if let Some(mut dw) = self.depth_writer.take() { dw.flush().context("flush depths")?; }
        if let Some(mut pw) = self.pixtar_writer.take() { pw.flush().context("flush pixtar")?; }
        let df = std::fs::File::open(self.dir.join("depths.bin")).context("open depths.bin")?;
        let pf = std::fs::File::open(self.dir.join("pixtar.bin")).context("open pixtar.bin")?;
        self.depth_mmap = Some(unsafe { memmap2::Mmap::map(&df) }.context("mmap depths")?);
        self.pixtar_mmap = Some(unsafe { memmap2::Mmap::map(&pf) }.context("mmap pixtar")?);
        Ok(())
    }

    pub fn len(&self) -> usize { self.widths.len() }

    pub fn depth_bytes(&self, i: usize) -> &[u8] {
        let mmap = self.depth_mmap.as_ref().expect("call seal() before reading");
        let off = self.depth_offsets[i];
        let len = self.widths[i] * self.heights[i] * 4;
        &mmap[off..off + len]
    }

    pub fn read_depth(&self, i: usize) -> Vec<f32> {
        self.depth_bytes(i).chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }

    pub fn pixtar_slices(&self, i: usize) -> (&[u8], &[u8]) {
        let mmap = self.pixtar_mmap.as_ref().expect("call seal() before reading");
        let off = self.pixtar_offsets[i];
        let n = self.widths[i] * self.heights[i] * 3;
        (&mmap[off..off + n], &mmap[off + n..off + n * 2])
    }

    pub fn dims(&self, i: usize) -> (usize, usize) { (self.widths[i], self.heights[i]) }
}

impl Drop for PagedCalibrationStore {
    fn drop(&mut self) {
        drop(self.depth_mmap.take());
        drop(self.pixtar_mmap.take());
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Run the feature extraction stage: fused RAUNE+depth inference, paged store.
///
/// RAUNE runs on proxy-resolution frames (for LUT training).
/// Depth runs on original frames (higher resolution → better depth maps).
pub fn run_feature_stage(
    cfg: &PipelineConfig,
    info: &dorea_video::ffmpeg::VideoInfo,
    mut inf_server: InferenceServer,
    kf_out: KeyframeStageOutput,
) -> Result<FeatureStageOutput> {
    let KeyframeStageOutput { keyframes, proxy_w, proxy_h, .. } = kf_out;

    log::info!(
        "Auto-calibrating from {} keyframes (parallel RAUNE+depth)",
        keyframes.len()
    );

    // Depth runs on full-res keyframes for better depth quality.
    // The depth model internally resizes to max_size, but starting from 4K source
    // preserves more detail than starting from proxy (518px).
    // After RAUNE completes and is unloaded, depth can use the freed VRAM for 1036px.
    let depth_max_size = 1518; // max depth res — RAUNE unloaded first to free VRAM
    log::info!(
        "Extracting {} keyframes at full resolution for depth (max_size={})",
        keyframes.len(), depth_max_size,
    );
    let mut fullres_pixels: Vec<Vec<u8>> = Vec::with_capacity(keyframes.len());
    for kf in &keyframes {
        let ts = kf.frame_index as f64 / info.fps.max(1.0);
        let pixels = dorea_video::ffmpeg::extract_frame_at(&cfg.input, ts, info.width, info.height)
            .with_context(|| format!("failed to extract full-res keyframe at frame {}", kf.frame_index))?;
        fullres_pixels.push(pixels);
    }

    let mut store = PagedCalibrationStore::new()
        .context("failed to create paged calibration store")?;
    let mut kf_depths: HashMap<u64, (Vec<f32>, usize, usize)> = HashMap::new();

    // --- RAUNE on proxy pixels (for LUT training) ---
    log::info!("Running RAUNE on {} keyframes (proxy {}x{})", keyframes.len(), proxy_w, proxy_h);
    let mut raune_enhanced: Vec<(Vec<u8>, usize, usize)> = Vec::with_capacity(keyframes.len());
    for (i, kf) in keyframes.iter().enumerate() {
        let (enhanced, ew, eh) = inf_server.run_raune(
            &format!("kf_f{}", kf.frame_index),
            &kf.proxy_pixels, proxy_w, proxy_h,
            proxy_w.max(proxy_h),
        ).with_context(|| format!("RAUNE failed for keyframe {i}"))?;
        raune_enhanced.push((enhanced, ew, eh));
    }
    log::info!("RAUNE complete");

    // --- Depth on full-res pixels (better depth quality) ---
    // Unload RAUNE first to free ~500MB VRAM for higher-res depth inference.
    inf_server.unload_raune().context("failed to unload RAUNE")?;
    log::info!("RAUNE unloaded — VRAM freed for depth at {}px", depth_max_size);
    log::info!("Running depth on {} keyframes (full-res, max_size={})", keyframes.len(), depth_max_size);
    use dorea_video::inference::DepthBatchItem;
    let mut depth_results: Vec<(String, Vec<f32>, usize, usize)> = Vec::with_capacity(keyframes.len());
    // Sub-batch depth to avoid OOM: 4 full-res frames at a time
    let depth_sub_batch = 2; // small batches at high res to fit 6GB VRAM
    for chunk_start in (0..keyframes.len()).step_by(depth_sub_batch) {
        let chunk_end = (chunk_start + depth_sub_batch).min(keyframes.len());
        let depth_items: Vec<DepthBatchItem> = (chunk_start..chunk_end).map(|i| {
            DepthBatchItem {
                id: format!("kf_f{}", keyframes[i].frame_index),
                pixels: fullres_pixels[i].clone(),
                width: info.width,
                height: info.height,
                max_size: depth_max_size,
            }
        }).collect();
        let batch_results = inf_server.run_depth_batch(&depth_items)
            .with_context(|| format!("Depth sub-batch {}/{} failed", chunk_start / depth_sub_batch + 1,
                (keyframes.len() + depth_sub_batch - 1) / depth_sub_batch))?;
        depth_results.extend(batch_results);
    }
    log::info!("Depth complete");

    // --- Combine RAUNE + depth results ---
    for (i, kf) in keyframes.iter().enumerate() {
        let (enhanced, enh_w, enh_h) = &raune_enhanced[i];
        let enhanced_proxy = if *enh_w == proxy_w && *enh_h == proxy_h {
            enhanced.clone()
        } else {
            dorea_video::resize::resize_rgb_bilinear(enhanced, *enh_w, *enh_h, proxy_w, proxy_h)
        };

        let (_, ref depth, dw, dh) = depth_results[i];
        let depth_for_store = if dw == proxy_w && dh == proxy_h {
            depth.clone()
        } else {
            InferenceServer::upscale_depth(depth, dw, dh, proxy_w, proxy_h)
        };
        store.push(&kf.proxy_pixels, &enhanced_proxy, &depth_for_store, proxy_w, proxy_h)
            .context("failed to page result to store")?;
        kf_depths.insert(kf.frame_index, (depth.clone(), dw, dh));
    }
    log::info!("Fused inference complete ({} keyframes)", keyframes.len());
    debug_assert_eq!(store.len(), keyframes.len(), "store/keyframes length diverged");
    let _ = inf_server.shutdown();

    store.seal().context("failed to seal calibration store")?;

    Ok(FeatureStageOutput {
        store,
        keyframe_depths: kf_depths,
        keyframes,
    })
}
