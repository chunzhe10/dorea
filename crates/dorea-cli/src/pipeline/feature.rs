//! Feature stage: fused RAUNE+Maxine+Depth inference → paged calibration store.

use std::collections::HashMap;
use anyhow::{Context, Result};

use dorea_video::inference::{InferenceServer, RauneDepthBatchItem};

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
pub fn run_feature_stage(
    cfg: &PipelineConfig,
    mut inf_server: InferenceServer,
    kf_out: KeyframeStageOutput,
) -> Result<FeatureStageOutput> {
    let KeyframeStageOutput { keyframes, proxy_w, proxy_h, .. } = kf_out;
    let proxy_size = cfg.proxy_size;

    log::info!(
        "Auto-calibrating from {} keyframes (fused RAUNE+depth)",
        keyframes.len()
    );

    let fused_items: Vec<RauneDepthBatchItem> = keyframes.iter().map(|kf| {
        RauneDepthBatchItem {
            id: format!("kf_f{}", kf.frame_index),
            pixels: kf.proxy_pixels.clone(),
            width: proxy_w,
            height: proxy_h,
            raune_max_size: proxy_w.max(proxy_h),
            depth_max_size: proxy_size.min(1036),
        }
    }).collect();

    let mut store = PagedCalibrationStore::new()
        .context("failed to create paged calibration store")?;
    let mut kf_depths: HashMap<u64, (Vec<f32>, usize, usize)> = HashMap::new();

    let n_batches = keyframes.chunks(cfg.fused_batch_size).count();
    for (batch_idx, (chunk_kfs, chunk_items)) in keyframes
        .chunks(cfg.fused_batch_size)
        .zip(fused_items.chunks(cfg.fused_batch_size))
        .enumerate()
    {
        log::info!(
            "RAUNE+depth batch {}/{n_batches} ({} frames)",
            batch_idx + 1,
            chunk_items.len(),
        );
        let results = inf_server.run_raune_depth_batch(chunk_items, cfg.maxine_in_fused_batch)
            .context(format!("Fused RAUNE+depth batch {}/{n_batches} failed", batch_idx + 1))?;

        anyhow::ensure!(
            results.len() == chunk_items.len(),
            "Fused batch {}/{n_batches} returned {} results for {} items",
            batch_idx + 1, results.len(), chunk_items.len()
        );

        for (kf, (_, enhanced, enh_w, enh_h, depth, dw, dh)) in
            chunk_kfs.iter().zip(results.into_iter())
        {
            let enhanced_proxy = if enh_w == proxy_w && enh_h == proxy_h {
                enhanced
            } else {
                dorea_video::resize::resize_rgb_bilinear(&enhanced, enh_w, enh_h, proxy_w, proxy_h)
            };
            let depth_for_store = if dw == proxy_w && dh == proxy_h {
                depth.clone()
            } else {
                InferenceServer::upscale_depth(&depth, dw, dh, proxy_w, proxy_h)
            };
            store.push(&kf.proxy_pixels, &enhanced_proxy, &depth_for_store, proxy_w, proxy_h)
                .context("failed to page fused result to store")?;
            kf_depths.insert(kf.frame_index, (depth, dw, dh));
        }
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
