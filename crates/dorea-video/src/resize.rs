// RGB frame resize utilities used by the grading pipeline.

/// Compute `(proxy_w, proxy_h)` scaled so the long edge ≤ `max_size`.
/// Returns the original dimensions unchanged if they are already within bounds.
pub fn proxy_dims(src_w: usize, src_h: usize, max_size: usize) -> (usize, usize) {
    let long_edge = src_w.max(src_h);
    if long_edge <= max_size {
        return (src_w, src_h);
    }
    let scale = max_size as f64 / long_edge as f64;
    let pw = ((src_w as f64 * scale).round() as usize).max(1);
    let ph = ((src_h as f64 * scale).round() as usize).max(1);
    (pw, ph)
}

/// Bilinearly downsample an RGB24 frame.
///
/// `src` is interleaved RGB u8, length = `src_w * src_h * 3`.
/// Returns interleaved RGB u8, length = `dst_w * dst_h * 3`.
pub fn resize_rgb_bilinear(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8> {
    assert_eq!(src.len(), src_w * src_h * 3, "src length mismatch");
    let mut out = vec![0u8; dst_w * dst_h * 3];
    let sw = (src_w as f32 - 1.0).max(0.0);
    let sh = (src_h as f32 - 1.0).max(0.0);
    let dw = (dst_w as f32 - 1.0).max(1.0);
    let dh = (dst_h as f32 - 1.0).max(1.0);
    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let sx = dx as f32 * sw / dw;
            let sy = dy as f32 * sh / dh;
            let x0 = sx.floor() as usize;
            let y0 = sy.floor() as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let y1 = (y0 + 1).min(src_h - 1);
            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;
            let out_base = (dy * dst_w + dx) * 3;
            for c in 0..3 {
                let v00 = src[(y0 * src_w + x0) * 3 + c] as f32;
                let v10 = src[(y0 * src_w + x1) * 3 + c] as f32;
                let v01 = src[(y1 * src_w + x0) * 3 + c] as f32;
                let v11 = src[(y1 * src_w + x1) * 3 + c] as f32;
                let v = v00 * (1.0 - fx) * (1.0 - fy)
                      + v10 * fx * (1.0 - fy)
                      + v01 * (1.0 - fx) * fy
                      + v11 * fx * fy;
                out[out_base + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proxy_dims_no_change_when_within_bounds() {
        assert_eq!(proxy_dims(518, 292, 518), (518, 292));
        assert_eq!(proxy_dims(100, 50, 1000), (100, 50));
    }

    #[test]
    fn proxy_dims_scales_down_landscape() {
        let (pw, ph) = proxy_dims(3840, 2160, 518);
        assert!(pw.max(ph) <= 518, "long edge {pw}×{ph} > 518");
        let ratio_orig = 3840.0f64 / 2160.0;
        let ratio_proxy = pw as f64 / ph as f64;
        assert!((ratio_proxy - ratio_orig).abs() < 0.1, "aspect ratio {ratio_proxy} vs {ratio_orig}");
    }

    #[test]
    fn proxy_dims_scales_down_portrait() {
        let (pw, ph) = proxy_dims(1080, 1920, 518);
        assert!(pw.max(ph) <= 518);
    }

    #[test]
    fn resize_rgb_bilinear_dimensions() {
        let src = vec![128u8; 4 * 4 * 3];
        let dst = resize_rgb_bilinear(&src, 4, 4, 2, 2);
        assert_eq!(dst.len(), 2 * 2 * 3);
    }

    #[test]
    fn resize_rgb_bilinear_solid_color_preserved() {
        let src = vec![200u8, 100u8, 50u8].repeat(16);
        let dst = resize_rgb_bilinear(&src, 4, 4, 2, 2);
        for chunk in dst.chunks_exact(3) {
            assert_eq!(chunk[0], 200, "R channel changed");
            assert_eq!(chunk[1], 100, "G channel changed");
            assert_eq!(chunk[2], 50,  "B channel changed");
        }
    }

    #[test]
    fn resize_rgb_bilinear_identity_same_size() {
        let src: Vec<u8> = (0u8..48).collect();
        let dst = resize_rgb_bilinear(&src, 4, 4, 4, 4);
        assert_eq!(dst, src, "same-size resize should be identity");
    }
}
