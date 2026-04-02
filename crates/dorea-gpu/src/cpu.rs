// CPU implementations of the full grading pipeline.
// `depth_aware_ambiance` is ported from `run_fixed_hsl_lut_poc.py::depth_aware_ambiance()`.
// `grade_frame_cpu` composes LUT apply + HSL correct + ambiance.

use dorea_cal::Calibration;
use dorea_color::lab::{srgb_to_lab, lab_to_srgb};
use dorea_lut::apply::apply_depth_luts;
use dorea_hsl::apply::apply_hsl_corrections;
use crate::GradeParams;

/// Apply depth-aware ambiance grading in place.
///
/// `rgb`: interleaved f32 RGB \[0,1\], length = width * height * 3.
/// `depth`: f32 depth \[0,1\], length = width * height.
/// `contrast_scale`: multiplier for contrast/clarity effects (1.0 = default).
///
/// # Panics
/// Panics if slice lengths are inconsistent with width/height.
pub fn depth_aware_ambiance(
    rgb: &mut [f32],
    depth: &[f32],
    width: usize,
    height: usize,
    contrast_scale: f32,
) {
    assert_eq!(rgb.len(), width * height * 3, "rgb length mismatch");
    assert_eq!(depth.len(), width * height, "depth length mismatch");

    // Mean depth (for clarity amount)
    let mean_d: f32 = depth.iter().sum::<f32>() / depth.len() as f32;

    let n = width * height;
    for i in 0..n {
        let d = depth[i];
        let r = rgb[i * 3];
        let g = rgb[i * 3 + 1];
        let b = rgb[i * 3 + 2];

        // --- RGB → LAB ---
        let (mut l_norm, mut a_ab, mut b_ab) = {
            let (l, a, b_l) = srgb_to_lab(r, g, b);
            (l / 100.0, a, b_l)
        };

        // 1. Shadow lift
        let lift_amount = 0.2 + 0.15 * d;
        let toe = 0.15_f32;
        let shadow_mask = ((toe - l_norm) / toe).clamp(0.0, 1.0);
        l_norm += shadow_mask * lift_amount * toe;

        // 2. S-curve contrast
        let strength = (0.3 + 0.3 * d) * contrast_scale;
        let slope = 4.0 + 4.0 * strength;
        let s_curve = 1.0 / (1.0 + (-(l_norm - 0.5) * slope).exp());
        l_norm += (s_curve - l_norm) * strength;

        // 3. Highlight compress
        let compress = 0.4 + 0.2 * (1.0 - d);
        let knee_h = 0.88_f32;
        if l_norm > knee_h {
            let over = l_norm - knee_h;
            let headroom = 1.0 - knee_h;
            l_norm = knee_h + headroom * ((over / headroom * (1.0 + compress)).tanh());
        }

        // 4. Warmth (LAB a*/b* push proportional to depth and luminance)
        let lum_weight = 4.0 * l_norm * (1.0 - l_norm);
        let warmth_a = 1.0 + 5.0 * d;
        let warmth_b = 4.0 * d;
        a_ab += warmth_a * lum_weight;
        b_ab += warmth_b * lum_weight;

        // 5. Vibrance (chroma boost for desaturated pixels)
        let vibrance = 0.4 + 0.5 * d;
        let chroma = (a_ab * a_ab + b_ab * b_ab + 1e-8).sqrt();
        let chroma_norm = (chroma / 40.0).clamp(0.0, 1.0);
        let boost = vibrance * (1.0 - chroma_norm) * (l_norm / 0.25).clamp(0.0, 1.0);
        a_ab *= 1.0 + boost;
        b_ab *= 1.0 + boost;

        // Clamp LAB components
        let l_out = (l_norm * 100.0).clamp(0.0, 100.0);
        let a_out = a_ab.clamp(-128.0, 127.0);
        let b_out = b_ab.clamp(-128.0, 127.0);

        // --- LAB → RGB ---
        let (ro, go, bo) = lab_to_srgb(l_out, a_out, b_out);

        // Final highlight knee
        let knee = 0.92_f32;
        let apply_knee = |v: f32| {
            if v > knee {
                let over = v - knee;
                let room = 1.0 - knee;
                knee + room * ((over / room).tanh())
            } else {
                v
            }
        };

        rgb[i * 3]     = apply_knee(ro).clamp(0.0, 1.0);
        rgb[i * 3 + 1] = apply_knee(go).clamp(0.0, 1.0);
        rgb[i * 3 + 2] = apply_knee(bo).clamp(0.0, 1.0);
    }

    // Clarity: separable box-blur approximation of Gaussian (σ=30px at proxy).
    // We run a 3-pass box blur (approximates Gaussian well).
    let radius = (30.0_f32 * 3.0).ceil() as usize;
    let clarity_amount = (0.2 + 0.25 * mean_d) * contrast_scale;
    apply_clarity(rgb, width, height, radius, clarity_amount);
}

/// Apply clarity enhancement: extract low-frequency blur, boost detail.
fn apply_clarity(rgb: &mut [f32], width: usize, height: usize, radius: usize, clarity: f32) {
    // Work on just the L channel in LAB space for clarity
    let n = width * height;
    let mut l_channel: Vec<f32> = (0..n)
        .map(|i| {
            let r = rgb[i * 3];
            let g = rgb[i * 3 + 1];
            let b = rgb[i * 3 + 2];
            let (l, _, _) = srgb_to_lab(r, g, b);
            l / 100.0
        })
        .collect();

    // 3-pass box blur to approximate Gaussian
    let blur = three_pass_box_blur(&l_channel, width, height, radius);

    // Apply clarity: detail = tanh((L - blur) * 3) / 3
    for i in 0..n {
        let detail = ((l_channel[i] - blur[i]) * 3.0).tanh() / 3.0;
        l_channel[i] = (l_channel[i] + detail * clarity).clamp(0.0, 1.0);
    }

    // Write back L channel modifications into rgb
    for i in 0..n {
        let r = rgb[i * 3];
        let g = rgb[i * 3 + 1];
        let b = rgb[i * 3 + 2];
        let (l_old, a, b_ab) = srgb_to_lab(r, g, b);

        let l_new = l_channel[i] * 100.0;
        let l_delta = l_new - l_old;
        let l_adjusted = (l_old + l_delta).clamp(0.0, 100.0);

        let (ro, go, bo) = lab_to_srgb(l_adjusted, a, b_ab);
        rgb[i * 3]     = ro.clamp(0.0, 1.0);
        rgb[i * 3 + 1] = go.clamp(0.0, 1.0);
        rgb[i * 3 + 2] = bo.clamp(0.0, 1.0);
    }
}

/// Three-pass box blur approximating a Gaussian, operating on a 1D luminance array.
fn three_pass_box_blur(input: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    let mut buf_a = input.to_vec();
    let mut buf_b = vec![0.0_f32; input.len()];
    for _ in 0..3 {
        box_blur_rows(&buf_a, &mut buf_b, width, height, radius);
        box_blur_cols(&buf_b, &mut buf_a, width, height, radius);
    }
    buf_a
}

fn box_blur_rows(src: &[f32], dst: &mut [f32], width: usize, height: usize, radius: usize) {
    let r = radius.min(width - 1) as isize;
    for row in 0..height {
        let base = row * width;
        for col in 0..width {
            let lo = (col as isize - r).max(0) as usize;
            let hi = (col as isize + r).min(width as isize - 1) as usize;
            let mut s = 0.0_f32;
            for k in lo..=hi {
                s += src[base + k];
            }
            dst[base + col] = s / (hi - lo + 1) as f32;
        }
    }
}

fn box_blur_cols(src: &[f32], dst: &mut [f32], width: usize, height: usize, radius: usize) {
    for col in 0..width {
        for row in 0..height {
            let r = radius.min(height - 1) as isize;
            let lo = (row as isize - r).max(0) as usize;
            let hi = (row as isize + r).min(height as isize - 1) as usize;
            let mut s = 0.0_f32;
            for k in lo..=hi {
                s += src[k * width + col];
            }
            dst[row * width + col] = s / (hi - lo + 1) as f32;
        }
    }
}

/// Full CPU grading pipeline: LUT apply → HSL correct → depth_aware_ambiance → user params.
pub fn grade_frame_cpu(
    pixels: &[u8],
    depth: &[f32],
    width: usize,
    height: usize,
    calibration: &Calibration,
    params: &GradeParams,
) -> Result<Vec<u8>, String> {
    let n = width * height;

    // Convert u8 → f32
    let mut rgb_f32: Vec<f32> = pixels.iter().map(|&p| p as f32 / 255.0).collect();

    // 1. Apply depth-stratified LUT
    let lut_result = apply_depth_luts(
        &rgb_f32
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect::<Vec<_>>(),
        depth,
        &calibration.depth_luts,
    );

    // Write LUT result back into rgb_f32
    for (i, px) in lut_result.iter().enumerate() {
        rgb_f32[i * 3]     = px[0];
        rgb_f32[i * 3 + 1] = px[1];
        rgb_f32[i * 3 + 2] = px[2];
    }

    // 2. Apply HSL qualifier corrections
    let pixels_arr: Vec<[f32; 3]> = rgb_f32
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let hsl_result = apply_hsl_corrections(&pixels_arr, &calibration.hsl_corrections);
    for (i, px) in hsl_result.iter().enumerate() {
        rgb_f32[i * 3]     = px[0];
        rgb_f32[i * 3 + 1] = px[1];
        rgb_f32[i * 3 + 2] = px[2];
    }

    // 3. Apply depth_aware_ambiance
    depth_aware_ambiance(&mut rgb_f32, depth, width, height, params.contrast);

    // 4. Apply warmth (scale LAB a*/b* channels)
    if (params.warmth - 1.0).abs() > 1e-4 {
        let warmth_factor = 1.0 + (params.warmth - 1.0) * 0.3;
        for i in 0..n {
            let r = rgb_f32[i * 3];
            let g = rgb_f32[i * 3 + 1];
            let b = rgb_f32[i * 3 + 2];
            let (l, a, b_ab) = srgb_to_lab(r, g, b);
            let (ro, go, bo) = lab_to_srgb(l, a * warmth_factor, b_ab * warmth_factor);
            rgb_f32[i * 3]     = ro.clamp(0.0, 1.0);
            rgb_f32[i * 3 + 1] = go.clamp(0.0, 1.0);
            rgb_f32[i * 3 + 2] = bo.clamp(0.0, 1.0);
        }
    }

    // 5. Blend with original using strength
    if params.strength < 1.0 - 1e-4 {
        for i in 0..rgb_f32.len() {
            let orig = pixels[i] as f32 / 255.0;
            rgb_f32[i] = orig * (1.0 - params.strength) + rgb_f32[i] * params.strength;
        }
    }

    // Convert back to u8
    let out: Vec<u8> = rgb_f32.iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_aware_ambiance_deterministic() {
        let width = 4;
        let height = 4;
        let n = width * height;
        let mut rgb: Vec<f32> = (0..n * 3).map(|i| (i as f32 % 256.0) / 255.0).collect();
        let depth: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();

        let rgb_copy = rgb.clone();
        depth_aware_ambiance(&mut rgb, &depth, width, height, 1.0);

        // Should change the pixel values
        assert!(rgb != rgb_copy, "ambiance should modify pixels");

        // All values should be in [0, 1]
        for &v in &rgb {
            assert!(v >= 0.0 && v <= 1.0, "out-of-range value: {v}");
        }

        // Second call with same input should produce same output
        let mut rgb2 = rgb_copy.clone();
        depth_aware_ambiance(&mut rgb2, &depth, width, height, 1.0);
        for (a, b) in rgb.iter().zip(rgb2.iter()) {
            assert!((a - b).abs() < 1e-5, "not deterministic: {a} vs {b}");
        }
    }

    #[test]
    fn depth_aware_ambiance_zero_contrast() {
        let width = 2;
        let height = 2;
        let n = width * height;
        let rgb_orig: Vec<f32> = vec![0.5; n * 3];
        let mut rgb = rgb_orig.clone();
        let depth: Vec<f32> = vec![0.5; n];

        // contrast_scale = 0 should still run without panic
        depth_aware_ambiance(&mut rgb, &depth, width, height, 0.0);

        // Values should remain in [0, 1]
        for &v in &rgb {
            assert!((0.0..=1.0).contains(&v), "out-of-range: {v}");
        }
    }
}
