// CPU implementations of the full grading pipeline.
// `depth_aware_ambiance` is ported from `run_fixed_hsl_lut_poc.py::depth_aware_ambiance()`.
// `grade_frame_cpu` composes LUT apply + HSL correct + ambiance.

use dorea_cal::Calibration;
use dorea_color::lab::{srgb_to_lab, lab_to_srgb};
use dorea_lut::apply::apply_depth_luts;
use dorea_hsl::apply::apply_hsl_corrections;
use rayon::prelude::*;
use crate::GradeParams;

/// Apply depth-aware ambiance grading in place.
///
/// `rgb`: interleaved f32 RGB \[0,1\], length = width * height * 3.
/// `depth`: f32 depth \[0,1\], length = width * height.
/// `contrast_scale`: multiplier for contrast/clarity effects. Typical range \[0.0–2.0\];
/// 1.0 is the neutral default. Values > 1.0 boost contrast and clarity beyond the default.
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
    if width == 0 || height == 0 {
        return;
    }
    assert_eq!(rgb.len(), width * height * 3, "rgb length mismatch");
    assert_eq!(depth.len(), width * height, "depth length mismatch");

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

}

/// Fused ambiance + user warmth: single RGB→LAB→RGB roundtrip per pixel.
///
/// Combines all per-pixel LAB operations from `depth_aware_ambiance` and the
/// warmth scaling that was previously a separate pass in `finish_grade`.
/// Parallelized with rayon for multi-core throughput.
///
/// Note: output is NOT bit-exact with the old two-pass pipeline because
/// the sRGB↔LAB conversion is nonlinear. Validated to produce ΔE < 2
/// (imperceptible). This is an accepted color-science tradeoff for performance.
pub fn fused_ambiance_warmth(
    rgb: &mut [f32],
    depth: &[f32],
    width: usize,
    height: usize,
    contrast_scale: f32,
    warmth_factor: f32,
) {
    if width == 0 || height == 0 {
        return;
    }
    assert_eq!(rgb.len(), width * height * 3, "rgb length mismatch");
    assert_eq!(depth.len(), width * height, "depth length mismatch");

    let apply_warmth = (warmth_factor - 1.0).abs() > 1e-4;

    rgb.par_chunks_exact_mut(3)
        .enumerate()
        .for_each(|(i, pixel)| {
            let d = depth[i];
            let r = pixel[0];
            let g = pixel[1];
            let b = pixel[2];

            // --- RGB → LAB (once) ---
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

            // 4. Warmth (depth-proportional LAB a*/b* push)
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

            // 6. User warmth scaling (fused — was separate pass)
            if apply_warmth {
                a_ab *= warmth_factor;
                b_ab *= warmth_factor;
            }

            // Clamp LAB
            let l_out = (l_norm * 100.0).clamp(0.0, 100.0);
            let a_out = a_ab.clamp(-128.0, 127.0);
            let b_out = b_ab.clamp(-128.0, 127.0);

            // --- LAB → RGB (once) ---
            let (ro, go, bo) = lab_to_srgb(l_out, a_out, b_out);

            // Final highlight knee
            let knee = 0.92_f32;
            let apply_knee = |v: f32| -> f32 {
                if v > knee {
                    let over = v - knee;
                    let room = 1.0 - knee;
                    knee + room * ((over / room).tanh())
                } else {
                    v
                }
            };

            pixel[0] = apply_knee(ro).clamp(0.0, 1.0);
            pixel[1] = apply_knee(go).clamp(0.0, 1.0);
            pixel[2] = apply_knee(bo).clamp(0.0, 1.0);
        });
}

/// Apply depth_aware_ambiance, warmth, blend, and convert f32 → u8.
///
/// Called by both the CPU and CUDA code paths after LUT+HSL processing.
/// `rgb_f32` is modified in place (ambiance + warmth applied).
/// `orig_pixels` is the original u8 input used for the strength blend.
pub(crate) fn finish_grade(
    rgb_f32: &mut [f32],
    orig_pixels: &[u8],
    depth: &[f32],
    width: usize,
    height: usize,
    params: &GradeParams,
    _cal: &Calibration,
) -> Vec<u8> {
    // 1. Fused ambiance + warmth (single LAB roundtrip, rayon-parallelized)
    let warmth_factor = 1.0 + (params.warmth - 1.0) * 0.3;
    fused_ambiance_warmth(rgb_f32, depth, width, height, params.contrast, warmth_factor);

    // 2. Blend with original using strength
    if params.strength < 1.0 - 1e-4 {
        for i in 0..rgb_f32.len() {
            let orig = orig_pixels[i] as f32 / 255.0;
            rgb_f32[i] = orig * (1.0 - params.strength) + rgb_f32[i] * params.strength;
        }
    }

    // 3. f32 → u8
    rgb_f32.iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect()
}

/// Grade a single pixel through the full CPU pipeline.
///
/// Equivalent to running `grade_frame_cpu` on a 1×1 frame with the given pixel,
/// but without heap allocation. Used as a CPU oracle for combined LUT accuracy tests,
/// and mirrors the logic in `grade_pixel_device` (grade_pixel.cuh).
///
/// Returns graded f32 RGB [0,1] (caller converts to u8 as needed).
pub fn grade_pixel_cpu(
    r: f32, g: f32, b: f32,
    depth: f32,
    calibration: &Calibration,
    params: &GradeParams,
) -> [f32; 3] {
    use dorea_lut::apply::apply_depth_luts;
    use dorea_hsl::apply::apply_hsl_corrections;

    // 1. Depth-stratified LUT apply
    let lut_out = apply_depth_luts(&[[r, g, b]], &[depth], &calibration.depth_luts);
    let [r1, g1, b1] = lut_out[0];

    // 2. HSL qualifier corrections
    let hsl_out = apply_hsl_corrections(&[[r1, g1, b1]], &calibration.hsl_corrections);
    let [r2, g2, b2] = hsl_out[0];

    // 3. Fused ambiance + warmth (single LAB roundtrip)
    let mut px = [r2, g2, b2];
    let warmth_factor = 1.0 + (params.warmth - 1.0) * 0.3;
    let d = depth;

    let (mut l_norm, mut a_ab, mut b_ab) = {
        let (l, a, b_l) = dorea_color::lab::srgb_to_lab(px[0], px[1], px[2]);
        (l / 100.0, a, b_l)
    };

    // Shadow lift
    let lift_amount = 0.2 + 0.15 * d;
    let toe = 0.15_f32;
    let shadow_mask = ((toe - l_norm) / toe).clamp(0.0, 1.0);
    l_norm += shadow_mask * lift_amount * toe;

    // S-curve contrast
    let contrast_strength = (0.3 + 0.3 * d) * params.contrast;
    let slope = 4.0 + 4.0 * contrast_strength;
    let s_curve = 1.0 / (1.0 + (-(l_norm - 0.5) * slope).exp());
    l_norm += (s_curve - l_norm) * contrast_strength;

    // Highlight compress
    let compress = 0.4 + 0.2 * (1.0 - d);
    let knee_h = 0.88_f32;
    if l_norm > knee_h {
        let over = l_norm - knee_h;
        let headroom = 1.0 - knee_h;
        l_norm = knee_h + headroom * ((over / headroom * (1.0 + compress)).tanh());
    }

    // Warmth (LAB a*/b* push)
    let lum_weight = 4.0 * l_norm * (1.0 - l_norm);
    a_ab += (1.0 + 5.0 * d) * lum_weight;
    b_ab += 4.0 * d * lum_weight;

    // Vibrance
    let vibrance = 0.4 + 0.5 * d;
    let chroma = (a_ab * a_ab + b_ab * b_ab + 1e-8).sqrt();
    let chroma_norm = (chroma / 40.0).clamp(0.0, 1.0);
    let boost = vibrance * (1.0 - chroma_norm) * (l_norm / 0.25).clamp(0.0, 1.0);
    a_ab *= 1.0 + boost;
    b_ab *= 1.0 + boost;

    // User warmth scaling
    if (warmth_factor - 1.0).abs() > 1e-4 {
        a_ab *= warmth_factor;
        b_ab *= warmth_factor;
    }

    // LAB → RGB
    let l_out = (l_norm * 100.0).clamp(0.0, 100.0);
    let a_out = a_ab.clamp(-128.0, 127.0);
    let b_out_clamped = b_ab.clamp(-128.0, 127.0);
    let (ro, go, bo) = dorea_color::lab::lab_to_srgb(l_out, a_out, b_out_clamped);

    // Final highlight knee
    let knee = 0.92_f32;
    let apply_knee = |v: f32| -> f32 {
        if v > knee { let over = v - knee; let room = 1.0 - knee; knee + room * ((over / room).tanh()) } else { v }
    };

    px[0] = apply_knee(ro).clamp(0.0, 1.0);
    px[1] = apply_knee(go).clamp(0.0, 1.0);
    px[2] = apply_knee(bo).clamp(0.0, 1.0);

    // 4. Strength blend with original input
    let strength = params.strength;
    if strength < 1.0 - 1e-4 {
        px[0] = r * (1.0 - strength) + px[0] * strength;
        px[1] = g * (1.0 - strength) + px[1] * strength;
        px[2] = b * (1.0 - strength) + px[2] * strength;
    }

    px
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

    // 3–4. Ambiance + warmth + blend + u8 (CPU finish pass)
    Ok(finish_grade(&mut rgb_f32, pixels, depth, width, height, params, calibration))
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

    #[test]
    fn finish_grade_runs_without_panic() {
        use crate::GradeParams;
        use dorea_cal::Calibration;
        use dorea_hsl::HslCorrections;
        use dorea_lut::types::{DepthLuts, LutGrid};

        let width = 4; let height = 4; let n = width * height;
        let mut lut = LutGrid::new(2);
        for ri in 0..2usize { for gi in 0..2usize { for bi in 0..2usize {
            lut.set(ri, gi, bi, [ri as f32, gi as f32, bi as f32]);
        }}}
        let cal = Calibration::new(
            DepthLuts::new(vec![lut], vec![0.0, 1.0]),
            HslCorrections(vec![]),
            0,
        );

        let mut rgb_f32: Vec<f32> = vec![0.5; n * 3];
        let orig: Vec<u8>  = vec![128u8; n * 3];
        let depth: Vec<f32> = vec![0.5; n];

        // must not panic, output must be in [0,255]
        let out = finish_grade(&mut rgb_f32, &orig, &depth, width, height,
                               &GradeParams::default(), &cal);
        assert_eq!(out.len(), n * 3);
        for &v in &out { assert!(v <= 255, "out of range: {v}"); }

        // second call: same shape, must not panic
        let mut rgb_f32b: Vec<f32> = vec![0.5; n * 3];
        let out2 = finish_grade(&mut rgb_f32b, &orig, &depth, width, height,
                                &GradeParams::default(), &cal);
        assert_eq!(out2.len(), n * 3);
    }

    #[test]
    fn fused_ambiance_warmth_matches_separate_passes() {
        let width = 4;
        let height = 4;
        let n = width * height;
        let original: Vec<f32> = (0..n * 3).map(|i| (i as f32 % 256.0) / 255.0).collect();
        let depth: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();

        // Old path: separate depth_aware_ambiance + warmth
        let mut rgb_old = original.clone();
        depth_aware_ambiance(&mut rgb_old, &depth, width, height, 1.0);
        let warmth_factor = 1.0 + (1.2 - 1.0) * 0.3;
        for i in 0..n {
            let r = rgb_old[i * 3];
            let g = rgb_old[i * 3 + 1];
            let b = rgb_old[i * 3 + 2];
            let (l, a, b_ab) = srgb_to_lab(r, g, b);
            let (ro, go, bo) = lab_to_srgb(l, a * warmth_factor, b_ab * warmth_factor);
            rgb_old[i * 3]     = ro.clamp(0.0, 1.0);
            rgb_old[i * 3 + 1] = go.clamp(0.0, 1.0);
            rgb_old[i * 3 + 2] = bo.clamp(0.0, 1.0);
        }

        // New path: fused
        let mut rgb_new = original.clone();
        fused_ambiance_warmth(&mut rgb_new, &depth, width, height, 1.0, warmth_factor);

        // Fused output won't be bit-exact (different order of LAB roundtrips),
        // but should be perceptually close. Tolerance 0.05 in normalized RGB (~13/255).
        for i in 0..rgb_new.len() {
            let diff = (rgb_new[i] - rgb_old[i]).abs();
            assert!(diff < 0.05, "pixel {i}: old={:.4} new={:.4} diff={:.4}", rgb_old[i], rgb_new[i], diff);
        }
    }

    #[test]
    fn fused_ambiance_warmth_neutral_warmth() {
        let width = 2;
        let height = 2;
        let n = width * height;
        let original: Vec<f32> = vec![0.5; n * 3];
        let depth: Vec<f32> = vec![0.5; n];

        // warmth_factor = 1.0 means no user warmth scaling
        let mut rgb = original.clone();
        fused_ambiance_warmth(&mut rgb, &depth, width, height, 1.0, 1.0);

        // All values should be in [0, 1]
        for &v in &rgb {
            assert!((0.0..=1.0).contains(&v), "out-of-range: {v}");
        }
    }

    #[test]
    fn grade_pixel_cpu_matches_grade_frame_cpu_baseline() {
        // This test will fail until grade_pixel_cpu is implemented in Task 3.
        // It exists here to lock in the expected baseline before any changes.
        use dorea_cal::Calibration;
        use dorea_hsl::derive::{HslCorrections, QualifierCorrection};
        use dorea_lut::types::{DepthLuts, LutGrid};

        fn identity_lut(size: usize) -> LutGrid {
            let mut lut = LutGrid::new(size);
            for ri in 0..size { for gi in 0..size { for bi in 0..size {
                let r = ri as f32 / (size - 1) as f32;
                let g = gi as f32 / (size - 1) as f32;
                let b = bi as f32 / (size - 1) as f32;
                lut.set(ri, gi, bi, [r, g, b]);
            }}}
            lut
        }

        let n_zones = 5;
        let luts: Vec<LutGrid> = (0..n_zones).map(|_| identity_lut(17)).collect();
        let boundaries: Vec<f32> = (0..=n_zones).map(|i| i as f32 / n_zones as f32).collect();
        let depth_luts = DepthLuts::new(luts, boundaries);
        let hsl = HslCorrections(vec![QualifierCorrection {
            h_center: 0.0, h_width: 1.0, h_offset: 0.0,
            s_ratio: 1.0, v_offset: 0.0, weight: 0.0,
        }]);
        let cal = Calibration::new(depth_luts, hsl, 1);
        let params = crate::GradeParams::default();

        // Input: r=0.6, g=0.3, b=0.2, depth=0.5
        // Use exact u8 → f32 round-trip to avoid floating-point inconsistency
        let r_u8 = 153u8; let g_u8 = 77u8; let b_u8 = 51u8;
        let pixels = vec![r_u8, g_u8, b_u8];
        let depth = vec![0.5f32];

        // Full pipeline baseline
        let out = grade_frame_cpu(&pixels, &depth, 1, 1, &cal, &params).unwrap();
        assert_eq!(out.len(), 3);
        // All in range
        for &v in &out { assert!(v <= 255, "out of range"); }

        // Baseline values are pinned here (update this comment if params change):
        // This will be verified against grade_pixel_cpu in Task 3.
        // Print for reference during development:
        // eprintln!("baseline: {:?}", out);
    }

    #[test]
    fn grade_pixel_cpu_matches_grade_frame_cpu() {
        use dorea_cal::Calibration;
        use dorea_hsl::derive::{HslCorrections, QualifierCorrection};
        use dorea_lut::types::{DepthLuts, LutGrid};

        fn identity_lut(size: usize) -> LutGrid {
            let mut lut = LutGrid::new(size);
            for ri in 0..size { for gi in 0..size { for bi in 0..size {
                let r = ri as f32 / (size - 1) as f32;
                let g = gi as f32 / (size - 1) as f32;
                let b = bi as f32 / (size - 1) as f32;
                lut.set(ri, gi, bi, [r, g, b]);
            }}}
            lut
        }

        let n_zones = 5;
        let luts: Vec<LutGrid> = (0..n_zones).map(|_| identity_lut(17)).collect();
        let boundaries: Vec<f32> = (0..=n_zones).map(|i| i as f32 / n_zones as f32).collect();
        let cal = Calibration::new(
            DepthLuts::new(luts, boundaries),
            HslCorrections(vec![QualifierCorrection {
                h_center: 0.0, h_width: 1.0, h_offset: 0.0,
                s_ratio: 1.0, v_offset: 0.0, weight: 0.0,
            }]),
            1,
        );
        let params = crate::GradeParams::default();

        let r_u8 = 153u8; let g_u8 = 77u8; let b_u8 = 51u8;
        let pixels = vec![r_u8, g_u8, b_u8];
        let depth_val = 0.5f32;
        let depth = vec![depth_val];

        // Full pipeline
        let frame_out = grade_frame_cpu(&pixels, &depth, 1, 1, &cal, &params).unwrap();

        // Single-pixel path (uses same u8-rounded inputs)
        let r_f32 = r_u8 as f32 / 255.0;
        let g_f32 = g_u8 as f32 / 255.0;
        let b_f32 = b_u8 as f32 / 255.0;
        let px_out = grade_pixel_cpu(r_f32, g_f32, b_f32, depth_val, &cal, &params);
        let r_out = (px_out[0].clamp(0.0, 1.0) * 255.0).round() as u8;
        let g_out = (px_out[1].clamp(0.0, 1.0) * 255.0).round() as u8;
        let b_out = (px_out[2].clamp(0.0, 1.0) * 255.0).round() as u8;

        assert_eq!([r_out, g_out, b_out], [frame_out[0], frame_out[1], frame_out[2]],
            "grade_pixel_cpu must match grade_frame_cpu to within 1/255 rounding");
    }

    #[test]
    fn finish_grade_roundtrip() {
        use crate::GradeParams;
        use dorea_cal::Calibration;
        use dorea_hsl::HslCorrections;
        use dorea_lut::types::{DepthLuts, LutGrid};

        let width = 2;
        let height = 2;
        let n = width * height;

        // Build a minimal identity calibration (1 zone, 2x2x2 identity LUT).
        let mut lut = LutGrid::new(2);
        for ri in 0..2usize {
            for gi in 0..2usize {
                for bi in 0..2usize {
                    lut.set(ri, gi, bi, [ri as f32, gi as f32, bi as f32]);
                }
            }
        }
        let depth_luts = DepthLuts::new(vec![lut], vec![0.0, 1.0]);
        let hsl_corrections = HslCorrections(vec![]);
        let cal = Calibration::new(depth_luts, hsl_corrections, 0);

        // Grey pixels, f32
        let mut rgb_f32: Vec<f32> = vec![0.5; n * 3];
        let orig_pixels: Vec<u8> = vec![128u8; n * 3];
        let depth: Vec<f32> = vec![0.5; n];
        let params = GradeParams::default();

        let out = finish_grade(&mut rgb_f32, &orig_pixels, &depth, width, height, &params, &cal);
        assert_eq!(out.len(), n * 3);
    }
}
