//! HSL qualifier definitions (6-vector, DaVinci/Adobe style).

/// A hue-range qualifier for HSL secondary correction.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HslQualifier {
    pub name: &'static str,
    /// Hue center in degrees [0, 360)
    pub h_center: f32,
    /// Hue half-width in degrees
    pub h_width: f32,
}

/// The 6 HSL qualifiers matching DaVinci's 6-vector secondary color corrector.
pub const HSL_QUALIFIERS: &[HslQualifier] = &[
    HslQualifier { name: "Red/Skin", h_center:   0.0, h_width: 40.0 },
    HslQualifier { name: "Yellow",   h_center:  40.0, h_width: 40.0 },
    HslQualifier { name: "Green",    h_center: 100.0, h_width: 50.0 },
    HslQualifier { name: "Cyan",     h_center: 170.0, h_width: 40.0 },
    HslQualifier { name: "Blue",     h_center: 210.0, h_width: 40.0 },
    HslQualifier { name: "Magenta",  h_center: 290.0, h_width: 50.0 },
];

/// Minimum saturation for a pixel to participate in qualifier matching.
pub const MIN_SATURATION: f32 = 0.08;
/// Minimum total weight for a qualifier to be considered active.
pub const MIN_WEIGHT: f32 = 100.0;
