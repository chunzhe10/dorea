// dorea-video — Video I/O (ffmpeg subprocess) + Python inference subprocess management.
//
// Public API:
// - `ffmpeg::probe` — probe video metadata
// - `ffmpeg::decode_frames` — iterate decoded frames
// - `ffmpeg::encode_frames` — encode frame stream to output file
// - `ffmpeg::extract_frame_at` — extract single frame at timestamp
// - `scene::histogram_distance` — compare two frames for scene change
// - `inference::InferenceServer` — manage Python inference subprocess

pub mod ffmpeg;
pub mod inference;
pub mod resize;
pub mod scene;
