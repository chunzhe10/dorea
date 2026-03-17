"""Phase 2: Claude Scene Analysis — Subject Detection (~2-5 min per clip, API cost)

Sends keyframe batches to Claude API for temporal scene analysis. Claude identifies
subjects (divers, fish species, coral), their bounding boxes, and first-appearance
frames. Output feeds SAM2 (Phase 3) with bounding boxes for mask initialization.

Usage:
    python 02_claude_scene_analysis.py --date 2026-03-17
    python 02_claude_scene_analysis.py --date 2026-03-17 --dry-run

Inputs:
    - working/keyframes/{clip_id}/ — extracted keyframes from Phase 1

Outputs:
    - working/scene_analysis/{clip_id}.json — per-clip scene analysis
      Schema: {"clip_id": str, "subjects": [{"label": str, "first_appearance_frame": int,
               "bbox_normalised": [x_min, y_min, x_max, y_max], "confidence": str}]}

Dependencies:
    - anthropic SDK
    - No GPU required (API call)
    - Estimated cost: <$2 per dive session

Architecture doc: Section 4.2
"""

import argparse
import base64
import json
import logging
import os
import re
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# System prompt for Claude scene analysis
SYSTEM_PROMPT = """\
You are analysing underwater dive footage. You will receive a sequence of keyframes \
extracted at 2-second intervals from a single video clip. Each frame is labelled with \
its frame number.

Your task is to identify distinct subjects visible in these frames:
- Marine life (identify to species level when possible, e.g. "napoleon_wrasse", \
"lionfish", "green_sea_turtle", "manta_ray", "clownfish", "grouper")
- Divers (label as "diver")
- Notable coral formations or terrain features (e.g. "coral_bommie", "sea_fan", \
"barrel_sponge")

For each NEW subject appearing in these frames, provide:
1. **label**: A snake_case identifier for the subject (species name or description)
2. **bbox_normalised**: Bounding box as [x_min, y_min, x_max, y_max] in normalised \
0.0-1.0 coordinates relative to frame dimensions
3. **first_appearance_frame**: The frame number (from the filename) where this subject \
FIRST appears
4. **confidence**: One of "high", "medium", or "low"

Important rules:
- Only report NEW subject appearances. Do not re-report subjects that were already \
identified in previous batches (provided below as "previously identified subjects").
- If the same species appears as a clearly different individual, give it a distinct \
label (e.g. "napoleon_wrasse_2").
- Bounding boxes should tightly enclose the subject in the frame where it first appears.
- Be conservative: only report subjects you can clearly identify. Use "low" confidence \
for uncertain identifications.

Return ONLY a JSON array of subject objects. No markdown, no explanation, just the \
JSON array. Example:
[
  {
    "label": "napoleon_wrasse",
    "first_appearance_frame": 14,
    "bbox_normalised": [0.3, 0.2, 0.7, 0.8],
    "confidence": "high"
  }
]

If no new subjects appear in these frames, return an empty array: []
"""


def find_workspace_root() -> Path:
    """Resolve the workspace root directory.

    Uses $CORVIA_WORKSPACE if set, otherwise walks up from the script location
    to find the directory containing repos/dorea.
    """
    env_root = os.environ.get("CORVIA_WORKSPACE")
    if env_root:
        return Path(env_root)

    # Walk up from script location: scripts/ -> dorea/ -> repos/ -> workspace root
    candidate = Path(__file__).resolve().parent.parent.parent.parent
    if (candidate / "repos" / "dorea").is_dir():
        return candidate

    # Fallback: hardcoded devcontainer path
    return Path("/workspaces/dorea-workspace")


def load_config(workspace_root: Path) -> dict:
    """Load pipeline config from repos/dorea/config.yaml."""
    config_path = workspace_root / "repos" / "dorea" / "config.yaml"
    if not config_path.is_file():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_api_key(config: dict) -> str:
    """Resolve the Anthropic API key from config or environment.

    The config value may be a literal key or '${ANTHROPIC_API_KEY}' placeholder.
    Falls back to the ANTHROPIC_API_KEY environment variable.
    """
    key = config.get("anthropic_api_key", "")
    if key and not key.startswith("${"):
        return key
    return os.environ.get("ANTHROPIC_API_KEY", "")


def discover_keyframe_dirs(keyframes_root: Path) -> list[Path]:
    """Find all clip keyframe directories under the keyframes root.

    Returns sorted list of directories containing .jpg files.
    """
    if not keyframes_root.is_dir():
        return []

    dirs = []
    for entry in sorted(keyframes_root.iterdir()):
        if entry.is_dir():
            # Check that it contains at least one .jpg file
            jpg_files = [f for f in entry.iterdir() if f.suffix == ".jpg"]
            if jpg_files:
                dirs.append(entry)
    return dirs


def get_sorted_frames(clip_dir: Path) -> list[tuple[int, Path]]:
    """Get sorted list of (frame_number, path) for all keyframes in a clip directory.

    Expects filenames like frame_000001.jpg, frame_000014.jpg, etc.
    Returns list sorted by frame number.
    """
    frame_pattern = re.compile(r"frame_(\d+)\.jpg$")
    frames = []
    for f in clip_dir.iterdir():
        if f.is_file():
            match = frame_pattern.match(f.name)
            if match:
                frame_num = int(match.group(1))
                frames.append((frame_num, f))
    frames.sort(key=lambda x: x[0])
    return frames


def encode_frame_base64(frame_path: Path) -> str:
    """Read a JPEG frame file and return its base64-encoded string."""
    with open(frame_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("ascii")


def build_batch_messages(
    batch: list[tuple[int, Path]],
    previously_identified: list[dict],
) -> list[dict]:
    """Build the messages array for a Claude API call.

    Each batch sends frames as base64 images with frame number labels,
    plus context about previously identified subjects.
    """
    content_blocks = []

    # If there are previously identified subjects, include them as context
    if previously_identified:
        prev_text = (
            "Previously identified subjects (do NOT re-report these):\n"
            + json.dumps(previously_identified, indent=2)
            + "\n\nNow analyse the following new frames for any NEW subjects:"
        )
        content_blocks.append({"type": "text", "text": prev_text})
    else:
        content_blocks.append({
            "type": "text",
            "text": "Analyse the following frames for subjects:",
        })

    # Add each frame as an image with its frame number label
    for frame_num, frame_path in batch:
        content_blocks.append({
            "type": "text",
            "text": f"Frame {frame_num}:",
        })
        content_blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": encode_frame_base64(frame_path),
            },
        })

    return [{"role": "user", "content": content_blocks}]


def parse_claude_response(response_text: str) -> list[dict]:
    """Parse Claude's JSON response into a list of subject dicts.

    Handles cases where Claude wraps JSON in markdown code blocks.
    Returns empty list if parsing fails.
    """
    text = response_text.strip()

    # Strip markdown code block wrappers if present
    if text.startswith("```"):
        # Remove opening ``` line (with optional language tag)
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        # Remove closing ```
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        logger.warning("Claude returned JSON but not an array; wrapping in array")
        return [result] if isinstance(result, dict) else []
    except json.JSONDecodeError as e:
        logger.error("Failed to parse Claude response as JSON: %s", e)
        logger.error("Raw response:\n%s", response_text[:2000])
        return []


def validate_subject(subject: dict) -> bool:
    """Validate that a subject dict has the required fields with correct types."""
    required_fields = {
        "label": str,
        "first_appearance_frame": int,
        "bbox_normalised": list,
        "confidence": str,
    }
    for field, expected_type in required_fields.items():
        if field not in subject:
            logger.warning("Subject missing field '%s': %s", field, subject)
            return False
        if not isinstance(subject[field], expected_type):
            logger.warning(
                "Subject field '%s' has wrong type (expected %s): %s",
                field, expected_type.__name__, subject,
            )
            return False

    # Validate bbox has 4 elements, all floats/ints between 0 and 1
    bbox = subject["bbox_normalised"]
    if len(bbox) != 4:
        logger.warning("bbox_normalised must have 4 elements: %s", subject)
        return False
    for val in bbox:
        if not isinstance(val, (int, float)) or val < 0.0 or val > 1.0:
            logger.warning("bbox_normalised values must be 0.0-1.0: %s", subject)
            return False

    # Validate confidence
    if subject["confidence"] not in ("high", "medium", "low"):
        logger.warning("confidence must be high/medium/low: %s", subject)
        return False

    return True


def estimate_batch_cost(batch: list[tuple[int, Path]], model: str) -> dict:
    """Estimate token usage and cost for a batch.

    Very rough estimation:
    - Each image at 1280px wide ~ 1600 tokens (medium detail)
    - System prompt ~ 500 tokens
    - Text content per frame ~ 10 tokens
    - Response ~ 200 tokens per subject found (estimate 2-3 subjects)

    Returns dict with estimated input_tokens, output_tokens, and cost_usd.
    """
    num_frames = len(batch)
    # Rough token estimates
    image_tokens = num_frames * 1600
    system_tokens = 500
    text_tokens = num_frames * 10 + 200  # frame labels + previously identified
    input_tokens = image_tokens + system_tokens + text_tokens
    output_tokens = 500  # rough estimate for response

    # Pricing for claude-sonnet-4-6 (rough): $3/MTok input, $15/MTok output
    cost_usd = (input_tokens * 3.0 + output_tokens * 15.0) / 1_000_000

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
    }


def analyse_clip(
    clip_dir: Path,
    clip_id: str,
    client,  # anthropic.Anthropic client
    model: str,
    batch_size: int,
    dry_run: bool = False,
) -> dict:
    """Run scene analysis on a single clip.

    Returns the output dict with clip_id and subjects list.
    """
    frames = get_sorted_frames(clip_dir)
    if not frames:
        logger.warning("No keyframes found in %s, skipping", clip_dir)
        return {"clip_id": clip_id, "subjects": []}

    logger.info("Clip %s: %d keyframe(s)", clip_id, len(frames))

    # Split frames into batches
    batches = []
    for i in range(0, len(frames), batch_size):
        batches.append(frames[i : i + batch_size])

    logger.info("Clip %s: %d batch(es) of up to %d frames", clip_id, len(batches), batch_size)

    all_subjects = []
    total_input_tokens = 0
    total_output_tokens = 0

    for batch_idx, batch in enumerate(batches):
        frame_range = f"{batch[0][0]}-{batch[-1][0]}"
        logger.info(
            "  Batch %d/%d (frames %s, %d frame(s))",
            batch_idx + 1, len(batches), frame_range, len(batch),
        )

        if dry_run:
            cost_est = estimate_batch_cost(batch, model)
            logger.info(
                "    [DRY RUN] Would send %d frame(s) to %s "
                "(~%d input tokens, est. $%.4f)",
                len(batch), model, cost_est["input_tokens"], cost_est["cost_usd"],
            )
            continue

        # Build messages with accumulated context
        messages = build_batch_messages(batch, all_subjects)

        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
        except Exception as e:
            logger.error(
                "    API call failed for batch %d/%d: %s",
                batch_idx + 1, len(batches), e,
            )
            continue

        # Track token usage
        if hasattr(response, "usage"):
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens
            logger.debug(
                "    Tokens: %d in, %d out",
                response.usage.input_tokens, response.usage.output_tokens,
            )

        # Extract text response
        response_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                response_text += block.text

        if not response_text:
            logger.warning("    Empty response from Claude for batch %d", batch_idx + 1)
            continue

        # Parse subjects from response
        new_subjects = parse_claude_response(response_text)

        # Validate and add subjects
        valid_count = 0
        for subject in new_subjects:
            if validate_subject(subject):
                all_subjects.append(subject)
                valid_count += 1
                logger.info(
                    "    Found: %s (frame %d, confidence: %s)",
                    subject["label"],
                    subject["first_appearance_frame"],
                    subject["confidence"],
                )

        if valid_count == 0 and new_subjects:
            logger.warning(
                "    %d subject(s) returned but none passed validation",
                len(new_subjects),
            )
        elif valid_count == 0:
            logger.debug("    No new subjects in this batch")

    # Log token totals for this clip
    if not dry_run and total_input_tokens > 0:
        est_cost = (total_input_tokens * 3.0 + total_output_tokens * 15.0) / 1_000_000
        logger.info(
            "Clip %s: %d subject(s) found | Tokens: %d in, %d out (~$%.4f)",
            clip_id, len(all_subjects),
            total_input_tokens, total_output_tokens, est_cost,
        )

    return {"clip_id": clip_id, "subjects": all_subjects}


def main():
    parser = argparse.ArgumentParser(
        description="Run Claude scene analysis on keyframes"
    )
    parser.add_argument("--date", required=True, help="Dive date YYYY-MM-DD")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be sent without making API calls",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    workspace_root = find_workspace_root()
    logger.info("Workspace root: %s", workspace_root)

    config = load_config(workspace_root)

    # Resolve paths
    working_dir = workspace_root / config["working_dir"]
    keyframes_root = working_dir / "keyframes"
    scene_analysis_dir = working_dir / "scene_analysis"

    # Check for keyframe directories
    # Note: We scan all clip directories under keyframes_root. The --date flag
    # is used for logging/context; keyframes are organised by clip_id, not date.
    # Phase 1 extracts keyframes into working/keyframes/{clip_id}/ for the given
    # date's footage, so all clip dirs present are assumed to belong to the session.
    clip_dirs = discover_keyframe_dirs(keyframes_root)
    if not clip_dirs:
        logger.error(
            "No keyframe directories found under %s. "
            "Run Phase 1 (01_extract_frames.py --date %s) first.",
            keyframes_root, args.date,
        )
        sys.exit(1)

    logger.info("Found %d clip(s) with keyframes for date %s", len(clip_dirs), args.date)

    # Resolve config values
    model = config.get("claude_model", "claude-sonnet-4-6")
    batch_size = config.get("claude_batch_size", 12)

    # Resolve API key (not needed for dry-run)
    api_key = resolve_api_key(config)
    if not args.dry_run and not api_key:
        logger.error(
            "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable "
            "or configure anthropic_api_key in config.yaml."
        )
        sys.exit(1)

    # Initialize Anthropic client (only if not dry-run)
    client = None
    if not args.dry_run:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

    logger.info("Model: %s | Batch size: %d | Dry run: %s", model, batch_size, args.dry_run)

    # Create output directory
    scene_analysis_dir.mkdir(parents=True, exist_ok=True)

    # Process each clip
    total_subjects = 0
    total_api_calls = 0
    successful_clips = 0
    failed_clips = 0

    for clip_dir in clip_dirs:
        clip_id = clip_dir.name
        logger.info("--- Processing clip: %s ---", clip_id)

        result = analyse_clip(
            clip_dir=clip_dir,
            clip_id=clip_id,
            client=client,
            model=model,
            batch_size=batch_size,
            dry_run=args.dry_run,
        )

        if not args.dry_run:
            # Write output JSON
            output_path = scene_analysis_dir / f"{clip_id}.json"
            try:
                with open(output_path, "w") as f:
                    json.dump(result, f, indent=2)
                logger.info("Wrote: %s", output_path)
                successful_clips += 1
            except OSError as e:
                logger.error("Failed to write %s: %s", output_path, e)
                failed_clips += 1
                continue
        else:
            successful_clips += 1

        total_subjects += len(result["subjects"])

        # Count API calls (batches) for this clip
        frames = get_sorted_frames(clip_dir)
        num_batches = (len(frames) + batch_size - 1) // batch_size if frames else 0
        total_api_calls += num_batches

    # Summary
    logger.info("--- Scene analysis complete ---")
    logger.info(
        "Clips: %d processed, %d failed | Subjects found: %d | API calls: %d",
        successful_clips, failed_clips, total_subjects, total_api_calls,
    )

    if args.dry_run:
        logger.info("[DRY RUN] No API calls were made. No output files written.")

    # TODO: Pass 2 refinement is deferred. Currently using keyframe-level precision
    # for first_appearance_frame. When original footage frames are accessible at this
    # stage, implement Pass 2: around each detected appearance from Pass 1, sample
    # every 10 frames from the original footage to pinpoint exact first frame.
    # See architecture doc Section 4.2.3.

    if failed_clips > 0 and successful_clips == 0:
        logger.error("All clips failed. Check errors above.")
        sys.exit(1)

    if failed_clips > 0:
        logger.warning("%d clip(s) failed. Check errors above.", failed_clips)


if __name__ == "__main__":
    main()
