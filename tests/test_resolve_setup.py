"""Tests for 05_resolve_setup.py pipeline logic.

Tests cover the functions that don't require a running DaVinci Resolve instance:
classify_subject, assign_subjects_to_nodes, discover_footage, load_scene_analysis,
find_mask_sequence_dir, find_depth_sequence_dir, and dry-run validation.
"""

import json
from pathlib import Path

import pytest

# Imported via conftest.py sys.path manipulation
from pipeline_utils import VIDEO_EXTENSIONS

# Import the functions under test from 05_resolve_setup
# Module name has a leading number, so use importlib
import importlib
resolve_setup = importlib.import_module("05_resolve_setup")

classify_subject = resolve_setup.classify_subject
assign_subjects_to_nodes = resolve_setup.assign_subjects_to_nodes
discover_footage = resolve_setup.discover_footage
load_scene_analysis = resolve_setup.load_scene_analysis
find_mask_sequence_dir = resolve_setup.find_mask_sequence_dir
find_depth_sequence_dir = resolve_setup.find_depth_sequence_dir
get_first_frame_path = resolve_setup.get_first_frame_path

NODE_FOREGROUND_POP = resolve_setup.NODE_FOREGROUND_POP
NODE_DIVER = resolve_setup.NODE_DIVER
NODE_MARINE_LIFE = resolve_setup.NODE_MARINE_LIFE


# --- classify_subject ---


class TestClassifySubject:
    def test_diver(self):
        assert classify_subject("diver") == NODE_DIVER
        assert classify_subject("Diver with camera") == NODE_DIVER
        assert classify_subject("DIVER") == NODE_DIVER
        assert classify_subject("scuba diver") == NODE_DIVER

    def test_marine_life(self):
        assert classify_subject("turtle") == NODE_MARINE_LIFE
        assert classify_subject("Green Sea Turtle") == NODE_MARINE_LIFE
        assert classify_subject("clownfish") == NODE_MARINE_LIFE
        assert classify_subject("coral reef") == NODE_MARINE_LIFE
        assert classify_subject("manta ray") == NODE_MARINE_LIFE
        assert classify_subject("nudibranch") == NODE_MARINE_LIFE

    def test_foreground(self):
        assert classify_subject("rock formation") == NODE_FOREGROUND_POP
        assert classify_subject("underwater cave") == NODE_FOREGROUND_POP
        assert classify_subject("anchor") == NODE_FOREGROUND_POP

    def test_case_insensitive(self):
        assert classify_subject("TURTLE") == NODE_MARINE_LIFE
        assert classify_subject("Diver") == NODE_DIVER
        assert classify_subject("CLOWNFISH") == NODE_MARINE_LIFE


# --- assign_subjects_to_nodes ---


class TestAssignSubjectsToNodes:
    def test_empty(self):
        assert assign_subjects_to_nodes([]) == {}

    def test_single_diver(self):
        subjects = [{"label": "diver", "confidence": "high"}]
        result = assign_subjects_to_nodes(subjects)
        assert NODE_DIVER in result
        assert result[NODE_DIVER]["label"] == "diver"

    def test_single_marine(self):
        subjects = [{"label": "turtle", "confidence": "high"}]
        result = assign_subjects_to_nodes(subjects)
        assert NODE_MARINE_LIFE in result

    def test_single_foreground(self):
        subjects = [{"label": "shipwreck", "confidence": "medium"}]
        result = assign_subjects_to_nodes(subjects)
        assert NODE_FOREGROUND_POP in result

    def test_mixed_subjects(self):
        subjects = [
            {"label": "diver", "confidence": "high"},
            {"label": "turtle", "confidence": "high"},
            {"label": "rock formation", "confidence": "medium"},
        ]
        result = assign_subjects_to_nodes(subjects)
        assert result[NODE_DIVER]["label"] == "diver"
        assert result[NODE_MARINE_LIFE]["label"] == "turtle"
        assert result[NODE_FOREGROUND_POP]["label"] == "rock formation"

    def test_duplicate_category_first_wins(self):
        """When two subjects map to the same node, the first one wins."""
        subjects = [
            {"label": "turtle", "confidence": "high"},
            {"label": "clownfish", "confidence": "medium"},
        ]
        result = assign_subjects_to_nodes(subjects)
        assert result[NODE_MARINE_LIFE]["label"] == "turtle"
        # clownfish should be unassigned (not in result)
        assert len(result) == 1

    def test_no_label_key_defaults_to_unknown(self):
        subjects = [{}]
        result = assign_subjects_to_nodes(subjects)
        # "unknown" doesn't match diver or marine life, so foreground
        assert NODE_FOREGROUND_POP in result


# --- discover_footage ---


class TestDiscoverFootage:
    def test_empty_dirs(self, tmp_path):
        raw = tmp_path / "raw"
        flat = tmp_path / "flat"
        raw.mkdir()
        flat.mkdir()
        assert discover_footage(raw, flat) == []

    def test_finds_video_files(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "clip1.mp4").write_bytes(b"fake")
        (raw / "clip2.MP4").write_bytes(b"fake")
        (raw / "readme.txt").write_bytes(b"not video")

        flat = tmp_path / "flat"
        flat.mkdir()

        result = discover_footage(raw, flat)
        assert len(result) == 2
        assert all(p.suffix in VIDEO_EXTENSIONS for p in result)

    def test_combines_raw_and_flat(self, tmp_path):
        raw = tmp_path / "raw"
        flat = tmp_path / "flat"
        raw.mkdir()
        flat.mkdir()
        (raw / "clip_a.mp4").write_bytes(b"fake")
        (flat / "clip_b.mov").write_bytes(b"fake")

        result = discover_footage(raw, flat)
        names = {p.name for p in result}
        assert "clip_a.mp4" in names
        assert "clip_b.mov" in names

    def test_nonexistent_dirs(self, tmp_path):
        result = discover_footage(tmp_path / "nope", tmp_path / "also_nope")
        assert result == []

    def test_sorted_output(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "z_clip.mp4").write_bytes(b"fake")
        (raw / "a_clip.mp4").write_bytes(b"fake")
        (raw / "m_clip.mp4").write_bytes(b"fake")

        flat = tmp_path / "flat"
        flat.mkdir()

        result = discover_footage(raw, flat)
        names = [p.name for p in result]
        assert names == sorted(names)


# --- load_scene_analysis ---


class TestLoadSceneAnalysis:
    def test_valid_json(self, tmp_path):
        data = {
            "subjects": [
                {"label": "diver", "first_appearance_frame": 0, "confidence": "high"},
                {"label": "turtle", "first_appearance_frame": 30, "confidence": "medium"},
            ]
        }
        json_path = tmp_path / "clip1.json"
        json_path.write_text(json.dumps(data))

        result = load_scene_analysis(tmp_path, "clip1")
        assert len(result) == 2
        assert result[0]["label"] == "diver"

    def test_missing_file(self, tmp_path):
        result = load_scene_analysis(tmp_path, "nonexistent")
        assert result == []

    def test_invalid_json(self, tmp_path):
        (tmp_path / "bad.json").write_text("{not valid json")
        result = load_scene_analysis(tmp_path, "bad")
        assert result == []

    def test_wrong_format(self, tmp_path):
        (tmp_path / "wrong.json").write_text(json.dumps({"data": []}))
        result = load_scene_analysis(tmp_path, "wrong")
        assert result == []

    def test_empty_subjects(self, tmp_path):
        (tmp_path / "empty.json").write_text(json.dumps({"subjects": []}))
        result = load_scene_analysis(tmp_path, "empty")
        assert result == []


# --- find_mask_sequence_dir ---


class TestFindMaskSequenceDir:
    def test_found(self, tmp_path):
        mask_dir = tmp_path / "clip1" / "diver"
        mask_dir.mkdir(parents=True)
        (mask_dir / "frame_0001.png").write_bytes(b"fake png")

        result = find_mask_sequence_dir(tmp_path, "clip1", "diver")
        assert result == mask_dir

    def test_missing_dir(self, tmp_path):
        result = find_mask_sequence_dir(tmp_path, "clip1", "diver")
        assert result is None

    def test_empty_dir(self, tmp_path):
        mask_dir = tmp_path / "clip1" / "diver"
        mask_dir.mkdir(parents=True)
        # No PNG files
        result = find_mask_sequence_dir(tmp_path, "clip1", "diver")
        assert result is None

    def test_non_png_files(self, tmp_path):
        mask_dir = tmp_path / "clip1" / "diver"
        mask_dir.mkdir(parents=True)
        (mask_dir / "frame.jpg").write_bytes(b"not png")
        result = find_mask_sequence_dir(tmp_path, "clip1", "diver")
        assert result is None


# --- find_depth_sequence_dir ---


class TestFindDepthSequenceDir:
    def test_found(self, tmp_path):
        depth_dir = tmp_path / "clip1"
        depth_dir.mkdir()
        (depth_dir / "frame_0001.png").write_bytes(b"fake png")

        result = find_depth_sequence_dir(tmp_path, "clip1")
        assert result == depth_dir

    def test_missing_dir(self, tmp_path):
        result = find_depth_sequence_dir(tmp_path, "clip1")
        assert result is None

    def test_empty_dir(self, tmp_path):
        depth_dir = tmp_path / "clip1"
        depth_dir.mkdir()
        result = find_depth_sequence_dir(tmp_path, "clip1")
        assert result is None


# --- get_first_frame_path ---


class TestGetFirstFramePath:
    def test_returns_first_sorted(self, tmp_path):
        (tmp_path / "frame_0003.png").write_bytes(b"c")
        (tmp_path / "frame_0001.png").write_bytes(b"a")
        (tmp_path / "frame_0002.png").write_bytes(b"b")

        result = get_first_frame_path(tmp_path)
        assert result == str(tmp_path / "frame_0001.png")

    def test_empty_dir(self, tmp_path):
        result = get_first_frame_path(tmp_path)
        assert result is None

    def test_non_png_excluded(self, tmp_path):
        (tmp_path / "frame_0001.jpg").write_bytes(b"not png")
        result = get_first_frame_path(tmp_path)
        assert result is None

    def test_single_frame(self, tmp_path):
        (tmp_path / "frame_0001.png").write_bytes(b"only one")
        result = get_first_frame_path(tmp_path)
        assert result == str(tmp_path / "frame_0001.png")
