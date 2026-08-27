import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from action_ontologies.prepare import prepare_project


def _write_video(path: Path, num_frames: int = 4, fps: float = 2.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (16, 16))
    for index in range(num_frames):
        writer.write(np.full((16, 16, 3), index * 20, dtype=np.uint8))
    writer.release()


def _write_annotation(path: Path) -> None:
    annotation = {
        "video_id": "clip",
        "video_path": "videos/clip.mp4",
        "frames": [
            {
                "frame_id": "clip_000000",
                "timestamp_seconds": 0.0,
                "description": "person picks up pan",
                "actions": [{"name": "pick up pan", "actor": "person", "target": "pan"}],
            },
            {
                "frame_id": "clip_000001",
                "timestamp_seconds": 0.5,
                "description": "person cooks egg in pan",
                "actions": [{"name": "cook egg", "actor": "person", "target": "egg"}],
            },
            {
                "frame_id": "clip_000002",
                "timestamp_seconds": 1.0,
                "description": "person plates the egg",
                "actions": [{"name": "plate egg", "actor": "person", "target": "egg"}],
            },
        ],
    }
    path.write_text(json.dumps(annotation), encoding="utf-8")


def _make_project(tmp_path: Path) -> Path:
    project_dir = tmp_path / "project"
    (project_dir / "videos").mkdir(parents=True)
    (project_dir / "annotations").mkdir(parents=True)
    _write_video(project_dir / "videos" / "clip.mp4")
    _write_annotation(project_dir / "annotations" / "clip.json")
    return project_dir


def _load_records(output_jsonl: Path) -> list[dict]:
    return [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]


def test_prepare_project_includes_growing_history_by_default(tmp_path: Path):
    project_dir = _make_project(tmp_path)
    output_jsonl = tmp_path / "train.jsonl"

    count = prepare_project(project_dir, sample_fps=2.0, output_jsonl=output_jsonl)

    assert count == 3
    records = _load_records(output_jsonl)
    user_prompts = [record["messages"][1]["content"] for record in records]

    assert "Earlier SAMPLED observations" not in user_prompts[0]
    assert "Earlier SAMPLED observations" in user_prompts[1]
    assert "pick up pan" in user_prompts[1]
    assert "person picks up pan" not in user_prompts[1]
    assert "Earlier SAMPLED observations" in user_prompts[2]
    assert "pick up pan" in user_prompts[2]
    assert "cook egg" in user_prompts[2]
    assert "person cooks egg in pan" not in user_prompts[2]
    # history must not leak the current frame's own ground truth
    assert "person plates the egg" not in user_prompts[2]


def test_prepare_project_context_frames_zero_disables_history(tmp_path: Path):
    project_dir = _make_project(tmp_path)
    output_jsonl = tmp_path / "train.jsonl"

    prepare_project(project_dir, sample_fps=2.0, output_jsonl=output_jsonl, context_frames=0)

    records = _load_records(output_jsonl)
    for record in records:
        assert "Earlier SAMPLED observations" not in record["messages"][1]["content"]


def _write_annotation_with_drifted_identifiers(path: Path) -> None:
    annotation = {
        "video_id": "clip",
        "video_path": "videos/clip.mp4",
        "frames": [
            {
                "frame_id": "clip_000000",
                "timestamp_seconds": 0.0,
                "description": "person holds bedding",
                "entities": [{"name": "duvet fabric", "identifier": "bedding"}],
            },
            {
                "frame_id": "clip_000001",
                "timestamp_seconds": 0.5,
                "description": "person still holds bedding",
                "entities": [{"name": "duvet fabric", "identifier": "fabric"}],
            },
            {
                "frame_id": "clip_000002",
                "timestamp_seconds": 1.0,
                "description": "person sets bedding down",
                "entities": [{"name": "duvet fabric", "identifier": "bedding"}],
            },
        ],
    }
    path.write_text(json.dumps(annotation), encoding="utf-8")


def test_prepare_project_prunes_drifted_identifier_duplicates(tmp_path: Path):
    project_dir = tmp_path / "project"
    (project_dir / "videos").mkdir(parents=True)
    (project_dir / "annotations").mkdir(parents=True)
    _write_video(project_dir / "videos" / "clip.mp4")
    _write_annotation_with_drifted_identifiers(project_dir / "annotations" / "clip.json")
    output_jsonl = tmp_path / "train.jsonl"

    prepare_project(project_dir, sample_fps=2.0, output_jsonl=output_jsonl)

    records = _load_records(output_jsonl)
    assistant_payloads = [json.loads(record["messages"][2]["content"]) for record in records]
    identifiers = [entity["identifier"] for payload in assistant_payloads for entity in payload["entities"]]
    # frame 1's raw annotation used "fabric" for the same-named entity already
    # registered as "bedding" -- the training target must be rewritten to the
    # existing identifier, not kept as a second one.
    assert identifiers == ["bedding", "bedding", "bedding"]

    last_prompt = records[2]["messages"][1]["content"]
    assert "bedding: duvet fabric" in last_prompt
    assert "fabric: duvet fabric" not in last_prompt


def _write_annotation_with_fuzzy_drift_and_same_frame_duplicate(path: Path) -> None:
    annotation = {
        "video_id": "clip",
        "video_path": "videos/clip.mp4",
        "frames": [
            {
                "frame_id": "clip_000000",
                "timestamp_seconds": 0.0,
                "description": "person holds bedding",
                "entities": [{"name": "duvet fabric", "identifier": "bedding"}],
            },
            {
                "frame_id": "clip_000001",
                "timestamp_seconds": 0.5,
                # a brand-new identifier "duvet" for the same real object,
                # matched via whole-word containment ("duvet" subset of the
                # already-registered "duvet fabric"), not an exact match
                "description": "person still holds it",
                "entities": [{"name": "duvet", "identifier": "duvet"}],
            },
            {
                "frame_id": "clip_000002",
                "timestamp_seconds": 1.0,
                # both names listed as separate entities in the SAME frame
                "description": "person sets bedding down",
                "entities": [
                    {"name": "duvet fabric", "identifier": "bedding"},
                    {"name": "duvet", "identifier": "duvet"},
                ],
            },
        ],
    }
    path.write_text(json.dumps(annotation), encoding="utf-8")


def test_prepare_project_prunes_fuzzy_and_same_frame_identifier_duplicates(tmp_path: Path):
    project_dir = tmp_path / "project"
    (project_dir / "videos").mkdir(parents=True)
    (project_dir / "annotations").mkdir(parents=True)
    _write_video(project_dir / "videos" / "clip.mp4")
    _write_annotation_with_fuzzy_drift_and_same_frame_duplicate(project_dir / "annotations" / "clip.json")
    output_jsonl = tmp_path / "train.jsonl"

    prepare_project(project_dir, sample_fps=2.0, output_jsonl=output_jsonl)

    records = _load_records(output_jsonl)
    assistant_payloads = [json.loads(record["messages"][2]["content"]) for record in records]

    # frame 1's "duvet" must be rewritten onto "bedding" (fuzzy, cross-frame)
    assert assistant_payloads[1]["entities"] == [{"name": "duvet", "identifier": "bedding"}]

    # frame 2's two entities collapse into one (same-frame duplicate)
    assert len(assistant_payloads[2]["entities"]) == 1
    assert assistant_payloads[2]["entities"][0]["identifier"] == "bedding"


def test_prepare_project_context_frames_caps_window(tmp_path: Path):
    project_dir = _make_project(tmp_path)
    output_jsonl = tmp_path / "train.jsonl"

    prepare_project(project_dir, sample_fps=2.0, output_jsonl=output_jsonl, context_frames=1)

    records = _load_records(output_jsonl)
    last_prompt = records[2]["messages"][1]["content"]
    # only the immediately preceding frame should appear, not the one before that
    assert "cook egg" in last_prompt
    assert "person cooks egg in pan" not in last_prompt
    assert "pick up pan" not in last_prompt
