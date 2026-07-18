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

    assert "Recent frame history" not in user_prompts[0]
    assert "Recent frame history" in user_prompts[1]
    assert "person picks up pan" in user_prompts[1]
    assert "Recent frame history" in user_prompts[2]
    assert "person picks up pan" in user_prompts[2]
    assert "person cooks egg in pan" in user_prompts[2]
    # history must not leak the current frame's own ground truth
    assert "person plates the egg" not in user_prompts[2]


def test_prepare_project_context_frames_zero_disables_history(tmp_path: Path):
    project_dir = _make_project(tmp_path)
    output_jsonl = tmp_path / "train.jsonl"

    prepare_project(project_dir, sample_fps=2.0, output_jsonl=output_jsonl, context_frames=0)

    records = _load_records(output_jsonl)
    for record in records:
        assert "Recent frame history" not in record["messages"][1]["content"]


def test_prepare_project_context_frames_caps_window(tmp_path: Path):
    project_dir = _make_project(tmp_path)
    output_jsonl = tmp_path / "train.jsonl"

    prepare_project(project_dir, sample_fps=2.0, output_jsonl=output_jsonl, context_frames=1)

    records = _load_records(output_jsonl)
    last_prompt = records[2]["messages"][1]["content"]
    # only the immediately preceding frame should appear, not the one before that
    assert "person cooks egg in pan" in last_prompt
    assert "person picks up pan" not in last_prompt
