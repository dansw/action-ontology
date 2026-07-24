from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

from frame_sampling import sample_by_information_gain

from .prompts import SYSTEM_PROMPT, frame_prompt
from .schema import FrameOntology
from .video import sample_video_frames, sample_video_frames_adaptive


def prepare_project(
    project_dir: str | Path,
    sample_fps: float,
    output_jsonl: str | Path,
    *,
    sampling: str = "fixed",
    min_fps: float = 1.0,
    max_fps: float = 15.0,
    motion_threshold: float = 6.0,
    change_threshold: float = 45.0,
    percentile: float = 90.0,
    max_gap_seconds: float | None = 2.0,
    context_frames: int = 4,
) -> int:
    if sampling not in ("fixed", "adaptive", "information-gain"):
        raise ValueError(f"unsupported sampling strategy: {sampling!r}")
    project_dir = Path(project_dir)
    videos_dir = project_dir / "videos"
    annotations_dir = project_dir / "annotations"
    frames_dir = project_dir / "frames"
    if not videos_dir.exists():
        raise FileNotFoundError(f"missing videos directory: {videos_dir}")
    if not annotations_dir.exists():
        raise FileNotFoundError(f"missing annotations directory: {annotations_dir}")
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_jsonl.open("w", encoding="utf-8") as output:
        for annotation_path in sorted(annotations_dir.glob("*.json")):
            annotation = _load_json(annotation_path)
            video_path = project_dir / annotation.get("video_path", f"videos/{annotation_path.stem}.mp4")
            frame_output_dir = frames_dir / annotation_path.stem
            if sampling == "adaptive":
                sampled = sample_video_frames_adaptive(
                    video_path, frame_output_dir, min_fps=min_fps, max_fps=max_fps, motion_threshold=motion_threshold
                )
            elif sampling == "information-gain":
                sampled = sample_by_information_gain(
                    video_path,
                    frame_output_dir,
                    change_threshold=change_threshold,
                    percentile=percentile,
                    max_gap_seconds=max_gap_seconds,
                )
            else:
                sampled = sample_video_frames(video_path, frame_output_dir, sample_fps)
            ontology_by_frame = {
                frame["frame_id"]: FrameOntology.from_dict(frame).to_dict()
                for frame in annotation.get("frames", [])
            }
            ontology_by_timestamp = {
                round(float(frame.get("timestamp_seconds", -1)), 3): FrameOntology.from_dict(frame).to_dict()
                for frame in annotation.get("frames", [])
                if "timestamp_seconds" in frame
            }
            history: deque[dict[str, Any]] = deque(maxlen=context_frames if context_frames > 0 else 0)
            for frame in sampled:
                expected = ontology_by_frame.get(frame.frame_id)
                if expected is None:
                    expected = ontology_by_timestamp.get(round(frame.timestamp_seconds, 3))
                if expected is None:
                    continue
                record = {
                    "image_path": str(frame.image_path),
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": frame_prompt(frame.frame_id, frame.timestamp_seconds, history=list(history)),
                        },
                        {"role": "assistant", "content": json.dumps(expected, ensure_ascii=False)},
                    ],
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                if context_frames > 0:
                    history.append(
                        {
                            "timestamp_seconds": expected.get("timestamp_seconds", frame.timestamp_seconds),
                            "description": expected.get("description", ""),
                            "actions": [action.get("name", "") for action in expected.get("actions", [])],
                        }
                    )
    if count == 0:
        raise ValueError("no training records were created; check annotation frame ids or timestamps")
    return count


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"annotation must be a JSON object: {path}")
    return value

