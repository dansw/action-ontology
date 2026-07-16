from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class SampledFrame:
    frame_id: str
    frame_index: int
    timestamp_seconds: float
    image_path: Path


def sample_video_frames(video_path: str | Path, output_dir: str | Path, sample_fps: float) -> list[SampledFrame]:
    if sample_fps <= 0:
        raise ValueError("sample_fps must be greater than zero")
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    try:
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if not source_fps or source_fps <= 0:
            source_fps = sample_fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(round(source_fps / sample_fps)))
        frames: list[SampledFrame] = []
        frame_index = 0
        video_stem = video_path.stem
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % step == 0:
                timestamp_seconds = frame_index / source_fps
                frame_id = f"{video_stem}_{frame_index:06d}"
                image_path = output_dir / f"{frame_id}.jpg"
                if not cv2.imwrite(str(image_path), frame):
                    raise OSError(f"failed to write sampled frame: {image_path}")
                frames.append(
                    SampledFrame(
                        frame_id=frame_id,
                        frame_index=frame_index,
                        timestamp_seconds=timestamp_seconds,
                        image_path=image_path,
                    )
                )
            frame_index += 1
            if total_frames and frame_index >= total_frames:
                break
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"no frames sampled from video: {video_path}")
    return frames

