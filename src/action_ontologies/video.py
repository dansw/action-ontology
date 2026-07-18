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


def _open_capture(video_path: Path, output_dir: Path) -> cv2.VideoCapture:
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    return cap


def sample_video_frames(video_path: str | Path, output_dir: str | Path, sample_fps: float) -> list[SampledFrame]:
    if sample_fps <= 0:
        raise ValueError("sample_fps must be greater than zero")
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    cap = _open_capture(video_path, output_dir)
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


_MOTION_DOWNSCALE = (160, 90)


def sample_video_frames_adaptive(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    min_fps: float = 1.0,
    max_fps: float = 15.0,
    motion_threshold: float = 6.0,
) -> list[SampledFrame]:
    """Sample frames at a variable rate driven by how much the picture is changing.

    A frame is kept as soon as either condition holds:
    - it has been longer than ``1 / min_fps`` since the last kept frame (a floor,
      so static stretches still get occasional coverage), or
    - it has been at least ``1 / max_fps`` since the last kept frame *and* the
      downscaled grayscale difference from the last kept frame exceeds
      ``motion_threshold`` (a ceiling on how dense fast motion can get sampled).

    This mirrors what a human scrubbing the video for changes would do: skim past
    static stretches and slow down where the picture is actually moving.
    """
    if min_fps <= 0 or max_fps <= 0:
        raise ValueError("min_fps and max_fps must be greater than zero")
    if max_fps < min_fps:
        raise ValueError("max_fps must be greater than or equal to min_fps")
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    cap = _open_capture(video_path, output_dir)

    # A small tolerance absorbs floating-point noise in elapsed-time
    # subtraction (e.g. 1.2 - 1.1 == 0.09999999999999987 in binary float),
    # which would otherwise cause legitimate frames right at a gap boundary
    # to be silently skipped.
    epsilon = 1e-6
    min_gap = 1.0 / max_fps - epsilon
    max_gap = 1.0 / min_fps - epsilon

    try:
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if not source_fps or source_fps <= 0:
            source_fps = max_fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        video_stem = video_path.stem

        frames: list[SampledFrame] = []
        frame_index = 0
        last_selected_time = -max_gap
        last_gray = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_seconds = frame_index / source_fps
            elapsed = timestamp_seconds - last_selected_time

            small = cv2.resize(frame, _MOTION_DOWNSCALE)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            select = last_gray is None or elapsed >= max_gap
            if not select and elapsed >= min_gap:
                motion = float(cv2.absdiff(gray, last_gray).mean())
                select = motion >= motion_threshold

            if select:
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
                last_selected_time = timestamp_seconds
                last_gray = gray

            frame_index += 1
            if total_frames and frame_index >= total_frames:
                break
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"no frames sampled from video: {video_path}")
    return frames

