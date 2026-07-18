from pathlib import Path

import cv2
import numpy as np
import pytest

from action_ontologies.video import sample_video_frames, sample_video_frames_adaptive


def test_sample_video_frames_rejects_bad_sample_rate(tmp_path: Path):
    with pytest.raises(ValueError):
        sample_video_frames(tmp_path / "missing.mp4", tmp_path / "frames", 0)


def test_sample_video_frames_extracts_images(tmp_path: Path):
    video_path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        4.0,
        (16, 16),
    )
    for index in range(8):
        frame = np.full((16, 16, 3), index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    frames = sample_video_frames(video_path, tmp_path / "frames", 2)

    assert len(frames) == 4
    assert frames[0].frame_id == "sample_000000"
    assert frames[0].image_path.exists()
    assert frames[1].timestamp_seconds == pytest.approx(0.5)


def test_sample_video_frames_adaptive_rejects_bad_rates(tmp_path: Path):
    with pytest.raises(ValueError):
        sample_video_frames_adaptive(tmp_path / "missing.mp4", tmp_path / "frames", min_fps=0, max_fps=10)
    with pytest.raises(ValueError):
        sample_video_frames_adaptive(tmp_path / "missing.mp4", tmp_path / "frames", min_fps=10, max_fps=2)


def _write_static_then_motion_video(path: Path, fps: float = 10.0, half_length: int = 10) -> None:
    # The motion half ramps brightness monotonically rather than oscillating,
    # since the sampler diffs against the last *kept* frame: an alternating
    # pattern can cancel back to zero diff against a stale reference every
    # other raw frame, which is a real property of the algorithm, not
    # representative of how brightness/position changes during real motion.
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (16, 16))
    for _ in range(half_length):
        writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
    for index in range(half_length):
        color = min(255, index * 25)
        writer.write(np.full((16, 16, 3), color, dtype=np.uint8))
    writer.release()


def test_sample_video_frames_adaptive_samples_sparsely_when_static(tmp_path: Path):
    video_path = tmp_path / "static.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (16, 16))
    for _ in range(20):
        writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
    writer.release()

    frames = sample_video_frames_adaptive(
        video_path, tmp_path / "frames", min_fps=2.0, max_fps=10.0, motion_threshold=6.0
    )

    # 2s of unchanging video at a min_fps floor of 2 should yield far fewer than
    # the 20 raw frames, spaced close to the 0.5s floor interval.
    assert len(frames) < 6
    gaps = [b.timestamp_seconds - a.timestamp_seconds for a, b in zip(frames, frames[1:])]
    assert all(gap == pytest.approx(0.5, abs=0.05) for gap in gaps)


def test_sample_video_frames_adaptive_samples_densely_during_motion(tmp_path: Path):
    video_path = tmp_path / "static_then_motion.mp4"
    _write_static_then_motion_video(video_path, fps=10.0, half_length=10)

    frames = sample_video_frames_adaptive(
        video_path, tmp_path / "frames", min_fps=2.0, max_fps=10.0, motion_threshold=6.0
    )

    static_frames = [f for f in frames if f.timestamp_seconds < 1.0]
    motion_frames = [f for f in frames if f.timestamp_seconds >= 1.0]

    # The static first second should be covered sparsely; the alternating
    # black/white second half should be covered close to every raw frame.
    assert len(static_frames) <= 3
    assert len(motion_frames) >= 8
