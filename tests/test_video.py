from pathlib import Path

import cv2
import numpy as np
import pytest

from action_ontologies.video import sample_video_frames


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
