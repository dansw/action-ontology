from pathlib import Path

import cv2
import numpy as np
import pytest

from frame_sampling import sample_by_information_gain


def _write_video(path: Path, frames: list[np.ndarray], fps: float = 10.0) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()


def _solid(color: int) -> np.ndarray:
    return np.full((16, 16, 3), color, dtype=np.uint8)


def test_rejects_bad_params(tmp_path: Path):
    with pytest.raises(ValueError):
        sample_by_information_gain(tmp_path / "missing.mp4", tmp_path / "frames", change_threshold=0)
    with pytest.raises(ValueError):
        sample_by_information_gain(tmp_path / "missing.mp4", tmp_path / "frames", max_gap_seconds=0)
    with pytest.raises(ValueError):
        sample_by_information_gain(tmp_path / "missing.mp4", tmp_path / "frames", chunk_size=1)
    with pytest.raises(FileNotFoundError):
        sample_by_information_gain(tmp_path / "missing.mp4", tmp_path / "frames")


def test_samples_sparsely_when_static(tmp_path: Path):
    video_path = tmp_path / "static.mp4"
    _write_video(video_path, [_solid(0) for _ in range(30)], fps=10.0)

    frames = sample_by_information_gain(
        video_path, tmp_path / "frames", change_threshold=30.0, max_gap_seconds=1.0
    )

    # 3s of unchanging video with a 1s floor should yield roughly one frame
    # per second, not one per raw frame.
    assert 3 <= len(frames) <= 5


def test_max_gap_none_allows_arbitrarily_long_static_gaps(tmp_path: Path):
    video_path = tmp_path / "static.mp4"
    _write_video(video_path, [_solid(0) for _ in range(50)], fps=10.0)

    frames = sample_by_information_gain(
        video_path, tmp_path / "frames", change_threshold=30.0, max_gap_seconds=None
    )

    # With no floor and nothing ever changing, only the mandatory first frame
    # should be kept.
    assert len(frames) == 1
    assert frames[0].frame_index == 0


def test_captures_a_single_frame_transient_between_static_frames(tmp_path: Path):
    # A single bright flash lasting exactly one raw frame, surrounded by
    # otherwise-static video. A fixed-rate or max-fps-capped sampler can step
    # right over a transient like this; information gain must not, since the
    # transient's own frame-to-frame delta alone exceeds the threshold.
    frames_data = [_solid(0) for _ in range(15)] + [_solid(255)] + [_solid(0) for _ in range(15)]
    video_path = tmp_path / "flash.mp4"
    _write_video(video_path, frames_data, fps=10.0)

    frames = sample_by_information_gain(
        video_path, tmp_path / "frames", change_threshold=100.0, max_gap_seconds=None
    )

    selected_indices = {f.frame_index for f in frames}
    assert 15 in selected_indices, f"transient frame missed; selected indices were {sorted(selected_indices)}"
    # And virtually nothing else should have been selected around it, since
    # the rest of the video never changes.
    assert len(frames) <= 3


def test_samples_densely_during_sustained_motion(tmp_path: Path):
    frames_data = [_solid(0) for _ in range(10)]
    frames_data += [_solid(min(255, index * 25)) for index in range(10)]
    video_path = tmp_path / "ramp.mp4"
    _write_video(video_path, frames_data, fps=10.0)

    frames = sample_by_information_gain(
        video_path, tmp_path / "frames", change_threshold=20.0, max_gap_seconds=2.0
    )

    static_frames = [f for f in frames if f.timestamp_seconds < 1.0]
    motion_frames = [f for f in frames if f.timestamp_seconds >= 1.0]
    assert len(static_frames) <= 2
    assert len(motion_frames) >= 6


def _frame_with_patch(background: int, patch_value: int, size: int = 20, patch_rows: int = 6) -> np.ndarray:
    frame = np.full((size, size, 3), background, dtype=np.uint8)
    frame[:patch_rows, :, :] = patch_value
    return frame


def test_captures_localized_gradual_change_confined_to_part_of_the_frame(tmp_path: Path):
    # A change confined to ~30% of the frame (e.g. hands and a wrapper in one
    # corner while the rest of the body and background hold still) that ramps
    # up steadily, exactly like a wrapper being gradually torn open. A
    # mean-based signal dilutes this by however much of the frame is static
    # and can under-react; the percentile-based signal reports the changed
    # region's own magnitude undiluted, so it should still sample densely
    # while the ramp is happening, the same way full-frame motion does in
    # test_samples_densely_during_sustained_motion.
    frames_data = [_frame_with_patch(0, 0) for _ in range(10)]
    frames_data += [_frame_with_patch(0, min(255, index * 25)) for index in range(10)]
    video_path = tmp_path / "localized_ramp.mp4"
    _write_video(video_path, frames_data, fps=10.0)

    frames = sample_by_information_gain(
        video_path,
        tmp_path / "frames",
        change_threshold=20.0,
        max_gap_seconds=2.0,
        downscale=(20, 20),
    )

    static_frames = [f for f in frames if f.timestamp_seconds < 1.0]
    motion_frames = [f for f in frames if f.timestamp_seconds >= 1.0]
    assert len(static_frames) <= 2
    assert len(motion_frames) >= 6


def test_cpu_and_cuda_paths_agree_when_cuda_available(tmp_path: Path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")

    frames_data = [_solid(0) for _ in range(10)]
    frames_data += [_solid(min(255, index * 25)) for index in range(10)]
    video_path = tmp_path / "ramp.mp4"
    _write_video(video_path, frames_data, fps=10.0)

    cpu_frames = sample_by_information_gain(
        video_path, tmp_path / "cpu_frames", change_threshold=20.0, max_gap_seconds=2.0, device="cpu"
    )
    gpu_frames = sample_by_information_gain(
        video_path, tmp_path / "gpu_frames", change_threshold=20.0, max_gap_seconds=2.0, device="cuda"
    )

    assert [f.frame_index for f in cpu_frames] == [f.frame_index for f in gpu_frames]


def test_invalid_device_raises(tmp_path: Path):
    video_path = tmp_path / "static.mp4"
    _write_video(video_path, [_solid(0) for _ in range(5)], fps=10.0)
    with pytest.raises(ValueError):
        sample_by_information_gain(video_path, tmp_path / "frames", device="tpu")
