from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


@dataclass(frozen=True)
class SampledFrame:
    frame_id: str
    frame_index: int
    timestamp_seconds: float
    image_path: Path


def _resolve_device(device: str) -> str:
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError(f"unsupported device: {device!r}")
    if device == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        if device == "cuda":
            raise RuntimeError("device='cuda' requested but torch is not installed")
        return "cpu"
    if not torch.cuda.is_available():
        if device == "cuda":
            raise RuntimeError("device='cuda' requested but no CUDA device is available")
        return "cpu"
    return "cuda"


def _pairwise_percentile_abs_diff_cpu(gray_stack: np.ndarray, percentile: float) -> np.ndarray:
    """gray_stack: (N, H, W) uint8. Returns (N-1,) float32: the given percentile of the
    pixelwise abs diff between consecutive frames, taken over each frame pair's pixels."""
    diff = np.abs(gray_stack[1:].astype(np.int16) - gray_stack[:-1].astype(np.int16))
    flattened = diff.reshape(diff.shape[0], -1)
    return np.percentile(flattened, percentile, axis=1).astype(np.float32)


def _pairwise_percentile_abs_diff_gpu(gray_stack: np.ndarray, percentile: float) -> np.ndarray:
    import torch

    tensor = torch.from_numpy(gray_stack).to("cuda", dtype=torch.float32)
    diff = (tensor[1:] - tensor[:-1]).abs()
    flattened = diff.reshape(diff.shape[0], -1)
    quantile = torch.quantile(flattened, percentile / 100.0, dim=1)
    return quantile.cpu().numpy()


def _single_pair_diff(a: np.ndarray, b: np.ndarray, percentile: float) -> float:
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return float(np.percentile(diff, percentile))


def _iter_gray_chunks(cap: cv2.VideoCapture, downscale: tuple[int, int], chunk_size: int):
    while True:
        bgr_chunk: list[np.ndarray] = []
        gray_chunk: list[np.ndarray] = []
        for _ in range(chunk_size):
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(frame, downscale)
            gray_chunk.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
            bgr_chunk.append(frame)
        if not bgr_chunk:
            return
        yield bgr_chunk, np.stack(gray_chunk, axis=0)


def sample_by_information_gain(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    change_threshold: float = 45.0,
    percentile: float = 90.0,
    max_gap_seconds: float | None = 2.0,
    downscale: tuple[int, int] = (160, 90),
    device: str = "auto",
    chunk_size: int = 256,
) -> list[SampledFrame]:
    """Select frames by integrating local frame-to-frame visual change and firing
    whenever enough new information has accumulated since the last selected frame.

    Unlike a fixed or capped sampling rate, there is no ceiling: a single
    frame-to-frame jump large enough on its own crosses ``change_threshold``
    immediately, so brief fast motion between two consecutive raw frames is
    still kept -- a fixed max-fps sampler can silently step over exactly that
    frame. In principle this can select every consecutive raw frame back to
    back (matching the source video's native frame rate) if the content
    keeps changing that fast; there is no artificial cap independent of that.
    During static stretches, frames are skipped until ``max_gap_seconds`` (a
    safety floor so long static runs still get occasional coverage; pass
    ``None`` to disable it and allow arbitrarily long gaps) forces a
    checkpoint frame.

    The per-frame-pair change signal is the ``percentile``-th percentile of
    the pixelwise absolute difference (default: the 90th), not the mean.
    A mean is dominated by however much of the frame is static -- a change
    confined to a small region (hands unwrapping something, in a frame that
    is mostly still torso and background) can sit well under a mean-based
    threshold for many consecutive frames even though that region is
    genuinely, continuously changing, so a real multi-second transition can
    accumulate too slowly to ever fire and gets silently collapsed into a
    single stale sample. A high percentile reports the magnitude of the
    frame's most-changed pixels instead of averaging them away against the
    static majority, so it stays sensitive to localized-but-real change while
    still ignoring sensor noise (which affects most pixels a little, not a
    concentrated region a lot).

    The pairwise pixel-difference computation -- the only part of this
    amenable to vectorization -- runs in chunks, on the GPU when available
    (``device="auto"``/``"cuda"``). The accumulate-and-fire selection logic is
    inherently sequential (each decision depends on the running total since
    the last selection) and stays a cheap scalar scan over the resulting
    delta array on the CPU.
    """
    if change_threshold <= 0:
        raise ValueError("change_threshold must be greater than zero")
    if not (0.0 < percentile <= 100.0):
        raise ValueError("percentile must be between 0 and 100")
    if max_gap_seconds is not None and max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds must be greater than zero or None")
    if chunk_size < 2:
        raise ValueError("chunk_size must be at least 2")

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")

    resolved_device = _resolve_device(device)
    compute_deltas: Callable[[np.ndarray, float], np.ndarray] = (
        _pairwise_percentile_abs_diff_gpu if resolved_device == "cuda" else _pairwise_percentile_abs_diff_cpu
    )

    # A small tolerance absorbs floating-point noise in elapsed-time
    # subtraction (e.g. 1.2 - 1.1 == 0.09999999999999987 in binary float),
    # which would otherwise let a frame right at the max_gap boundary slip
    # past the check by a hair.
    epsilon = 1e-6

    frames: list[SampledFrame] = []
    accumulated = 0.0
    last_selected_time = 0.0
    video_stem = video_path.stem

    def select(bgr_frame: np.ndarray, frame_index: int, timestamp_seconds: float) -> None:
        nonlocal accumulated, last_selected_time
        frame_id = f"{video_stem}_{frame_index:06d}"
        image_path = output_dir / f"{frame_id}.jpg"
        if not cv2.imwrite(str(image_path), bgr_frame):
            raise OSError(f"failed to write sampled frame: {image_path}")
        frames.append(SampledFrame(frame_id, frame_index, timestamp_seconds, image_path))
        accumulated = 0.0
        last_selected_time = timestamp_seconds

    try:
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if not source_fps or source_fps <= 0:
            source_fps = 30.0

        frame_index = 0
        prev_gray: np.ndarray | None = None
        for bgr_chunk, gray_chunk in _iter_gray_chunks(cap, downscale, chunk_size):
            deltas: list[float | None] = []
            if prev_gray is not None:
                deltas.append(_single_pair_diff(prev_gray, gray_chunk[0], percentile))
            else:
                deltas.append(None)  # very first frame of the whole video: always selected
            if len(gray_chunk) > 1:
                deltas.extend(compute_deltas(gray_chunk, percentile).tolist())

            for offset, delta in enumerate(deltas):
                timestamp_seconds = frame_index / source_fps
                if delta is None:
                    select(bgr_chunk[offset], frame_index, timestamp_seconds)
                else:
                    accumulated += delta
                    elapsed = timestamp_seconds - last_selected_time
                    if accumulated >= change_threshold or (
                        max_gap_seconds is not None and elapsed >= max_gap_seconds - epsilon
                    ):
                        select(bgr_chunk[offset], frame_index, timestamp_seconds)
                frame_index += 1

            prev_gray = gray_chunk[-1]
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"no frames sampled from video: {video_path}")
    return frames
