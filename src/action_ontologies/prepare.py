from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

from frame_sampling import sample_by_information_gain

from .prompts import SYSTEM_PROMPT, frame_prompt
from .schema import FrameOntology, normalize_key
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
            known_identifiers: dict[str, str] = {}
            identifier_names: dict[str, set] = {}
            for frame in sampled:
                expected = ontology_by_frame.get(frame.frame_id)
                if expected is None:
                    expected = ontology_by_timestamp.get(round(frame.timestamp_seconds, 3))
                if expected is None:
                    continue
                prompt = frame_prompt(
                    frame.frame_id,
                    frame.timestamp_seconds,
                    history=list(history),
                    known_identifiers=dict(known_identifiers),
                )
                _canonicalize_known_identifiers(known_identifiers, identifier_names, expected)
                record = {
                    "image_path": str(frame.image_path),
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": json.dumps(expected, ensure_ascii=False)},
                    ],
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                if context_frames > 0:
                    history.append(
                        {
                            "timestamp_seconds": expected.get("timestamp_seconds", frame.timestamp_seconds),
                            "actions": [action.get("name", "") for action in expected.get("actions", [])],
                        }
                    )
    if count == 0:
        raise ValueError("no training records were created; check annotation frame ids or timestamps")
    return count


_MAX_KNOWN_IDENTIFIERS = 30


def _name_tokens(name: str) -> frozenset:
    return frozenset(normalize_key(name).split())


def _same_referent(tokens_a: frozenset, tokens_b: frozenset) -> bool:
    """See infer.py's identical helper: true if one name's words are a
    non-empty subset of the other's, e.g. "duvet" vs "duvet fabric" -- but
    not for names that merely share a generic word, so a deliberate
    state-change re-identifier (e.g. "wrapper" -> "wrapper fragment") is
    preserved rather than merged."""
    return bool(tokens_a) and bool(tokens_b) and (tokens_a <= tokens_b or tokens_b <= tokens_a)


def _canonicalize_known_identifiers(
    known_identifiers: dict[str, str],
    identifier_names: dict[str, set],
    expected: dict[str, Any],
) -> None:
    """Mirrors infer.py's drift-pruning logic: if this frame's annotation
    gives an element a name that matches -- exactly, or by whole-word
    containment -- a name already registered under a different identifier,
    rewrite it to the existing identifier in place so the training target
    itself stays consistent, and the model isn't trained against a
    video-local registry that already contains the duplicate. Runs
    identically for a duplicate introduced in an earlier frame or earlier in
    the SAME frame's own resources/entities list, then collapses any
    elements within one list that end up sharing an identifier."""
    for key in ("resources", "entities"):
        elements = expected.get(key) or []
        for element in elements:
            identifier = element.get("identifier")
            name = element.get("name")
            if not identifier or not name:
                continue
            new_tokens = _name_tokens(name)
            canonical = identifier
            for existing_identifier, historical in identifier_names.items():
                if existing_identifier == identifier:
                    continue
                if any(_same_referent(new_tokens, existing) for existing in historical):
                    canonical = existing_identifier
                    break
            if canonical != identifier:
                element["identifier"] = canonical
                identifier = canonical
            identifier_names.setdefault(canonical, set()).add(new_tokens)
            known_identifiers[canonical] = name

        seen: set = set()
        deduped = []
        for element in elements:
            dedup_key = element.get("identifier") or normalize_key(element.get("name", ""))
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            deduped.append(element)
        expected[key] = deduped

    while len(known_identifiers) > _MAX_KNOWN_IDENTIFIERS:
        oldest = next(iter(known_identifiers))
        known_identifiers.pop(oldest)
        identifier_names.pop(oldest, None)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"annotation must be a JSON object: {path}")
    return value
