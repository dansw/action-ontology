import json
from pathlib import Path

import pytest

from action_ontologies import prepare_v8 as module


def _create_expected_videos(directory: Path) -> None:
    directory.mkdir()
    for _, filename in module.V8_VIDEO_PROJECTS.values():
        (directory / filename).touch()


def test_prepare_v8_reports_all_missing_videos(tmp_path: Path) -> None:
    videos = tmp_path / "originals"
    videos.mkdir()

    with pytest.raises(FileNotFoundError, match=r"missing 15 required V8 video\(s\)"):
        module.prepare_v8(videos)


def test_prepare_v8_extracts_manifest_frames_and_splits_projects(tmp_path: Path, monkeypatch) -> None:
    videos = tmp_path / "originals"
    data = tmp_path / "data"
    _create_expected_videos(videos)
    manifest = data / "combined_v8/prepared/train.jsonl"
    manifest.parent.mkdir(parents=True)
    records = [
        {"image_path": "data/egg_catch/frames/egg_catch_001/egg_catch_001_000008.jpg"},
        {"image_path": "data/diverse_actions/frames/yo_yo/yo_yo_000123.jpg"},
    ]
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    extracted = []
    monkeypatch.setattr(module, "_extract_frames", lambda video, targets: extracted.append((video, targets)))
    monkeypatch.chdir(tmp_path)

    assert module.prepare_v8(videos) == 2
    assert len(extracted) == 2
    assert (data / "egg_catch/prepared/train.jsonl").read_text(encoding="utf-8").count("\n") == 1
    assert (data / "diverse_actions/prepared/train.jsonl").read_text(encoding="utf-8").count("\n") == 1
