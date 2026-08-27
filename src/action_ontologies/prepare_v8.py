from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2


V8_VIDEO_PROJECTS = {
    "egg_catch_001": ("egg_catch", "egg_catch_001.mov"),
    "carry_coffee_table": ("diverse_actions", "carry_coffee_table.mov"),
    "duvet_cover": ("diverse_actions", "duvet_cover.mov"),
    "get_into_car": ("diverse_actions", "get_into_car.mov"),
    "make_a_bed": ("diverse_actions", "make_a_bed.mov"),
    "yo_yo": ("diverse_actions", "yo_yo.mov"),
    "clean_ketchup": ("survey_actions", "clean_ketchup.mov"),
    "cook_sunny_side_up_egg": ("survey_actions", "cook_sunny_side_up_egg.mov"),
    "fitted_sheet_on_bed": ("survey_actions", "fitted_sheet_on_bed.mov"),
    "hammer_and_nail": ("survey_actions", "hammer_and_nail.mov"),
    "opening_granola_bar": ("survey_actions", "opening_granola_bar.mov"),
    "separate_the_yolk": ("survey_actions", "separate_the_yolk.mov"),
    "sort_poker_hand": ("survey_actions", "sort_poker_hand.mov"),
    "turn_page_paperback": ("survey_actions", "turn_page_paperback.mov"),
    "take_out_trash": ("survey_actions_trash", "take_out_trash.mov"),
}


def _extract_frames(video_path: Path, targets: list[tuple[int, Path]]) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    try:
        for frame_index, output_path in sorted(targets):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"could not read frame {frame_index} from {video_path}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), frame):
                raise RuntimeError(f"could not write extracted frame: {output_path}")
    finally:
        capture.release()


def prepare_v8(videos_dir: str | Path) -> int:
    videos_dir = Path(videos_dir)
    data_dir = Path("data")
    if not videos_dir.is_dir():
        raise FileNotFoundError(f"missing original-video directory: {videos_dir}")

    missing = sorted(filename for _, filename in V8_VIDEO_PROJECTS.values() if not (videos_dir / filename).is_file())
    if missing:
        raise FileNotFoundError(
            f"{videos_dir} is missing {len(missing)} required V8 video(s): " + ", ".join(missing)
        )

    manifest = data_dir / "combined_v8" / "prepared" / "train.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing committed V8 training manifest: {manifest}")

    targets_by_video: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    records_by_project: dict[str, list[str]] = defaultdict(list)
    seen_paths: set[Path] = set()
    with manifest.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            record = json.loads(line)
            image_path = Path(record["image_path"])
            parts = image_path.parts
            if len(parts) < 5 or parts[0] != "data" or parts[2] != "frames":
                raise ValueError(f"unexpected image_path on manifest line {line_number}: {image_path}")
            project, video_id = parts[1], parts[3]
            if video_id not in V8_VIDEO_PROJECTS or V8_VIDEO_PROJECTS[video_id][0] != project:
                raise ValueError(f"unknown V8 video on manifest line {line_number}: {video_id}")
            frame_index = int(image_path.stem.rsplit("_", 1)[1])
            output_path = data_dir.joinpath(*parts[1:])
            if output_path not in seen_paths:
                targets_by_video[video_id].append((frame_index, output_path))
                seen_paths.add(output_path)
            records_by_project[project].append(line)

    for video_id, targets in targets_by_video.items():
        _, filename = V8_VIDEO_PROJECTS[video_id]
        _extract_frames(videos_dir / filename, targets)

    for project, records in records_by_project.items():
        output = data_dir / project / "prepared" / "train.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(records), encoding="utf-8")
        print(f"{project}: created {len(records)} records")

    total = sum(len(records) for records in records_by_project.values())
    print(f"V8 dataset ready: {total} records at {manifest}")
    return total
