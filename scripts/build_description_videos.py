from __future__ import annotations

import bisect
import json
import subprocess
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
OUTPUT_WIDTH = 540
OUTPUT_FPS_TARGET = 24.0
OUTPUT_DIR = REPO_ROOT / "outputs" / "videos"

VIDEOS = {
    "egg_catch_001": (
        REPO_ROOT / "data/original_videos/egg_catch_001.mov",
        REPO_ROOT / "outputs/diverse_validation/egg_catch_001.ontology.json",
    ),
    "carry_coffee_table": (
        REPO_ROOT / "data/original_videos/carry_coffee_table.mov",
        REPO_ROOT / "outputs/diverse_validation/carry_coffee_table.ontology.json",
    ),
    "duvet_cover": (
        REPO_ROOT / "data/original_videos/duvet_cover.mov",
        REPO_ROOT / "outputs/diverse_validation/duvet_cover.ontology.json",
    ),
    "make_a_bed": (
        REPO_ROOT / "data/original_videos/make_a_bed.mov",
        REPO_ROOT / "outputs/diverse_validation/make_a_bed.ontology.json",
    ),
    "yo_yo": (
        REPO_ROOT / "data/original_videos/yo_yo.mov",
        REPO_ROOT / "outputs/diverse_validation/yo_yo.ontology.json",
    ),
    "get_into_car": (
        REPO_ROOT / "data/original_videos/get_into_car.mov",
        REPO_ROOT / "outputs/diverse_validation/get_into_car.ontology.json",
    ),
}


def load_captions(ontology_path: Path) -> tuple[list[float], list[str]]:
    data = json.loads(ontology_path.read_text(encoding="utf-8"))
    entries = sorted(
        (
            (float(f["timestamp_seconds"]), f.get("description", ""))
            for f in data["frames"]
            if not f.get("error")
        ),
        key=lambda pair: pair[0],
    )
    timestamps = [t for t, _ in entries]
    descriptions = [d for _, d in entries]
    return timestamps, descriptions


def caption_for_time(timestamps: list[float], descriptions: list[str], t: float) -> str:
    idx = bisect.bisect_right(timestamps, t) - 1
    idx = max(0, min(idx, len(descriptions) - 1))
    return descriptions[idx]


def overlay_caption(img: Image.Image, text: str) -> Image.Image:
    width, height = img.size
    draw = ImageDraw.Draw(img)
    font_size = max(14, width // 24)
    font = ImageFont.truetype(FONT_PATH, font_size)
    max_chars_per_line = max(10, int(width / (font_size * 0.52)))
    wrapped = textwrap.wrap(text, width=max_chars_per_line) or [""]
    line_height = font_size + 6
    bar_height = line_height * len(wrapped) + 16
    draw.rectangle([0, height - bar_height, width, height], fill=(0, 0, 0))
    y = height - bar_height + 8
    for line in wrapped:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) / 2
        draw.text((x, y), line, font=font, fill=(255, 255, 255))
        y += line_height
    return img


def build_video(slug: str, video_path: Path, ontology_path: Path) -> None:
    timestamps, descriptions = load_captions(ontology_path)
    if not timestamps:
        print(f"no usable captions for {slug}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or OUTPUT_FPS_TARGET
    step = max(1, round(source_fps / OUTPUT_FPS_TARGET))
    output_fps = source_fps / step

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        print(f"could not read first frame for {slug}")
        return
    h, w = first_frame.shape[:2]
    out_h = int(h * (OUTPUT_WIDTH / w))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    intermediate_path = OUTPUT_DIR / f"{slug}.raw.mp4"
    writer = cv2.VideoWriter(
        str(intermediate_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (OUTPUT_WIDTH, out_h)
    )
    try:
        frame_index = 0
        written = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % step == 0:
                timestamp = frame_index / source_fps
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb).resize((OUTPUT_WIDTH, out_h), Image.LANCZOS)
                caption = caption_for_time(timestamps, descriptions, timestamp)
                img = overlay_caption(img, caption)
                bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                writer.write(bgr)
                written += 1
            frame_index += 1
    finally:
        writer.release()
        cap.release()

    final_path = OUTPUT_DIR / f"{slug}.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(intermediate_path),
            "-c:v", "libx264", "-profile:v", "main", "-level", "3.1",
            "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p",
            str(final_path),
        ],
        check=True,
    )
    intermediate_path.unlink()
    size_mb = final_path.stat().st_size / (1024 * 1024)
    print(f"wrote {final_path} ({written} frames @ {output_fps:.1f}fps, {size_mb:.1f} MB)")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for slug, (video_path, ontology_path) in VIDEOS.items():
        build_video(slug, video_path, ontology_path)


if __name__ == "__main__":
    main()
