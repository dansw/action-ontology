from __future__ import annotations

import json
import textwrap
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
GIF_WIDTH = 400
MIN_HOLD_MS = 1000
MAX_HOLD_MS = 2200
MS_PER_CHAR = 35
OUTPUT_DIR = REPO_ROOT / "outputs" / "gifs"

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


def overlay_caption(img: Image.Image, text: str) -> Image.Image:
    width, height = img.size
    draw = ImageDraw.Draw(img)
    font_size = max(14, width // 20)
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


def build_gif(slug: str, video_path: Path, ontology_path: Path) -> None:
    data = json.loads(ontology_path.read_text(encoding="utf-8"))
    frames_data = [f for f in data["frames"] if not f.get("error")]
    if not frames_data:
        print(f"no usable frames for {slug}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")

    images: list[Image.Image] = []
    durations: list[int] = []
    try:
        for f in frames_data:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f["frame_index"])
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            w, h = img.size
            new_h = int(h * (GIF_WIDTH / w))
            img = img.resize((GIF_WIDTH, new_h), Image.LANCZOS)
            description = f.get("description", "")
            img = overlay_caption(img, description)
            images.append(img.convert("RGB"))
            durations.append(min(MAX_HOLD_MS, max(MIN_HOLD_MS, len(description) * MS_PER_CHAR)))
    finally:
        cap.release()

    if not images:
        print(f"no frames could be decoded for {slug}")
        return

    out_path = OUTPUT_DIR / f"{slug}.gif"
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"wrote {out_path} ({len(images)} frames, {sum(durations) / 1000:.1f}s playback)")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for slug, (video_path, ontology_path) in VIDEOS.items():
        build_gif(slug, video_path, ontology_path)


if __name__ == "__main__":
    main()
