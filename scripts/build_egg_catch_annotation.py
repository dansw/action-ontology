from __future__ import annotations

import json
from pathlib import Path

VIDEO_ID = "egg_catch_001"
SOURCE_FPS = 60.0
SAMPLE_FPS = 8.0
STEP = round(SOURCE_FPS / SAMPLE_FPS)
TOTAL_FRAMES = 63

LEFT_HAND = {"name": "left hand", "description": "the person's left gloved hand", "identifier": "left_hand"}
RIGHT_HAND = {"name": "right hand", "description": "the person's right gloved hand", "identifier": "right_hand"}
LEFT_ARM = {"name": "left arm", "description": "the person's left arm", "identifier": "left_arm"}
RIGHT_ARM = {"name": "right arm", "description": "the person's right arm", "identifier": "right_arm"}

TABLE_ENTITY = {"name": "table", "description": "white counter the person stands behind", "identifier": "white_table"}


def limbs():
    return [dict(LEFT_HAND), dict(RIGHT_HAND), dict(LEFT_ARM), dict(RIGHT_ARM)]


def standing_frame(frame_id: str, timestamp: float, frame_index: int) -> dict:
    return {
        "frame_id": frame_id,
        "timestamp_seconds": round(timestamp, 6),
        "frame_index": frame_index,
        "description": "A person in a silver suit stands facing forward behind a table, both arms extended out to the sides with empty gloved hands.",
        "resources": limbs(),
        "entities": [dict(TABLE_ENTITY)],
        "actions": [
            {
                "name": "stand",
                "actor": "person",
                "target": "table",
                "description": "the person stands still behind the table with both arms held open and hands empty",
            }
        ],
        "ontological_phrases": ["person stand with arms open and hands empty"],
    }


FALL_POSITIONS = {
    64: ("face", "directly in front of the person's face"),
    72: ("chest", "at chest height in front of the person"),
    80: ("waist", "near waist height as the person's hands begin to close"),
}


def falling_frame(frame_id: str, timestamp: float, frame_index: int) -> dict:
    place, place_desc = FALL_POSITIONS[frame_index]
    return {
        "frame_id": frame_id,
        "timestamp_seconds": round(timestamp, 6),
        "frame_index": frame_index,
        "description": f"An egg thrown from off-camera falls {place_desc}, while the person's own gloved hands remain mostly at their sides.",
        "resources": limbs(),
        "entities": [
            dict(TABLE_ENTITY),
            {
                "name": "egg",
                "description": f"a brown egg in mid-air {place_desc}, thrown from off-camera and not yet touched by the person",
                "identifier": "falling_egg",
            },
        ],
        "actions": [
            {
                "name": "egg falls",
                "actor": "egg",
                "target": "person",
                "description": f"the egg descends {place_desc} after being thrown from off-camera",
            }
        ],
        "ontological_phrases": [f"egg falls toward person's {place}"],
    }


def clasped_frame(frame_id: str, timestamp: float, frame_index: int) -> dict:
    resources = limbs()
    resources.append(
        {
            "name": "egg",
            "description": "the caught egg, enclosed and hidden from view inside the person's clasped hands",
            "identifier": "caught_egg",
        }
    )
    return {
        "frame_id": frame_id,
        "timestamp_seconds": round(timestamp, 6),
        "frame_index": frame_index,
        "description": "A person in a silver suit stands with both hands clasped together at their waist, looking down; the egg they just caught is enclosed between their hands and not visible.",
        "resources": resources,
        "entities": [dict(TABLE_ENTITY)],
        "actions": [
            {
                "name": "catch egg",
                "actor": "person",
                "target": "egg",
                "description": "the person's hands close around the falling egg at their waist, concealing it from view",
            }
        ],
        "ontological_phrases": ["person catch egg with clasped hands"],
    }


def revealed_frame(frame_id: str, timestamp: float, frame_index: int) -> dict:
    resources = limbs()
    resources.append(
        {
            "name": "egg",
            "description": "a brown egg resting visibly in the person's open right hand palm",
            "identifier": "caught_egg",
        }
    )
    return {
        "frame_id": frame_id,
        "timestamp_seconds": round(timestamp, 6),
        "frame_index": frame_index,
        "description": "A person in a silver suit stands holding a caught brown egg in their open right palm, with the left hand's fingers positioned near it.",
        "resources": resources,
        "entities": [dict(TABLE_ENTITY)],
        "actions": [
            {
                "name": "hold egg",
                "actor": "person",
                "target": "egg",
                "description": "the person holds the caught egg resting in their open right palm",
            }
        ],
        "ontological_phrases": ["person hold egg in open palm"],
    }


def build_frames() -> list[dict]:
    frames = []
    for i in range(TOTAL_FRAMES):
        frame_index = i * STEP
        timestamp = frame_index / SOURCE_FPS
        frame_id = f"{VIDEO_ID}_{frame_index:06d}"
        if frame_index <= 56:
            frames.append(standing_frame(frame_id, timestamp, frame_index))
        elif frame_index in FALL_POSITIONS:
            frames.append(falling_frame(frame_id, timestamp, frame_index))
        elif 88 <= frame_index <= 144:
            frames.append(clasped_frame(frame_id, timestamp, frame_index))
        else:
            frames.append(revealed_frame(frame_id, timestamp, frame_index))
    return frames


def main() -> None:
    annotation = {
        "video_id": VIDEO_ID,
        "video_path": f"videos/{VIDEO_ID}.mov",
        "frames": build_frames(),
    }
    out_path = Path(__file__).resolve().parent.parent / "data" / "egg_catch" / "annotations" / f"{VIDEO_ID}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(annotation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(annotation['frames'])} frame annotations to {out_path}")


if __name__ == "__main__":
    main()
