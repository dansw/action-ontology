from __future__ import annotations

import json
from pathlib import Path

VIDEO_ID = "egg_catch_001"
SOURCE_FPS = 60.0
SAMPLE_FPS = 8.0
STEP = round(SOURCE_FPS / SAMPLE_FPS)
TOTAL_FRAMES = 63

TABLE_ENTITY = {"name": "table", "description": "white counter the person stands behind", "identifier": "white_table"}

# Resource granularity: name the smallest sub-part actually bearing load or
# making contact right now (arm > hand > fingers), and omit a limb entirely
# when it is not engaged in the task -- do not default to "hand"/"arm" for an
# idle limb just because it exists.


def standing_frame(frame_id: str, timestamp: float, frame_index: int) -> dict:
    return {
        "frame_id": frame_id,
        "timestamp_seconds": round(timestamp, 6),
        "frame_index": frame_index,
        "description": "A person in a silver suit stands facing forward behind a table, both arms extended out to the sides with empty gloved hands.",
        "resources": [],
        "entities": [dict(TABLE_ENTITY)],
        "actions": [
            {
                "name": "stand",
                "actor": "person",
                "target": "table",
                "description": "the person stands still behind the table with both arms held open and hands empty; no limb is actively engaged with anything",
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
        "resources": [],
        "entities": [
            dict(TABLE_ENTITY),
            {
                "name": "egg",
                "description": f"a brown egg in mid-air {place_desc}, thrown from off-camera and not yet touched by the person, so no limb is engaged with it yet",
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
    resources = [
        {
            "name": "left fingers",
            "description": "interlocked with the right fingers, gripping the caught egg between both hands",
            "identifier": "left_fingers",
        },
        {
            "name": "right fingers",
            "description": "interlocked with the left fingers, gripping the caught egg between both hands",
            "identifier": "right_fingers",
        },
        {
            "name": "egg",
            "description": "the caught egg, enclosed and hidden from view inside the person's interlocked fingers",
            "identifier": "caught_egg",
        },
    ]
    return {
        "frame_id": frame_id,
        "timestamp_seconds": round(timestamp, 6),
        "frame_index": frame_index,
        "description": "A person in a silver suit stands with both hands clasped together at their waist, looking down; their fingers are interlocked around the egg they just caught, concealing it from view.",
        "resources": resources,
        "entities": [dict(TABLE_ENTITY)],
        "actions": [
            {
                "name": "catch egg",
                "actor": "left fingers and right fingers",
                "target": "egg",
                "description": "the person's fingers interlock and close around the falling egg at their waist, concealing it from view -- the palms are not what is gripping it",
            }
        ],
        "ontological_phrases": ["fingers catch and grip egg"],
    }


def revealed_frame(frame_id: str, timestamp: float, frame_index: int) -> dict:
    resources = [
        {
            "name": "left palm",
            "description": "open and flat, bearing the egg's weight directly on its surface",
            "identifier": "left_palm",
        },
        {
            "name": "left fingers",
            "description": "curled loosely around the sides of the egg for containment, not gripping it tightly",
            "identifier": "left_fingers",
        },
        {
            "name": "right fingers",
            "description": "positioned near the egg, lightly touching but not bearing any of its weight",
            "identifier": "right_fingers",
        },
        {
            "name": "egg",
            "description": "a brown egg resting visibly on the person's open left palm",
            "identifier": "caught_egg",
        },
    ]
    return {
        "frame_id": frame_id,
        "timestamp_seconds": round(timestamp, 6),
        "frame_index": frame_index,
        "description": "A person in a silver suit stands holding a caught brown egg resting on their open left palm, with the left fingers curled around it and the right fingers positioned near it.",
        "resources": resources,
        "entities": [dict(TABLE_ENTITY)],
        "actions": [
            {
                "name": "hold egg",
                "actor": "left palm",
                "target": "egg",
                "description": "the egg rests on the open left palm, which bears its weight, while the left fingers curl around it for containment",
            }
        ],
        "ontological_phrases": ["palm hold egg", "fingers curl around egg"],
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
