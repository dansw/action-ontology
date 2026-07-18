from __future__ import annotations

import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent / "data" / "diverse_actions"

LEFT_HAND = {"name": "left hand", "description": "the person's left gloved hand"}
RIGHT_HAND = {"name": "right hand", "description": "the person's right gloved hand"}
LEFT_ARM = {"name": "left arm", "description": "the person's left arm"}
RIGHT_ARM = {"name": "right arm", "description": "the person's right arm"}


def limbs() -> list[dict]:
    return [dict(LEFT_HAND), dict(RIGHT_HAND), dict(LEFT_ARM), dict(RIGHT_ARM)]


def frame(frame_index: int, source_fps: float, description: str, *, resources=None, entities=None, actions=None) -> dict:
    return {
        "frame_id": f"{{video_id}}_{frame_index:06d}",
        "frame_index": frame_index,
        "timestamp_seconds": round(frame_index / source_fps, 6),
        "description": description,
        "resources": resources or [],
        "entities": entities or [],
        "actions": actions or [],
        "ontological_phrases": [],
    }


VIDEOS = {
    "carry_coffee_table": {
        "source_fps": 60.0,
        "frames": [
            frame(
                0,
                60.0,
                "A dog rests on a sofa in a living room; no person is yet visible and no task action has started.",
                entities=[
                    {"name": "dog", "description": "golden retriever resting on the sofa"},
                    {"name": "sofa", "description": "gray sofa the dog rests on"},
                ],
            ),
            frame(
                600,
                60.0,
                "A person in a silver suit walks through an open doorway into the room, backlit by daylight outside.",
                resources=limbs(),
                entities=[
                    {"name": "doorway", "description": "open glass doorway the person walks through"},
                    {"name": "dog", "description": "golden retriever watching from the sofa"},
                ],
                actions=[
                    {
                        "name": "walk",
                        "actor": "person",
                        "target": "doorway",
                        "description": "the person walks through the doorway into the room",
                    }
                ],
            ),
            frame(
                813,
                60.0,
                "A person in a silver suit and a woman stand at a round marble table near the entryway, the person's hands resting near its edge.",
                resources=limbs(),
                entities=[
                    {"name": "table", "description": "round white marble coffee table", "identifier": "coffee_table"},
                    {"name": "woman", "description": "a second person assisting with the task"},
                    {"name": "dog", "description": "golden retriever watching from the sofa"},
                ],
                actions=[
                    {"name": "stand at table", "actor": "person", "target": "table", "description": "the person stands at the table"},
                    {"name": "stand at table", "actor": "woman", "target": "table", "description": "the woman stands at the table"},
                ],
            ),
            frame(
                1236,
                60.0,
                "A person in a silver suit carries a round marble table across the room toward a woman standing near the door, with a dog watching from the couch.",
                resources=limbs(),
                entities=[
                    {"name": "table", "identifier": "coffee_table"},
                    {"name": "woman", "description": "a second person assisting with the task"},
                    {"name": "dog", "description": "golden retriever watching from the couch"},
                ],
                actions=[
                    {"name": "carry table", "actor": "person", "target": "table", "description": "the person carries the table across the room"}
                ],
            ),
            frame(
                1368,
                60.0,
                "A person in a silver suit and a woman stand on either side of a round marble table, both with hands near its surface, positioning it together.",
                resources=limbs(),
                entities=[
                    {"name": "table", "identifier": "coffee_table"},
                    {"name": "woman", "description": "a second person assisting with the task"},
                    {"name": "dog", "description": "golden retriever visible nearby"},
                ],
                actions=[
                    {"name": "position table", "actor": "person", "target": "table", "description": "the person helps position the table"},
                    {"name": "position table", "actor": "woman", "target": "table", "description": "the woman helps position the table"},
                ],
            ),
            frame(
                1552,
                60.0,
                "A woman bends over a round marble table, reaching toward it, while a dog rests on the couch in the background.",
                entities=[
                    {"name": "table", "identifier": "coffee_table"},
                    {"name": "woman", "description": "a second person assisting with the task"},
                    {"name": "dog", "description": "golden retriever resting on the couch"},
                ],
                actions=[
                    {"name": "reach for table", "actor": "woman", "target": "table", "description": "the woman reaches toward the table"}
                ],
            ),
        ],
    },
    "duvet_cover": {
        "source_fps": 60.0,
        "frames": [
            frame(
                0,
                60.0,
                "An empty bedroom with an unmade bed; no person is yet visible and no task action has started.",
                entities=[{"name": "bed", "description": "unmade bed with rumpled gray bedding"}],
            ),
            frame(
                2218,
                60.0,
                "A person in a silver suit kneels at the edge of the bed, pulling a large tan sheet over the mattress with one arm extended.",
                resources=limbs(),
                entities=[{"name": "bed"}, {"name": "sheet", "description": "large tan sheet being pulled over the mattress"}],
                actions=[
                    {"name": "pull sheet", "actor": "person", "target": "sheet", "description": "the person pulls the sheet across the mattress"}
                ],
            ),
            frame(
                6977,
                60.0,
                "A person in a silver suit stands still beside the bed, facing away from the camera.",
                resources=limbs(),
                entities=[{"name": "bed"}],
                actions=[{"name": "stand", "actor": "person", "target": "bed", "description": "the person stands beside the bed"}],
            ),
            frame(
                8962,
                60.0,
                "A person in a silver suit is largely obscured by a white sheet hanging in the foreground, standing near the bed.",
                resources=limbs(),
                entities=[{"name": "bed"}, {"name": "sheet", "description": "white sheet hanging in the foreground"}],
                actions=[
                    {
                        "name": "handle sheet",
                        "actor": "person",
                        "target": "sheet",
                        "description": "the person handles a sheet near the bed, partially hidden from view",
                    }
                ],
            ),
        ],
    },
    "make_a_bed": {
        "source_fps": 60.0,
        "frames": [
            frame(
                0,
                60.0,
                "A person in a silver suit stands beside an unmade bed with rumpled bedding, at the start of a bed-making task.",
                resources=limbs(),
                entities=[{"name": "bed", "description": "unmade bed with rumpled bedding"}],
                actions=[{"name": "stand", "actor": "person", "target": "bed", "description": "the person stands beside the unmade bed"}],
            ),
            frame(
                3535,
                60.0,
                "A person in a silver suit bends over near the headboard, adjusting the smoothed white duvet and pillows on the bed.",
                resources=limbs(),
                entities=[{"name": "bed", "description": "bed with a smoothed white duvet and two pillows"}],
                actions=[
                    {
                        "name": "adjust bedding",
                        "actor": "person",
                        "target": "bedding",
                        "description": "the person adjusts the duvet and pillows near the headboard",
                    }
                ],
            ),
            frame(
                4960,
                60.0,
                "The bed is neatly made with a white duvet, a brown runner across the middle, and arranged pillows; no person is visible in the room.",
                entities=[{"name": "bed", "description": "neatly made bed with white duvet and brown runner"}],
            ),
            frame(
                6121,
                60.0,
                "A person in a silver suit walks near the window in the bedroom; the bed is neatly made with a different set of yellow and blue pillows than before.",
                resources=limbs(),
                entities=[{"name": "bed", "description": "neatly made bed, now with yellow and blue pillows"}],
                actions=[{"name": "walk", "actor": "person", "target": "bedroom", "description": "the person walks near the window"}],
            ),
        ],
    },
    "yo_yo": {
        "source_fps": 60.0,
        "frames": [
            frame(
                0,
                60.0,
                "A person in a silver suit stands holding a yo-yo string in their right hand, with the yo-yo hanging near their foot.",
                resources=limbs() + [{"name": "yo-yo", "description": "red and white yo-yo hanging on its string near the floor"}],
                actions=[
                    {
                        "name": "hold yo-yo string",
                        "actor": "person",
                        "target": "yo-yo",
                        "description": "the person holds the yo-yo by its string, letting it hang near the floor",
                    }
                ],
            ),
            frame(
                180,
                60.0,
                "A person in a silver suit holds the yo-yo close to their chest with both hands, winding its string.",
                resources=limbs() + [{"name": "yo-yo", "description": "red and white yo-yo held close to the chest"}],
                actions=[
                    {
                        "name": "wind yo-yo string",
                        "actor": "person",
                        "target": "yo-yo",
                        "description": "the person winds the yo-yo's string using both hands",
                    }
                ],
            ),
            frame(
                368,
                60.0,
                "A person in a silver suit continues winding the yo-yo's string with both hands held close to their chest.",
                resources=limbs() + [{"name": "yo-yo", "description": "red and white yo-yo held close to the chest"}],
                actions=[
                    {
                        "name": "wind yo-yo string",
                        "actor": "person",
                        "target": "yo-yo",
                        "description": "the person winds the yo-yo's string with both hands",
                    }
                ],
            ),
            frame(
                550,
                60.0,
                "A person in a silver suit holds the yo-yo with both hands, still winding its string.",
                resources=limbs() + [{"name": "yo-yo", "description": "red and white yo-yo held close to the chest"}],
                actions=[
                    {
                        "name": "wind yo-yo string",
                        "actor": "person",
                        "target": "yo-yo",
                        "description": "the person winds the yo-yo's string with both hands",
                    }
                ],
            ),
            frame(
                674,
                60.0,
                "A person in a silver suit holds the yo-yo up near their chest in one hand, with the other hand relaxed at their side.",
                resources=limbs() + [{"name": "yo-yo", "description": "red and white yo-yo held in one hand"}],
                actions=[
                    {"name": "hold yo-yo", "actor": "person", "target": "yo-yo", "description": "the person holds the yo-yo in one hand"}
                ],
            ),
        ],
    },
    "get_into_car": {
        "source_fps": 60.0,
        "frames": [
            frame(
                0,
                60.0,
                "A person in a silver suit stands beside a silver minivan in a driveway, facing the vehicle.",
                resources=limbs(),
                entities=[{"name": "minivan", "description": "silver minivan parked in the driveway", "identifier": "minivan"}],
                actions=[
                    {
                        "name": "approach minivan",
                        "actor": "person",
                        "target": "minivan",
                        "description": "the person stands facing the minivan, about to enter it",
                    }
                ],
            ),
            frame(
                238,
                60.0,
                "A person in a silver suit steps into the open driver-side door of the silver minivan.",
                resources=limbs(),
                entities=[{"name": "minivan", "identifier": "minivan"}],
                actions=[
                    {
                        "name": "enter minivan",
                        "actor": "person",
                        "target": "minivan",
                        "description": "the person steps into the minivan through the open driver-side door",
                    }
                ],
            ),
            frame(
                420,
                60.0,
                "A dog sits inside the silver minivan, visible through the windshield.",
                entities=[
                    {"name": "minivan", "identifier": "minivan"},
                    {"name": "dog", "description": "golden retriever sitting inside the minivan"},
                ],
                actions=[{"name": "sit in minivan", "actor": "dog", "target": "minivan", "description": "the dog sits inside the minivan"}],
            ),
            frame(
                594,
                60.0,
                "A person in a silver suit sits in the driver's seat of the silver minivan, adjusting their seatbelt with both hands, while a dog sits beside them.",
                resources=limbs() + [{"name": "seatbelt", "description": "seatbelt the person is adjusting"}],
                entities=[
                    {"name": "minivan", "identifier": "minivan"},
                    {"name": "dog", "description": "golden retriever sitting in the passenger seat"},
                ],
                actions=[
                    {
                        "name": "adjust seatbelt",
                        "actor": "person",
                        "target": "seatbelt",
                        "description": "the person adjusts their seatbelt with both hands",
                    }
                ],
            ),
        ],
    },
}


def main() -> None:
    annotations_dir = PROJECT_DIR / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for video_id, spec in VIDEOS.items():
        frames = []
        for f in spec["frames"]:
            f = dict(f)
            f["frame_id"] = f["frame_id"].format(video_id=video_id)
            frames.append(f)
        annotation = {
            "video_id": video_id,
            "video_path": f"videos/{video_id}.mov",
            "frames": frames,
        }
        out_path = annotations_dir / f"{video_id}.json"
        out_path.write_text(json.dumps(annotation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {len(frames)} frame annotations to {out_path}")
        total += len(frames)
    print(f"total: {total} frames across {len(VIDEOS)} videos")


if __name__ == "__main__":
    main()
