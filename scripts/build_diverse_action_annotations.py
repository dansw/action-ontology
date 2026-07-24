from __future__ import annotations

import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent / "data" / "diverse_actions"

# Resource granularity: name the smallest sub-part actually bearing load or
# making contact right now (arm > hand > palm/fingers), and omit a limb
# entirely when it is not engaged in the task -- do not default to
# "hand"/"arm" for an idle limb just because it exists. When two actors share
# one action, give each actor's own engaged sub-part explicitly rather than
# collapsing to a single actor.


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
                entities=[
                    {"name": "doorway", "description": "open glass doorway the person walks through"},
                    {"name": "dog", "description": "golden retriever watching from the sofa"},
                ],
                actions=[
                    {
                        "name": "walk",
                        "actor": "person",
                        "target": "doorway",
                        "description": "the person walks through the doorway into the room; this is locomotion, no limb is gripping or bearing an object",
                    }
                ],
            ),
            frame(
                813,
                60.0,
                "A person in a silver suit and a woman stand at a round marble table near the entryway, fingertips of both people resting lightly on its edge.",
                resources=[
                    {"name": "right fingers", "description": "the person's right fingertips resting lightly on the table's edge, not yet gripping to lift", "identifier": "person_right_fingers"},
                    {"name": "right fingers", "description": "the woman's right fingertips resting lightly on the table's surface", "identifier": "woman_right_fingers"},
                ],
                entities=[
                    {"name": "table", "description": "round white marble coffee table", "identifier": "coffee_table"},
                    {"name": "woman", "description": "a second person assisting with the task"},
                    {"name": "dog", "description": "golden retriever watching from the sofa"},
                ],
                actions=[
                    {"name": "rest fingertips on table", "actor": "person's right fingers", "target": "table", "description": "the person's fingertips rest lightly on the table's edge"},
                    {"name": "rest fingertips on table", "actor": "woman's right fingers", "target": "table", "description": "the woman's fingertips rest lightly on the table's surface"},
                ],
            ),
            frame(
                1236,
                60.0,
                "A person in a silver suit carries a round marble table across the room, their right fingers hooked under its edge bearing its weight, toward a woman standing near the door as a dog watches from the couch.",
                resources=[
                    {"name": "right fingers", "description": "hooked under the table's underside edge, bearing the table's weight while carrying it", "identifier": "person_right_fingers"},
                ],
                entities=[
                    {"name": "table", "identifier": "coffee_table"},
                    {"name": "woman", "description": "a second person assisting with the task"},
                    {"name": "dog", "description": "golden retriever watching from the couch"},
                ],
                actions=[
                    {"name": "carry table", "actor": "person's right fingers", "target": "table", "description": "the person's fingers, hooked under the table's edge, bear its weight while carrying it across the room"}
                ],
            ),
            frame(
                1368,
                60.0,
                "A person in a silver suit and a woman jointly carry a round marble table: the person's right fingers hook under its underside edge bearing its weight, while the woman's right hand rests flat on its top surface guiding it.",
                resources=[
                    {"name": "right fingers", "description": "hooked under the table's underside edge, bearing its weight", "identifier": "person_right_fingers"},
                    {"name": "right hand", "description": "resting flat, palm and fingers spread, on the table's top surface -- guiding it rather than bearing its weight", "identifier": "woman_right_hand"},
                ],
                entities=[
                    {"name": "table", "identifier": "coffee_table"},
                    {"name": "woman", "description": "a second person assisting with the task"},
                    {"name": "dog", "description": "golden retriever visible nearby"},
                ],
                actions=[
                    {
                        "name": "carry table",
                        "actor": "person's right fingers and woman's right hand",
                        "target": "table",
                        "description": "the person and woman carry the table together: the person's fingers grip its underside edge and bear its weight, while the woman's hand rests on top guiding its position -- this is one shared action, not two independent ones",
                    }
                ],
            ),
            frame(
                1552,
                60.0,
                "A woman bends over a round marble table, her right fingers reaching toward its edge but not yet gripping it, while a dog rests on the couch in the background.",
                resources=[
                    {"name": "right fingers", "description": "reaching toward the table's edge, not yet in contact with it", "identifier": "woman_right_fingers"},
                ],
                entities=[
                    {"name": "table", "identifier": "coffee_table"},
                    {"name": "woman", "description": "a second person assisting with the task"},
                    {"name": "dog", "description": "golden retriever resting on the couch"},
                ],
                actions=[
                    {"name": "reach for table", "actor": "woman's right fingers", "target": "table", "description": "the woman's fingers reach toward the table's edge"}
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
                "A person in a silver suit kneels at the edge of the bed, their right fingers clutching a bunched handful of a large tan sheet and pulling it over the mattress.",
                resources=[
                    {"name": "right fingers", "description": "curled into a clutching grip around a bunched handful of the sheet fabric", "identifier": "right_fingers"},
                ],
                entities=[{"name": "bed"}, {"name": "sheet", "description": "large tan sheet being pulled over the mattress"}],
                actions=[
                    {"name": "pull sheet", "actor": "right fingers", "target": "sheet", "description": "the fingers clutch a handful of the sheet and pull it across the mattress"}
                ],
            ),
            frame(
                6977,
                60.0,
                "A person in a silver suit stands still beside the bed, facing away from the camera, both hands relaxed at their sides and not engaged with anything.",
                resources=[],
                entities=[{"name": "bed"}],
                actions=[{"name": "stand", "actor": "person", "target": "bed", "description": "the person stands beside the bed; no limb is currently engaged with an object"}],
            ),
            frame(
                8962,
                60.0,
                "A person in a silver suit is largely obscured by a white sheet hanging in the foreground; their right fingers grip its edge, holding it up near the bed.",
                resources=[
                    {"name": "right fingers", "description": "gripping the edge of the white sheet, holding it up", "identifier": "right_fingers"},
                ],
                entities=[{"name": "bed"}, {"name": "sheet", "description": "white sheet hanging in the foreground"}],
                actions=[
                    {
                        "name": "hold sheet",
                        "actor": "right fingers",
                        "target": "sheet",
                        "description": "the fingers grip the sheet's edge, holding it up near the bed while the person is partially hidden behind it",
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
                "A person in a silver suit stands beside an unmade bed with rumpled bedding, at the start of a bed-making task, both hands relaxed and not yet engaged.",
                resources=[],
                entities=[{"name": "bed", "description": "unmade bed with rumpled bedding"}],
                actions=[{"name": "stand", "actor": "person", "target": "bed", "description": "the person stands beside the unmade bed; no limb is engaged with the bedding yet"}],
            ),
            frame(
                3535,
                60.0,
                "A person in a silver suit bends over near the headboard, both hands' fingers gripping and tucking the smoothed white duvet into place among the pillows.",
                resources=[
                    {"name": "left fingers", "description": "gripping and tucking the duvet near the headboard", "identifier": "left_fingers"},
                    {"name": "right fingers", "description": "gripping and tucking the duvet near the headboard", "identifier": "right_fingers"},
                ],
                entities=[{"name": "bed", "description": "bed with a smoothed white duvet and two pillows"}],
                actions=[
                    {
                        "name": "tuck bedding",
                        "actor": "left fingers and right fingers",
                        "target": "bedding",
                        "description": "both hands' fingers grip and tuck the duvet into place near the headboard",
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
                "A person in a silver suit walks near the window in the bedroom, both hands' fingers wrapped around a yellow pillow held against their body; the bed is neatly made with a different set of yellow and blue pillows than before.",
                resources=[
                    {"name": "left fingers", "description": "wrapped around the yellow pillow, gripping it against the body", "identifier": "left_fingers"},
                    {"name": "right fingers", "description": "wrapped around the yellow pillow, gripping it against the body", "identifier": "right_fingers"},
                ],
                entities=[{"name": "bed", "description": "neatly made bed, now with yellow and blue pillows"}],
                actions=[
                    {"name": "carry pillow", "actor": "left fingers and right fingers", "target": "pillow", "description": "both hands' fingers grip the pillow while walking near the window"}
                ],
            ),
        ],
    },
    "yo_yo": {
        "source_fps": 60.0,
        "frames": [
            frame(
                0,
                60.0,
                "A person in a silver suit stands with their right fingers pinching the yo-yo's string, letting the yo-yo hang near their foot; the left hand is relaxed and not engaged.",
                resources=[
                    {"name": "right fingers", "description": "pinching the yo-yo's string, suspending it", "identifier": "right_fingers"},
                    {"name": "yo-yo", "description": "red and white yo-yo hanging on its string near the floor"},
                ],
                actions=[
                    {
                        "name": "hold yo-yo string",
                        "actor": "right fingers",
                        "target": "yo-yo",
                        "description": "the fingers pinch the yo-yo's string, letting it hang near the floor",
                    }
                ],
            ),
            frame(
                180,
                60.0,
                "A person in a silver suit holds the yo-yo close to their chest, both hands' fingers winding its string around the axle.",
                resources=[
                    {"name": "left fingers", "description": "winding the yo-yo's string around its axle", "identifier": "left_fingers"},
                    {"name": "right fingers", "description": "winding the yo-yo's string around its axle", "identifier": "right_fingers"},
                    {"name": "yo-yo", "description": "red and white yo-yo held close to the chest"},
                ],
                actions=[
                    {
                        "name": "wind yo-yo string",
                        "actor": "left fingers and right fingers",
                        "target": "yo-yo",
                        "description": "both hands' fingers wind the yo-yo's string around its axle",
                    }
                ],
            ),
            frame(
                368,
                60.0,
                "A person in a silver suit continues winding the yo-yo's string with both hands' fingers held close to their chest.",
                resources=[
                    {"name": "left fingers", "description": "winding the yo-yo's string around its axle", "identifier": "left_fingers"},
                    {"name": "right fingers", "description": "winding the yo-yo's string around its axle", "identifier": "right_fingers"},
                    {"name": "yo-yo", "description": "red and white yo-yo held close to the chest"},
                ],
                actions=[
                    {
                        "name": "wind yo-yo string",
                        "actor": "left fingers and right fingers",
                        "target": "yo-yo",
                        "description": "both hands' fingers continue winding the yo-yo's string",
                    }
                ],
            ),
            frame(
                550,
                60.0,
                "A person in a silver suit holds the yo-yo with both hands' fingers, still winding its string.",
                resources=[
                    {"name": "left fingers", "description": "winding the yo-yo's string around its axle", "identifier": "left_fingers"},
                    {"name": "right fingers", "description": "winding the yo-yo's string around its axle", "identifier": "right_fingers"},
                    {"name": "yo-yo", "description": "red and white yo-yo held close to the chest"},
                ],
                actions=[
                    {
                        "name": "wind yo-yo string",
                        "actor": "left fingers and right fingers",
                        "target": "yo-yo",
                        "description": "both hands' fingers wind the yo-yo's string",
                    }
                ],
            ),
            frame(
                674,
                60.0,
                "A person in a silver suit holds the yo-yo up near their chest, right fingers wrapped around its body, with the left hand relaxed and not engaged.",
                resources=[
                    {"name": "right fingers", "description": "wrapped around the yo-yo's body, holding it up near the chest", "identifier": "right_fingers"},
                    {"name": "yo-yo", "description": "red and white yo-yo held in one hand"},
                ],
                actions=[
                    {"name": "hold yo-yo", "actor": "right fingers", "target": "yo-yo", "description": "the fingers wrap around the yo-yo's body, holding it near the chest"}
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
                "A person in a silver suit stands beside a silver minivan in a driveway, facing the vehicle, both hands relaxed and not yet touching it.",
                resources=[],
                entities=[{"name": "minivan", "description": "silver minivan parked in the driveway", "identifier": "minivan"}],
                actions=[
                    {
                        "name": "approach minivan",
                        "actor": "person",
                        "target": "minivan",
                        "description": "the person stands facing the minivan, about to enter it; no limb is in contact with it yet",
                    }
                ],
            ),
            frame(
                238,
                60.0,
                "A person in a silver suit steps into the open driver-side door of the silver minivan.",
                resources=[],
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
                "A person in a silver suit sits in the driver's seat of the silver minivan, both hands' fingers gripping the seatbelt strap and pulling it across their chest, while a dog sits beside them.",
                resources=[
                    {"name": "left fingers", "description": "gripping the seatbelt strap higher up, near the shoulder", "identifier": "left_fingers"},
                    {"name": "right fingers", "description": "gripping the seatbelt strap near the buckle, guiding it into place", "identifier": "right_fingers"},
                    {"name": "seatbelt", "description": "seatbelt strap being pulled across the chest and buckled"},
                ],
                entities=[
                    {"name": "minivan", "identifier": "minivan"},
                    {"name": "dog", "description": "golden retriever sitting in the passenger seat"},
                ],
                actions=[
                    {
                        "name": "buckle seatbelt",
                        "actor": "left fingers and right fingers",
                        "target": "seatbelt",
                        "description": "both hands' fingers grip the seatbelt strap, pulling it across the chest and guiding it toward the buckle",
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
