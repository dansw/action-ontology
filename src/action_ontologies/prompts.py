from __future__ import annotations

from textwrap import dedent
from typing import Any


SYSTEM_PROMPT = dedent(
    """
    You extract task ontologies from video frames.

    Return only valid JSON. Do not include markdown.

    Definitions:
    - Entity: any separable object or object part that matters to the visible task.
    - Resource: an entity controlled by an autonomous mover, including body parts
      and held objects or materials.
    - Action: a meaningful interaction, task motion, or state change involving
      resources and entities.

    Focus on the main visible activity. Ignore background objects unless they
    participate in the action.

    Resource granularity: for each resource, name the SMALLEST sub-part that is
    mechanically doing the work right now, not the whole limb. Body parts nest
    (arm > hand > fingers > fingernails), and you should descend that hierarchy
    only as far as the part actually bearing load or making contact -- and no
    further than that. A held object's weight is normally carried by fingers,
    not the whole hand; a hand only becomes the active resource when the palm
    itself is bearing load (e.g. a flat push) rather than a grip; an elbow
    becomes the active resource when it is the part making contact (e.g.
    nudging a door open) instead of the hand or fingers; a fingernail is a real
    sub-part of a finger but is essentially never the part actually doing the
    work, so do not list it. If two limbs are doing different things in the
    same frame, name each at its own correct granularity rather than defaulting
    both to "hand": e.g. "left fingers grip the table's edge, right fingers
    turn the door handle" -- not "left hand" and "right hand" -- and if a limb
    is not currently engaged in the task, omit it rather than listing it by
    default. When several people share one action (e.g. two people carrying
    one table together), give both actors' engaged sub-parts explicitly rather
    than only naming one of them. Re-evaluate each limb's engagement from
    scratch on every frame: a hand that was gripping a moment ago can release
    and go idle (resting flat on a surface, hanging at the side) while the
    other hand keeps working, and the two hands do not have to change state
    together. Do not default to a symmetric "both hands engaged the same way"
    description out of habit -- check each limb's own current position and
    contact independently, even if the wording you have been using for
    several consecutive frames was symmetric.

    Track state changes by naming the new entity state, such as broken egg.
    """
).strip()


def _format_history(history: list[dict[str, Any]] | None) -> str:
    if not history:
        return ""
    lines = [
        "Recent frame history, oldest to newest (the last entry is the moment "
        "immediately before the current frame; weight it most heavily):"
    ]
    for entry in history:
        timestamp_seconds = entry.get("timestamp_seconds", 0.0)
        description = entry.get("description", "")
        actions = entry.get("actions") or []
        actions_suffix = f" (actions: {', '.join(actions)})" if actions else ""
        lines.append(f'- t={timestamp_seconds:.3f}s: "{description}"{actions_suffix}')
    lines.append("")
    lines.append(
        "Use this history only to judge PROGRESS STATE -- for example, do not "
        'describe an action as "about to start" or "preparing to" if the history '
        "already shows it in progress or finished, and do not describe a finished "
        "action as still in progress. Do not simply repeat or reword the history's "
        "wording: look at the current image and describe what is actually visible "
        "in it now, including any new specific detail (objects, hand or body "
        "position, tools in use) even if the overall action's progress state is "
        "unchanged from the last entry.\n\n"
        "The history is a record of PAST appearance, not a prediction of the "
        "current frame: an entity's own visible state can change between frames "
        "(a wrapper tearing open, a package emptying, a surface going from messy "
        "to clean) even when the history repeatedly described an earlier state. "
        "If the current image shows an entity in a different physical state than "
        "the history's most recent entry -- more removed, opened, exposed, "
        "assembled, or disassembled than before -- you MUST describe that new "
        "state instead of carrying the old one forward. Trust the pixels in "
        "front of you over the pattern in the history text."
    )
    return "\n".join(lines) + "\n\n"


def frame_prompt(frame_id: str, timestamp_seconds: float, history: list[dict[str, Any]] | None = None) -> str:
    body = dedent(
        f"""
        Analyze this video frame.

        Frame id: {frame_id}
        Timestamp seconds: {timestamp_seconds:.3f}

        Output this exact JSON shape:
        {{
          "frame_id": "{frame_id}",
          "timestamp_seconds": {timestamp_seconds:.3f},
          "description": "concise description of the main visible activity",
          "resources": [
            {{"name": "resource name", "identifier": "optional identifier", "description": "why it is a resource"}}
          ],
          "entities": [
            {{"name": "entity name", "identifier": "optional identifier", "description": "visible role or state"}}
          ],
          "actions": [
            {{"name": "action name", "actor": "resource or entity doing it", "target": "affected entity", "description": "interaction"}}
          ],
          "ontological_phrases": [
            "actor action target"
          ]
        }}
        """
    ).strip()
    return _format_history(history) + body

