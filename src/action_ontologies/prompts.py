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
    participate in the action. Include object parts when they are meaningful to
    the action, such as fingers, hand, pan handle, or individual piano keys.
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
        "unchanged from the last entry."
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

