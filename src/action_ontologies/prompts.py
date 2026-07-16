from __future__ import annotations

from textwrap import dedent


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


def frame_prompt(frame_id: str, timestamp_seconds: float) -> str:
    return dedent(
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

