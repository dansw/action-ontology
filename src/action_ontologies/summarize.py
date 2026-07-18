from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import ActionElement, OntologyElement, _dedupe_actions, _dedupe_elements


def summarize_ontology(data: dict[str, Any]) -> dict[str, Any]:
    frames = data.get("frames", [])
    resources: list[OntologyElement] = []
    entities: list[OntologyElement] = []
    actions: list[ActionElement] = []
    for frame in frames:
        resources.extend(OntologyElement.from_value(item) for item in frame.get("resources", []) or [])
        entities.extend(OntologyElement.from_value(item) for item in frame.get("entities", []) or [])
        actions.extend(ActionElement.from_value(item) for item in frame.get("actions", []) or [])
    return {
        "video_path": data.get("video_path"),
        "frame_count": len(frames),
        "resources": [item.to_dict() for item in _dedupe_elements(resources)],
        "entities": [item.to_dict() for item in _dedupe_elements(entities)],
        "actions": [item.to_dict() for item in _dedupe_actions(actions)],
    }


def summarize_file(input_path: str | Path) -> dict[str, Any]:
    with open(input_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return summarize_ontology(data)
