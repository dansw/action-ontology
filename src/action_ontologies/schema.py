from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OntologyElement:
    name: str
    description: str = ""
    identifier: str | None = None

    @classmethod
    def from_value(cls, value: Any) -> "OntologyElement":
        if isinstance(value, str):
            return cls(name=value.strip())
        if not isinstance(value, dict):
            raise ValueError(f"expected ontology element object or string, got {type(value).__name__}")
        name = _clean_required_string(value.get("name"), "name")
        return cls(
            name=name,
            description=_clean_optional_string(value.get("description")),
            identifier=_clean_optional_string(value.get("identifier")) or None,
        )

    def to_dict(self) -> dict[str, str]:
        data = {"name": self.name}
        if self.description:
            data["description"] = self.description
        if self.identifier:
            data["identifier"] = self.identifier
        return data


@dataclass(frozen=True)
class ActionElement:
    name: str
    actor: str = ""
    target: str = ""
    description: str = ""

    @classmethod
    def from_value(cls, value: Any) -> "ActionElement":
        if isinstance(value, str):
            return cls(name=value.strip())
        if not isinstance(value, dict):
            raise ValueError(f"expected action object or string, got {type(value).__name__}")
        return cls(
            name=_clean_required_string(value.get("name"), "name"),
            actor=_clean_optional_string(value.get("actor")),
            target=_clean_optional_string(value.get("target")),
            description=_clean_optional_string(value.get("description")),
        )

    def to_dict(self) -> dict[str, str]:
        data = {"name": self.name}
        for key in ("actor", "target", "description"):
            value = getattr(self, key)
            if value:
                data[key] = value
        return data


@dataclass(frozen=True)
class FrameOntology:
    frame_id: str
    timestamp_seconds: float
    description: str = ""
    resources: list[OntologyElement] = field(default_factory=list)
    entities: list[OntologyElement] = field(default_factory=list)
    actions: list[ActionElement] = field(default_factory=list)
    ontological_phrases: list[str] = field(default_factory=list)
    frame_index: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, fallback_frame_id: str = "", fallback_timestamp: float = 0.0) -> "FrameOntology":
        if not isinstance(data, dict):
            raise ValueError("frame ontology must be a JSON object")
        frame_id = _clean_optional_string(data.get("frame_id")) or fallback_frame_id
        if not frame_id:
            raise ValueError("frame_id is required")
        timestamp = data.get("timestamp_seconds", fallback_timestamp)
        try:
            timestamp_seconds = float(timestamp)
        except (TypeError, ValueError) as exc:
            raise ValueError("timestamp_seconds must be numeric") from exc
        frame_index = data.get("frame_index")
        if frame_index is not None:
            try:
                frame_index = int(frame_index)
            except (TypeError, ValueError) as exc:
                raise ValueError("frame_index must be an integer") from exc
        return cls(
            frame_id=frame_id,
            timestamp_seconds=timestamp_seconds,
            frame_index=frame_index,
            description=_clean_optional_string(data.get("description")),
            resources=_dedupe_elements(OntologyElement.from_value(item) for item in _as_list(data.get("resources"))),
            entities=_dedupe_elements(OntologyElement.from_value(item) for item in _as_list(data.get("entities"))),
            actions=_dedupe_actions(ActionElement.from_value(item) for item in _as_list(data.get("actions"))),
            ontological_phrases=_dedupe_strings(_clean_optional_string(item) for item in _as_list(data.get("ontological_phrases"))),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "frame_id": self.frame_id,
            "timestamp_seconds": round(self.timestamp_seconds, 6),
            "description": self.description,
            "resources": [item.to_dict() for item in self.resources],
            "entities": [item.to_dict() for item in self.entities],
            "actions": [item.to_dict() for item in self.actions],
            "ontological_phrases": self.ontological_phrases,
        }
        if self.frame_index is not None:
            data["frame_index"] = self.frame_index
        return data


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _clean_required_string(value: Any, field_name: str) -> str:
    text = _clean_optional_string(value)
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _clean_optional_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _dedupe_strings(values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _dedupe_elements(values: Any) -> list[OntologyElement]:
    seen: set[tuple[str, str | None]] = set()
    result: list[OntologyElement] = []
    for value in values:
        key = (value.name.casefold(), value.identifier.casefold() if value.identifier else None)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _dedupe_actions(values: Any) -> list[ActionElement]:
    seen: set[tuple[str, str, str]] = set()
    result: list[ActionElement] = []
    for value in values:
        key = (value.name.casefold(), value.actor.casefold(), value.target.casefold())
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result

