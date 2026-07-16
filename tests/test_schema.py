import pytest

from action_ontologies.schema import FrameOntology


def test_frame_ontology_normalizes_and_dedupes_values():
    frame = FrameOntology.from_dict(
        {
            "frame_id": "f1",
            "timestamp_seconds": "1.5",
            "resources": ["left hand", {"name": "Left Hand", "description": "duplicate"}],
            "entities": [{"name": "egg"}, "egg"],
            "actions": [
                {"name": "catch", "actor": "left hand", "target": "egg"},
                {"name": "Catch", "actor": "Left Hand", "target": "Egg"},
            ],
            "ontological_phrases": ["left hand catches egg", "Left Hand Catches Egg"],
        }
    )
    assert len(frame.resources) == 1
    assert len(frame.entities) == 1
    assert len(frame.actions) == 1
    assert len(frame.ontological_phrases) == 1
    assert frame.timestamp_seconds == 1.5


def test_frame_ontology_requires_frame_id():
    with pytest.raises(ValueError):
        FrameOntology.from_dict({"timestamp_seconds": 0})

