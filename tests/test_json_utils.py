import pytest

from action_ontologies.json_utils import extract_json_object


def test_extracts_plain_json_object():
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_extracts_json_from_markdown_fence():
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}


def test_extracts_json_with_surrounding_text():
    assert extract_json_object('result:\n{"a": 1}\ndone') == {"a": 1}


def test_rejects_non_object_json():
    with pytest.raises(ValueError):
        extract_json_object("[1, 2]")

