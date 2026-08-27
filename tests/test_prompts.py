from action_ontologies.prompts import SYSTEM_PROMPT, frame_prompt


def test_system_prompt_defines_ontology_terms():
    assert "Entity" in SYSTEM_PROMPT
    assert "Resource" in SYSTEM_PROMPT
    assert "Action" in SYSTEM_PROMPT


def test_frame_prompt_contains_exact_frame_context():
    prompt = frame_prompt("egg_000001", 0.5)
    assert "egg_000001" in prompt
    assert "0.500" in prompt
    assert "ontological_phrases" in prompt


def test_frame_prompt_without_history_omits_history_section():
    prompt = frame_prompt("egg_000001", 0.5)
    assert "Recent frame history" not in prompt


def test_frame_prompt_with_history_includes_ordered_entries():
    history = [
        {"timestamp_seconds": 0.0, "description": "person picks up pan", "actions": ["pick up pan"]},
        {"timestamp_seconds": 1.0, "description": "person cooks egg in pan", "actions": ["cook egg"]},
    ]
    prompt = frame_prompt("egg_000002", 2.0, history=history)
    assert "Earlier SAMPLED observations" in prompt
    assert "person picks up pan" not in prompt
    assert "person cooks egg in pan" not in prompt
    assert prompt.index("pick up pan") < prompt.index("cook egg")
    assert "cook egg" in prompt
    assert "2.000s ago" in prompt
    assert "1.000s ago" in prompt
    assert "not necessarily adjacent video frames" in prompt
    # the history section must come before the current-frame instructions
    assert prompt.index("Earlier SAMPLED observations") < prompt.index("Analyze this video frame")


def test_frame_prompt_history_empty_list_omits_section():
    prompt = frame_prompt("egg_000001", 0.5, history=[])
    assert "Recent frame history" not in prompt


def test_frame_prompt_without_known_identifiers_omits_section():
    prompt = frame_prompt("egg_000001", 0.5)
    assert "identifiers already used" not in prompt


def test_frame_prompt_with_known_identifiers_lists_them_before_history():
    prompt = frame_prompt(
        "hammer_000010",
        1.0,
        history=[{"timestamp_seconds": 0.0, "description": "person grips hammer", "actions": ["grip hammer"]}],
        known_identifiers={"wood_block": "wooden block", "hammer": "hammer"},
    )
    assert "identifiers already used" in prompt
    assert "wood_block: wooden block" in prompt
    assert "hammer: hammer" in prompt
    assert prompt.index("identifiers already used") < prompt.index("Earlier SAMPLED observations")


def test_frame_prompt_with_empty_known_identifiers_omits_section():
    prompt = frame_prompt("egg_000001", 0.5, known_identifiers={})
    assert "identifiers already used" not in prompt

