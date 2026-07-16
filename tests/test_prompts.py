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

