from action_ontologies.infer import _canonicalize_known_identifiers
from action_ontologies.schema import FrameOntology, OntologyElement


def _entity_ontology(frame_id: str, timestamp: float, *pairs: tuple[str, str]) -> FrameOntology:
    return FrameOntology(
        frame_id=frame_id,
        timestamp_seconds=timestamp,
        entities=[OntologyElement(name=name, identifier=identifier) for name, identifier in pairs],
    )


def test_canonicalize_known_identifiers_registers_new_identifier():
    known: dict[str, str] = {}
    identifier_names: dict[str, set] = {}
    ontology = _entity_ontology("f0", 0.0, ("duvet fabric", "bedding"))

    result = _canonicalize_known_identifiers(known, identifier_names, ontology)

    assert known == {"bedding": "duvet fabric"}
    assert result.entities[0].identifier == "bedding"


def test_canonicalize_known_identifiers_rewrites_exact_name_duplicate_across_frames():
    known = {"bedding": "duvet fabric"}
    identifier_names = {"bedding": {frozenset({"duvet", "fabric"})}}
    ontology = _entity_ontology("f1", 1.0, ("duvet fabric", "fabric"))

    result = _canonicalize_known_identifiers(known, identifier_names, ontology)

    assert result.entities[0].identifier == "bedding"
    assert known == {"bedding": "duvet fabric"}
    assert "fabric" not in known


def test_canonicalize_known_identifiers_rewrites_fuzzy_duplicate_across_frames():
    # "duvet" is a whole-word subset of the already-registered "duvet fabric"
    # -- this is the exact drift seen on duvet_cover.mov, where the model
    # introduced a brand-new identifier "duvet" for what was already tracked
    # as "bedding" (displayed at various points as "duvet fabric").
    known = {"bedding": "bedding"}
    identifier_names = {"bedding": {frozenset({"bedding"}), frozenset({"duvet", "fabric"})}}
    ontology = _entity_ontology("f2", 2.0, ("duvet", "duvet"))

    result = _canonicalize_known_identifiers(known, identifier_names, ontology)

    assert result.entities[0].identifier == "bedding"
    assert "duvet" not in known


def test_canonicalize_known_identifiers_collapses_same_frame_duplicate():
    known: dict[str, str] = {}
    identifier_names: dict[str, set] = {}
    # "duvet" and "duvet fabric" both appear as separate entities within the
    # SAME frame -- this must collapse to a single entity, not two entries
    # that merely share an identifier.
    ontology = _entity_ontology("f0", 0.0, ("duvet fabric", "bedding"), ("duvet", "duvet"))

    result = _canonicalize_known_identifiers(known, identifier_names, ontology)

    assert len(result.entities) == 1
    assert result.entities[0].identifier == "bedding"


def test_canonicalize_known_identifiers_leaves_distinct_names_untouched():
    known = {"bedding": "duvet fabric"}
    identifier_names = {"bedding": {frozenset({"duvet", "fabric"})}}
    ontology = _entity_ontology("f2", 2.0, ("mattress", "mattress"))

    result = _canonicalize_known_identifiers(known, identifier_names, ontology)

    assert result.entities[0].identifier == "mattress"
    assert known == {"bedding": "duvet fabric", "mattress": "mattress"}


def test_canonicalize_known_identifiers_does_not_merge_state_change_split():
    # "wrapper fragment" must NOT be folded back onto "bar" (granola bar)
    # just because it appeared alongside it once the wrapper tore off -- the
    # two share no whole-word overlap, so a deliberate state-change
    # re-identification is preserved rather than erased.
    known = {"bar": "granola bar"}
    identifier_names = {"bar": {frozenset({"granola", "bar"})}}
    ontology = _entity_ontology("f3", 3.0, ("granola bar", "bar"), ("wrapper fragment", "wrapper_fragment"))

    result = _canonicalize_known_identifiers(known, identifier_names, ontology)

    identifiers = {entity.identifier for entity in result.entities}
    assert identifiers == {"bar", "wrapper_fragment"}


def test_canonicalize_known_identifiers_evicts_oldest_beyond_cap():
    known: dict[str, str] = {}
    identifier_names: dict[str, set] = {}
    ontology = _entity_ontology("f0", 0.0, ("first item", "first_item"))
    _canonicalize_known_identifiers(known, identifier_names, ontology)

    for index in range(30):
        ontology = _entity_ontology(f"f{index + 1}", float(index + 1), (f"item {index}", f"item_{index}"))
        _canonicalize_known_identifiers(known, identifier_names, ontology)

    assert "first_item" not in known
    assert "first_item" not in identifier_names
    assert len(known) == 30
