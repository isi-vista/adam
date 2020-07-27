from adam.ontology.phase1_ontology import (
    BABY,
    DAD,
    GAILA_PHASE_1_ONTOLOGY,
    HAT,
    IS_HUMAN,
    MOM,
    PERSON,
    THING,
)
from immutablecollections import immutableset


def test_descends_from():
    # Test if we can correctly check when one node descends from another
    # Not a descendant of self
    assert not GAILA_PHASE_1_ONTOLOGY.descends_from(MOM, MOM)
    # Immediate descendant
    assert GAILA_PHASE_1_ONTOLOGY.descends_from(PERSON, THING)
    # Second-order descendant
    assert GAILA_PHASE_1_ONTOLOGY.descends_from(MOM, THING)

    # Test if we can correctly check when one descends from a set
    assert GAILA_PHASE_1_ONTOLOGY.descends_from(MOM, immutableset([HAT, PERSON]))
    assert GAILA_PHASE_1_ONTOLOGY.descends_from(MOM, immutableset([THING, PERSON]))
    assert not GAILA_PHASE_1_ONTOLOGY.descends_from(MOM, immutableset([HAT]))
    assert not GAILA_PHASE_1_ONTOLOGY.descends_from(MOM, immutableset([DAD]))

    # Should never be true when query_ancestors is the empty set
    assert not GAILA_PHASE_1_ONTOLOGY.descends_from(MOM, immutableset())


def test_nodes_with_banned_ontology_types():
    nodes = GAILA_PHASE_1_ONTOLOGY.nodes_with_properties(
        THING, [IS_HUMAN], banned_ontology_types=[DAD]
    )
    assert DAD not in nodes
    assert MOM in nodes
    assert BABY in nodes
