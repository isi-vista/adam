import pytest
from more_itertools import first

from adam.ontology import OntologyNode
from adam.ontology.phase1_ontology import (
    BIRD,
    GAILA_PHASE_1_ONTOLOGY,
    GROUND,
    INANIMATE_OBJECT,
    IS_BODY_PART,
    LIQUID,
    TABLE,
)
from tests.perception.perception_graph_test import do_object_on_table_test

OBJECTS_TO_MATCH = GAILA_PHASE_1_ONTOLOGY.nodes_with_properties(
    INANIMATE_OBJECT, banned_properties={LIQUID, IS_BODY_PART}
)

# We use bird and table for testing
def match_object(object_to_match):
    if object_to_match not in [BIRD, TABLE, GROUND]:
        schemata = GAILA_PHASE_1_ONTOLOGY.structural_schemata(object_to_match)
        if len(schemata) == 1:
            assert do_object_on_table_test(object_to_match, first(schemata), BIRD)


@pytest.mark.parametrize("object_to_match", OBJECTS_TO_MATCH)
def test_object_matching(object_to_match: OntologyNode, benchmark):
    benchmark.name = object_to_match.handle
    benchmark.group = f"Object matching"
    benchmark.pedantic(match_object, [object_to_match], iterations=1, rounds=1)
