from immutablecollections import immutableset

from adam.ontology.phase1_ontology import (
    AGENT,
    FALL,
    GAILA_PHASE_1_ONTOLOGY,
    GOAL,
    THEME,
    THROW,
)
from adam.ontology.selectors import SubcategorizationSelector


def test_subcategorization_selector():
    query_frame = immutableset([AGENT, THEME, GOAL])
    selector = SubcategorizationSelector(query_frame)
    matching_actions = selector.select_nodes(GAILA_PHASE_1_ONTOLOGY)
    assert THROW in matching_actions
    assert FALL not in matching_actions
