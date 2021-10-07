from typing import Iterable, Optional

from adam.ontology import OntologyNode
from adam.ontology.phase1_ontology import GAILA_PHASE_1_ONTOLOGY
from adam.perception import ObjectPerception
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.random_utils import RandomChooser
from adam.situation import SituationObject
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    all_possible,
)


def perception_with_handle(
    frame: DevelopmentalPrimitivePerceptionFrame, handle: str
) -> ObjectPerception:
    for object_perception in frame.perceived_objects:
        if object_perception.debug_handle == handle:
            return object_perception
    raise RuntimeError(
        f"Could not find object perception with handle {handle} " f"in {frame}"
    )


def all_possible_test(
    template: Phase1SituationTemplate,
) -> Iterable[HighLevelSemanticsSituation]:
    """
    Shorcut for `all_possible` with the GAILA phase 1 ontology and a `RandomChooser` with seed 0.
    """
    return all_possible(
        template, chooser=RandomChooser.for_seed(0), ontology=GAILA_PHASE_1_ONTOLOGY
    )


def situation_object(
    ontology_node: OntologyNode,
    *,
    debug_handle: Optional[str] = None,
    properties: Iterable[OntologyNode] = tuple(),
) -> SituationObject:
    return SituationObject.instantiate_ontology_node(
        ontology_node=ontology_node,
        ontology=GAILA_PHASE_1_ONTOLOGY,
        debug_handle=debug_handle,
        properties=properties,
    )
