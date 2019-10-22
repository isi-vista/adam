from typing import Iterable

from more_itertools import first

from adam.axes import WORLD_AXES
from adam.ontology import OntologyNode, IS_SUBSTANCE
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
    template: Phase1SituationTemplate
) -> Iterable[HighLevelSemanticsSituation]:
    """
    Shorcut for `all_possible` with the GAILA phase 1 ontology and a `RandomChooser` with seed 0.
    """
    return all_possible(
        template, chooser=RandomChooser.for_seed(0), ontology=GAILA_PHASE_1_ONTOLOGY
    )


def situation_object(object_type: OntologyNode) -> SituationObject:
    if GAILA_PHASE_1_ONTOLOGY.has_property(object_type, IS_SUBSTANCE):
        # it's not clear what the axes should be for substances,
        # so we just use the world axes for now
        return SituationObject(object_type, axes=WORLD_AXES)
    else:
        structural_schemata = GAILA_PHASE_1_ONTOLOGY.structural_schemata(object_type)
        if not structural_schemata:
            raise RuntimeError(f"No structural schema found for {object_type}")
        if len(structural_schemata) > 1:
            raise RuntimeError(
                f"Multiple structural schemata available for {object_type}, "
                f"please construct the SituationObject manually: "
                f"{structural_schemata}"
            )
        return first(structural_schemata).axes
