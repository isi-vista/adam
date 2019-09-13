"""
Curricula for DARPA GAILA Phase 1
"""
from typing import Iterable

from adam.curriculum import GeneratedFromSituationsInstanceGroup, InstanceGroup
from adam.language.dependency import LinearizedDependencyTree
from adam.language_specific.english.english_language_generator import (
    GAILA_PHASE_1_LANGUAGE_GENERATOR,
)
from adam.ontology.phase1_ontology import GAILA_PHASE_1_ONTOLOGY, LEARNER
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    GAILA_PHASE_1_PERCEPTION_GENERATOR,
)
from adam.random_utils import RandomChooser
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    all_possible,
    color_variable,
    object_variable,
    sampled,
)

_CHOOSER = RandomChooser.for_seed(0)


def _phase1_instances(
    description: str, situations: Iterable[HighLevelSemanticsSituation]
) -> InstanceGroup[
    HighLevelSemanticsSituation,
    LinearizedDependencyTree,
    DevelopmentalPrimitivePerceptionFrame,
]:
    """
    Convenience method for more compactly creating sub-curricula for phase 1.
    """

    return GeneratedFromSituationsInstanceGroup(
        description,
        situations=situations,
        language_generator=GAILA_PHASE_1_LANGUAGE_GENERATOR,
        perception_generator=GAILA_PHASE_1_PERCEPTION_GENERATOR,
        chooser=_CHOOSER,
    )


# Show each object once by itself

_LEARNER_OBJECT = object_variable("learner", LEARNER)

SINGLE_OBJECT_TEMPLATE = Phase1SituationTemplate(
    object_variables=[object_variable("object"), _LEARNER_OBJECT]
)

EACH_OBJECT_BY_ITSELF_SUB_CURRICULUM = _phase1_instances(
    "each object by itself",
    situations=all_possible(
        SINGLE_OBJECT_TEMPLATE, chooser=_CHOOSER, ontology=GAILA_PHASE_1_ONTOLOGY
    ),
)

# Show each object in 20 different colors

_COLOR = color_variable("color")
_COLOR_OBJECT = object_variable("object", added_properties=[_COLOR])
_OBJECT_WITH_COLOR_TEMPLATE = Phase1SituationTemplate(
    object_variables=[_COLOR_OBJECT, _LEARNER_OBJECT]
)

OBJECTS_WITH_COLORS_SUB_CURRICULUM = _phase1_instances(
    "objects with colors",
    situations=sampled(
        _OBJECT_WITH_COLOR_TEMPLATE,
        chooser=_CHOOSER,
        ontology=GAILA_PHASE_1_ONTOLOGY,
        max_to_sample=20,
    ),
)


GAILA_PHASE_1_CURRICULUM = [
    EACH_OBJECT_BY_ITSELF_SUB_CURRICULUM,
    OBJECTS_WITH_COLORS_SUB_CURRICULUM,
]
"""
One particular instantiation of the curriculum for GAILA Phase 1.
"""
