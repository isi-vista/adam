from typing import Iterable, Union

from immutablecollections import immutableset

from adam.curriculum import InstanceGroup, GeneratedFromSituationsInstanceGroup
from adam.language.dependency import LinearizedDependencyTree
from adam.language_specific.english.english_language_generator import (
    GAILA_PHASE_1_LANGUAGE_GENERATOR,
    GAILA_PHASE_2_LANGUAGE_GENERATOR,
)
from adam.ontology import OntologyNode
from adam.ontology.phase1_ontology import (
    GROUND,
    INANIMATE_OBJECT,
    IS_BODY_PART,
    THING,
    LIQUID,
    LEARNER,
)
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    GAILA_PHASE_1_PERCEPTION_GENERATOR,
    HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator,
)
from adam.random_utils import RandomChooser
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.situation.templates.phase1_templates import (
    object_variable,
    TemplatePropertyVariable,
    TemplateObjectVariable,
)

GROUND_OBJECT_TEMPLATE = object_variable("ground", GROUND)
PHASE1_CHOOSER_FACTORY = lambda: RandomChooser.for_seed(0)  # noqa: E731
Phase1InstanceGroup = InstanceGroup[  # pylint:disable=invalid-name
    HighLevelSemanticsSituation,
    LinearizedDependencyTree,
    DevelopmentalPrimitivePerceptionFrame,
]


def standard_object(
    debug_handle: str,
    root_node: OntologyNode = INANIMATE_OBJECT,
    *,
    required_properties: Iterable[OntologyNode] = tuple(),
    banned_properties: Iterable[OntologyNode] = immutableset(),
    added_properties: Iterable[
        Union[OntologyNode, TemplatePropertyVariable]
    ] = immutableset(),
    banned_ontology_types: Iterable[OntologyNode] = immutableset(),
) -> TemplateObjectVariable:
    """
    Preferred method of generating template objects as this automatically prevent liquids and
    body parts from object selection.
    """
    banned_properties_final = [IS_BODY_PART, LIQUID]
    banned_properties_final.extend(banned_properties)
    return object_variable(
        debug_handle=debug_handle,
        root_node=root_node,
        banned_properties=banned_properties_final,
        required_properties=required_properties,
        added_properties=added_properties,
        banned_ontology_types=banned_ontology_types,
    )


def body_part_object(
    debug_handle: str,
    root_node: OntologyNode = THING,
    *,
    required_properties: Iterable[OntologyNode] = tuple(),
    banned_properties: Iterable[OntologyNode] = immutableset(),
    added_properties: Iterable[
        Union[OntologyNode, TemplatePropertyVariable]
    ] = immutableset(),
) -> TemplateObjectVariable:
    """
    Method for generating template objects that are body parts.
    """
    required_properties_final = [IS_BODY_PART]
    required_properties_final.extend(required_properties)
    return object_variable(
        debug_handle=debug_handle,
        root_node=root_node,
        banned_properties=banned_properties,
        required_properties=required_properties_final,
        added_properties=added_properties,
    )


def phase1_instances(
    description: str,
    situations: Iterable[HighLevelSemanticsSituation],
    perception_generator: HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator = GAILA_PHASE_1_PERCEPTION_GENERATOR,
    language_generator=GAILA_PHASE_1_LANGUAGE_GENERATOR,
) -> Phase1InstanceGroup:
    """
    Convenience method for more compactly creating sub-curricula for phase 1.
    """

    return GeneratedFromSituationsInstanceGroup(
        description,
        situations=situations,
        language_generator=language_generator,
        perception_generator=perception_generator,
        chooser=PHASE1_CHOOSER_FACTORY(),
    )


def phase2_instances(
    description: str,
    situations: Iterable[HighLevelSemanticsSituation],
    perception_generator: HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator = GAILA_PHASE_1_PERCEPTION_GENERATOR,
    language_generator=GAILA_PHASE_2_LANGUAGE_GENERATOR,
) -> Phase1InstanceGroup:
    """
    Convenience method for more compactly creating sub-curricula for phase 2.
    """

    return GeneratedFromSituationsInstanceGroup(
        description,
        situations=situations,
        language_generator=language_generator,
        perception_generator=perception_generator,
        chooser=PHASE1_CHOOSER_FACTORY(),
    )


def make_background(
    salient: Iterable[TemplateObjectVariable],
    all_objects: Iterable[TemplateObjectVariable],
) -> Iterable[TemplateObjectVariable]:
    """
    Convenience method for determining which objects in the situation should be background objects
    """
    return immutableset(object_ for object_ in all_objects if object_ not in salient)


def learner_template_factory() -> TemplateObjectVariable:
    return standard_object("learner_factory", LEARNER)
