from random import Random

from typing import Iterable, Union, Optional, Sequence, List, Tuple, Any

from adam.axes import HorizontalAxisOfObject, FacingAddresseeAxis
from adam.ontology import IS_SPEAKER, IS_ADDRESSEE
from immutablecollections import immutableset
from adam.language.language_generator import LanguageGenerator
from adam.language.dependency import LinearizedDependencyTree
from adam.curriculum import (
    InstanceGroup,
    GeneratedFromSituationsInstanceGroup,
    AblatedPerceptionSituationsInstanceGroup,
)
from adam.language_specific.english.english_language_generator import (
    GAILA_PHASE_1_LANGUAGE_GENERATOR,
    GAILA_PHASE_2_LANGUAGE_GENERATOR,
    GAILA_PHASE_3_LANGUAGE_GENERATOR,
)
from adam.ontology import OntologyNode
from adam.ontology.phase1_ontology import (
    GROUND,
    INANIMATE_OBJECT,
    IS_BODY_PART,
    THING,
    LIQUID,
    LEARNER,
    on,
    near,
    strictly_under,
    far,
    PHASE_3_CONCEPT,
)
from adam.ontology.phase1_spatial_relations import Direction, DISTAL, PROXIMAL
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    GAILA_PHASE_1_PERCEPTION_GENERATOR,
    HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator,
    GAILA_PHASE_2_PERCEPTION_GENERATOR,
)
from adam.random_utils import RandomChooser
from adam.relation import flatten_relations, Relation
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.situation.templates.phase1_templates import (
    object_variable,
    TemplatePropertyVariable,
    TemplateObjectVariable,
)
from adam.language import LinguisticDescriptionT
from adam.perception import (
    PerceptionT,
    PerceptualRepresentation,
    PerceptualRepresentationFrame,
)
from adam.situation import SituationT

GROUND_OBJECT_TEMPLATE = object_variable("ground", GROUND)
CHOOSER_FACTORY = lambda: RandomChooser.for_seed(0)  # noqa: E731
TEST_CHOOSER_FACTORY = lambda: RandomChooser.for_seed(1)  # noqa: E731
Phase1InstanceGroup = InstanceGroup[
    HighLevelSemanticsSituation,
    LinearizedDependencyTree,
    DevelopmentalPrimitivePerceptionFrame,
]
Phase3InstanceGroup = InstanceGroup[
    HighLevelSemanticsSituation, LinearizedDependencyTree, PerceptualRepresentationFrame
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


# Get only a Phase 3 object in the standard set
def phase3_standard_object(
    debug_handle: str,
    root_node: OntologyNode = THING,
    *,
    required_properties: Iterable[OntologyNode] = tuple(),
    banned_properties: Iterable[OntologyNode] = immutableset(),
    added_properties: Iterable[
        Union[OntologyNode, TemplatePropertyVariable]
    ] = immutableset(),
    banned_ontology_types: Iterable[OntologyNode] = immutableset(),
) -> TemplateObjectVariable:
    """
    Preferred method of generating phase 3 template objects as this automatically limits to concepts which are
    marked as Phase 3 and prevents liquids and body parts from object selection.
    """
    required_properties_final = [PHASE_3_CONCEPT]
    required_properties_final.extend(required_properties)
    banned_properties_final = [IS_BODY_PART, LIQUID]
    banned_properties_final.extend(banned_properties)
    return object_variable(
        debug_handle=debug_handle,
        root_node=root_node,
        banned_properties=banned_properties_final,
        required_properties=required_properties_final,
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
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ] = GAILA_PHASE_1_LANGUAGE_GENERATOR,
) -> Phase1InstanceGroup:
    """
    Convenience method for more compactly creating sub-curricula for phase 1.
    """

    return GeneratedFromSituationsInstanceGroup(
        description,
        situations=situations,
        language_generator=language_generator,
        perception_generator=perception_generator,
        chooser=CHOOSER_FACTORY(),
    )


def phase2_instances(
    description: str,
    situations: Iterable[HighLevelSemanticsSituation],
    perception_generator: HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator = GAILA_PHASE_2_PERCEPTION_GENERATOR,
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ] = GAILA_PHASE_2_LANGUAGE_GENERATOR,
) -> Phase1InstanceGroup:
    """
    Convenience method for more compactly creating sub-curricula for phase 2.
    """

    return GeneratedFromSituationsInstanceGroup(
        description,
        situations=situations,
        language_generator=language_generator,
        perception_generator=perception_generator,
        chooser=CHOOSER_FACTORY(),
    )


def phase3_instances(
    description: str,
    situations: Iterable[HighLevelSemanticsSituation],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ] = GAILA_PHASE_3_LANGUAGE_GENERATOR,
) -> Phase3InstanceGroup:
    """
    Convenience method for more compactly creating sub-curricula for phase 3.
    """

    return AblatedPerceptionSituationsInstanceGroup(
        description,
        situations=situations,
        language_generator=language_generator,
        chooser=CHOOSER_FACTORY(),
    )


def make_background(
    salient: Iterable[TemplateObjectVariable],
    all_objects: Iterable[TemplateObjectVariable],
) -> Iterable[TemplateObjectVariable]:
    """
    Convenience method for determining which objects in the situation should be background objects
    """
    return immutableset(object_ for object_ in all_objects if object_ not in salient)


def make_noise_objects(
    noise_objects: Optional[int],
    banned_ontology_types: Iterable[OntologyNode] = immutableset(),
) -> Iterable[TemplateObjectVariable]:
    return immutableset(
        standard_object(
            f"noise_object_{x}",
            banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
            banned_ontology_types=banned_ontology_types,
        )
        for x in range(noise_objects if noise_objects else 0)
    )


def learner_template_factory() -> TemplateObjectVariable:
    return standard_object("learner_factory", LEARNER)


def shuffle_curriculum(
    curriculum: List[
        Tuple[SituationT, LinguisticDescriptionT, PerceptualRepresentation[PerceptionT]]
    ],
    *,
    rng: Random,
) -> Sequence[
    Tuple[SituationT, LinguisticDescriptionT, PerceptualRepresentation[PerceptionT]]
]:
    mod_curriculum = list(curriculum)
    rng.shuffle(mod_curriculum)
    return mod_curriculum


NOISE_RELATION_DSL_OPTIONS = immutableset(["on", "beside", "under", "in_front"])
BOOL_SET = (True, False)


def background_relations_builder(
    background_objects: Iterable[TemplateObjectVariable],
    num_relations: int,
    *,
    target: Optional[TemplateObjectVariable] = None,
    target_2: Optional[TemplateObjectVariable] = None,
    add_noise: bool = True,
    include_targets_in_noise: bool = False,
    chooser: RandomChooser = RandomChooser.for_seed(0),
) -> Iterable[Relation[Any]]:
    if add_noise:
        potential_objects = list(background_objects)
        if target and include_targets_in_noise:
            potential_objects.append(target)
        if target_2 and include_targets_in_noise:
            potential_objects.append(target_2)

        if len(potential_objects) < 2:
            return immutableset()

        relations = []
        for _ in range(num_relations):
            choice = chooser.choice(NOISE_RELATION_DSL_OPTIONS)
            if choice == "on":
                relations.append(
                    on(
                        chooser.choice(potential_objects),
                        chooser.choice(potential_objects),
                    )
                )
            elif choice == "beside":
                obj_choice_2 = chooser.choice(potential_objects)
                relations.append(
                    near(
                        chooser.choice(potential_objects),
                        obj_choice_2,
                        direction=Direction(
                            positive=chooser.choice(BOOL_SET),
                            relative_to_axis=HorizontalAxisOfObject(
                                obj_choice_2, index=0
                            ),
                        ),
                    )
                )
            elif choice == "under":
                relations.append(
                    strictly_under(
                        chooser.choice(potential_objects),
                        chooser.choice(potential_objects),
                        dist=DISTAL if chooser.choice(BOOL_SET) else PROXIMAL,
                    )
                )
            elif choice == "in_front":
                obj_choice_2 = chooser.choice(potential_objects)
                direction = Direction(
                    positive=chooser.choice(BOOL_SET),
                    relative_to_axis=FacingAddresseeAxis(obj_choice_2),
                )
                relations.append(
                    near(
                        chooser.choice(potential_objects),
                        obj_choice_2,
                        direction=direction,
                    )
                    if chooser.choice(BOOL_SET)
                    else far(
                        chooser.choice(potential_objects),
                        obj_choice_2,
                        direction=direction,
                    )
                )
            else:
                raise RuntimeError("Invalid relation type in background relations")

        return flatten_relations(relations)
    else:
        return immutableset()
