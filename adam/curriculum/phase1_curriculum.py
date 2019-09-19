"""
Curricula for DARPA GAILA Phase 1
"""
from typing import Iterable

from adam.curriculum import GeneratedFromSituationsInstanceGroup, InstanceGroup
from adam.language.dependency import LinearizedDependencyTree
from adam.language_specific.english.english_language_generator import (
    GAILA_PHASE_1_LANGUAGE_GENERATOR,
)
from adam.ontology import THING
from adam.ontology.ontology import Ontology
from adam.ontology.phase1_ontology import (
    GAILA_PHASE_1_ONTOLOGY,
    LEARNER,
    PHASE_1_CURRICULUM_OBJECTS,
    RECOGNIZED_PARTICULAR_PROPERTY,
    LIQUID,
    GROUND,
    on,
    IS_BODY_PART,
    HAS, PERSON, INANIMATE_OBJECT)
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    GAILA_PHASE_1_PERCEPTION_GENERATOR,
)
from adam.random_utils import RandomChooser
from adam.relation import Relation
from adam.situation import SituationObject
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


_ARBITRARY_OBJECT = object_variable("object_0", THING)
_NOT_A_BODY_PART = object_variable(
    "not-body-part_object_0", THING, banned_properties=[IS_BODY_PART]
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


def build_object_multiples_situations(
    ontology: Ontology, *, samples_per_object: int = 3, chooser: RandomChooser
) -> Iterable[HighLevelSemanticsSituation]:
    for object_type in PHASE_1_CURRICULUM_OBJECTS:
        # don't want multiples of named people
        is_recognized_particular = any(
            ontology.is_subtype_of(property_, RECOGNIZED_PARTICULAR_PROPERTY)
            for property_ in ontology.properties_for_node(object_type)
        )
        is_liquid = ontology.has_all_properties(object_type, [LIQUID])
        if not is_recognized_particular and not is_liquid:
            for _ in range(samples_per_object):
                num_objects = chooser.choice(range(2, 4))
                yield HighLevelSemanticsSituation(
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    objects=[
                        SituationObject(
                            ontology_node=object_type,
                            debug_handle=object_type.handle + f"_{idx}",
                        )
                        for idx in range(num_objects)
                    ],
                )


MULTIPLE_OF_THE_SAME_OBJECT_SUB_CURRICULUM = _phase1_instances(
    "multiples of the same object",
    build_object_multiples_situations(
        GAILA_PHASE_1_ONTOLOGY, samples_per_object=3, chooser=_CHOOSER
    ),
)

_GROUND = object_variable("the ground", GROUND)

_OBJECT_ON_GROUND_TEMPLATE = Phase1SituationTemplate(
    object_variables=[_GROUND, _NOT_A_BODY_PART],
    asserted_persisting_relations=[on(_NOT_A_BODY_PART, _GROUND)],
)

_OBJECT_ON_GROUND_SUB_CURRICULUM = _phase1_instances(
    "object on ground",
    situations=all_possible(
        _OBJECT_ON_GROUND_TEMPLATE, ontology=GAILA_PHASE_1_ONTOLOGY, chooser=_CHOOSER
    ))

_person_0 = object_variable("person", PERSON)
_inanimate_object_0 = object_variable("inanimate-object", INANIMATE_OBJECT)
PERSON_HAS_OBJECT_TEMPLATE = Phase1SituationTemplate(
    object_variables=[
        _person_0,
        _inanimate_object_0, _LEARNER_OBJECT],
    asserted_persisting_relations=[Relation(HAS, _person_0, _inanimate_object_0)]
)

PERSON_HAS_OBJECT_SUB_CURRICULUM = _phase1_instances(
    "person has object",
    situations=sampled(
        PERSON_HAS_OBJECT_TEMPLATE, chooser=_CHOOSER, ontology=GAILA_PHASE_1_ONTOLOGY,
        max_to_sample=100
    ),
)

GAILA_PHASE_1_CURRICULUM = [
    EACH_OBJECT_BY_ITSELF_SUB_CURRICULUM,
    OBJECTS_WITH_COLORS_SUB_CURRICULUM,
    MULTIPLE_OF_THE_SAME_OBJECT_SUB_CURRICULUM,
    _OBJECT_ON_GROUND_SUB_CURRICULUM,
    PERSON_HAS_OBJECT_SUB_CURRICULUM
]
"""
One particular instantiation of the curriculum for GAILA Phase 1.
"""
