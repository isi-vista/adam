"""
Curricula for DARPA GAILA Phase 1
"""
from itertools import chain
from typing import Iterable

from adam.curriculum import GeneratedFromSituationsInstanceGroup, InstanceGroup
from adam.language.dependency import LinearizedDependencyTree
from adam.language_specific.english.english_language_generator import (
    GAILA_PHASE_1_LANGUAGE_GENERATOR,
    USE_ADVERBIAL_PATH_MODIFIER,
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
    HAS,
    PERSON,
    INANIMATE_OBJECT,
    bigger_than,
    THEME,
    FALL,
)
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    GAILA_PHASE_1_PERCEPTION_GENERATOR,
)
from adam.random_utils import RandomChooser
from adam.relation import Relation
from adam.situation import SituationObject, Action
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    all_possible,
    color_variable,
    object_variable,
    sampled,
    action_variable,
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
    asserted_always_relations=[on(_NOT_A_BODY_PART, _GROUND)],
)

_OBJECT_ON_GROUND_SUB_CURRICULUM = _phase1_instances(
    "object on ground",
    situations=all_possible(
        _OBJECT_ON_GROUND_TEMPLATE, ontology=GAILA_PHASE_1_ONTOLOGY, chooser=_CHOOSER
    ),
)

_PERSON_0 = object_variable("person", PERSON)
_INANIMATE_OBJECT_0 = object_variable("inanimate-object", INANIMATE_OBJECT)
PERSON_HAS_OBJECT_TEMPLATE = Phase1SituationTemplate(
    object_variables=[_PERSON_0, _INANIMATE_OBJECT_0, _LEARNER_OBJECT],
    asserted_always_relations=[Relation(HAS, _PERSON_0, _INANIMATE_OBJECT_0)],
    constraining_relations=[bigger_than(_PERSON_0, _INANIMATE_OBJECT_0)],
)

# PERSON_HAS_OBJECT_SUB_CURRICULUM = _phase1_instances(
#     "person has object",
#     situations=sampled(
#         PERSON_HAS_OBJECT_TEMPLATE,
#         chooser=_CHOOSER,
#         ontology=GAILA_PHASE_1_ONTOLOGY,
#         max_to_sample=100,
#     ),
# )

_VERB_WITH_ONLY_THEME = action_variable(
    "verb_with_only_theme", with_subcategorization_frame=[THEME]
)

_ANY_OBJECT_INTRANSITIVES_TEMPLATE = Phase1SituationTemplate(
    object_variables=[_ARBITRARY_OBJECT],
    actions=[
        Action(
            action_type=_VERB_WITH_ONLY_THEME,
            argument_roles_to_fillers=[(THEME, _ARBITRARY_OBJECT)],
        )
    ],
)

_ANY_OBJECT_INTRANSITIVES_SUBCURRICULUM = _phase1_instances(
    "any object with an intransitive verb",
    all_possible(
        _ANY_OBJECT_INTRANSITIVES_TEMPLATE,
        ontology=GAILA_PHASE_1_ONTOLOGY,
        chooser=_CHOOSER,
    ),
)


def _make_object_falls_template(
    use_adverbvial_path_modifier: bool
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        object_variables=[_ARBITRARY_OBJECT],
        actions=[
            Action(
                action_type=FALL, argument_roles_to_fillers=[(THEME, _ARBITRARY_OBJECT)]
            )
        ],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER]
        if use_adverbvial_path_modifier
        else [],
    )


_OBJECTS_FALLING_SUBCURRICULUM = _phase1_instances(
    "any object falling",
    chain(
        *[
            all_possible(
                _make_object_falls_template(use_adv_mod),
                ontology=GAILA_PHASE_1_ONTOLOGY,
                chooser=_CHOOSER,
            )
            for use_adv_mod in (True, False)
        ]
    ),
)

GAILA_PHASE_1_CURRICULUM = [
    EACH_OBJECT_BY_ITSELF_SUB_CURRICULUM,
    OBJECTS_WITH_COLORS_SUB_CURRICULUM,
    MULTIPLE_OF_THE_SAME_OBJECT_SUB_CURRICULUM,
    _OBJECT_ON_GROUND_SUB_CURRICULUM,
    #    PERSON_HAS_OBJECT_SUB_CURRICULUM,
    _ANY_OBJECT_INTRANSITIVES_SUBCURRICULUM,
    _OBJECTS_FALLING_SUBCURRICULUM,
]
"""
One particular instantiation of the curriculum for GAILA Phase 1.
"""
