"""
Curricula for DARPA GAILA Phase 1
"""

from itertools import chain
from typing import Iterable

from more_itertools import flatten

from adam.curriculum import GeneratedFromSituationsInstanceGroup, InstanceGroup
from adam.language.dependency import LinearizedDependencyTree
from adam.language_specific.english.english_language_generator import (
    GAILA_PHASE_1_LANGUAGE_GENERATOR,
    PREFER_DITRANSITIVE,
    USE_ADVERBIAL_PATH_MODIFIER,
)
from adam.ontology import IN_REGION, THING
from adam.ontology.during import DuringAction
from adam.ontology.ontology import Ontology
from adam.ontology.phase1_ontology import (
    AGENT,
    BIRD,
    FALL,
    FLY,
    GAILA_PHASE_1_ONTOLOGY,
    GIVE,
    GOAL,
    GROUND,
    HAS,
    HAS_SPACE_UNDER,
    INANIMATE_OBJECT,
    IS_BODY_PART,
    LEARNER,
    LIQUID,
    PERSON,
    PERSON_CAN_HAVE,
    PHASE_1_CURRICULUM_OBJECTS,
    RECOGNIZED_PARTICULAR_PROPERTY,
    THEME,
    TRANSFER_OF_POSSESSION,
    CAN_HAVE_THINGS_RESTING_ON_THEM,
    BIGGER_THAN,
    HOLLOW,
    inside,
    on,
)
from adam.ontology.phase1_spatial_relations import (
    AWAY_FROM,
    DISTAL,
    GRAVITATIONAL_DOWN,
    GRAVITATIONAL_UP,
    Region,
    SpatialPath,
    TOWARD,
)
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    GAILA_PHASE_1_PERCEPTION_GENERATOR,
)
from adam.random_utils import RandomChooser
from adam.relation import Relation
from adam.situation import Action, SituationObject
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    action_variable,
    all_possible,
    color_variable,
    object_variable,
    sampled,
)

_CHOOSER = RandomChooser.for_seed(0)

_Phase1InstanceGroup = InstanceGroup[  # pylint:disable=invalid-name
    HighLevelSemanticsSituation,
    LinearizedDependencyTree,
    DevelopmentalPrimitivePerceptionFrame,
]


def _phase1_instances(
    description: str, situations: Iterable[HighLevelSemanticsSituation]
) -> _Phase1InstanceGroup:
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
_INANIMATE_OBJECT_0 = object_variable(
    "inanimate-object", INANIMATE_OBJECT, required_properties=[PERSON_CAN_HAVE]
)
PERSON_HAS_OBJECT_TEMPLATE = Phase1SituationTemplate(
    object_variables=[_PERSON_0, _INANIMATE_OBJECT_0, _LEARNER_OBJECT],
    asserted_always_relations=[Relation(HAS, _PERSON_0, _INANIMATE_OBJECT_0)],
)

PERSON_HAS_OBJECT_SUB_CURRICULUM = _phase1_instances(
    "person has object",
    situations=sampled(
        PERSON_HAS_OBJECT_TEMPLATE,
        chooser=_CHOOSER,
        ontology=GAILA_PHASE_1_ONTOLOGY,
        max_to_sample=100,
    ),
)

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


def _make_transfer_of_possession_curriculum() -> _Phase1InstanceGroup:
    action_variable("transfer-verb", with_properties=[TRANSFER_OF_POSSESSION])
    giver = object_variable("person_0", PERSON)
    recipient = object_variable("person_1", PERSON)
    given_object = object_variable("give_object_0", INANIMATE_OBJECT)

    return _phase1_instances(
        "transfer-of-possession",
        chain(
            *[
                sampled(
                    Phase1SituationTemplate(
                        object_variables=[giver, recipient, given_object],
                        #
                        actions=[
                            Action(
                                GIVE,
                                argument_roles_to_fillers=[
                                    (AGENT, giver),
                                    (GOAL, recipient),
                                    (THEME, given_object),
                                ],
                            )
                        ],
                        syntax_hints=[PREFER_DITRANSITIVE] if prefer_ditransitive else [],
                    ),
                    max_to_sample=100,
                    chooser=_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                )
                for prefer_ditransitive in (True, False)
            ]
        ),
    )


def _make_object_on_object_curriculum() -> _Phase1InstanceGroup:
    object_ = object_variable("object_0", INANIMATE_OBJECT)
    object_with_surface = object_variable(
        "object_1",
        INANIMATE_OBJECT,
        required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM],
    )
    situation_template = Phase1SituationTemplate(
        object_variables=[object_, object_with_surface],
        constraining_relations=[Relation(BIGGER_THAN, object_with_surface, object_)],
        asserted_always_relations=[on(object_, object_with_surface)],
    )

    return _phase1_instances(
        "objects-on-surfaces",
        sampled(
            situation_template,
            max_to_sample=100,
            chooser=_CHOOSER,
            ontology=GAILA_PHASE_1_ONTOLOGY,
        ),
    )


def _make_object_in_other_object_curriculum() -> _Phase1InstanceGroup:
    object_ = object_variable("object_0", banned_properties=[IS_BODY_PART])
    liquid = object_variable("liquid_0", required_properties=[LIQUID], banned_properties=[IS_BODY_PART])
    containing_object = object_variable(
        "object_1", required_properties=[HOLLOW], banned_properties=[IS_BODY_PART]
    )
    solid_template = Phase1SituationTemplate(
        object_variables=[object_, containing_object],
        constraining_relations=[Relation(BIGGER_THAN, containing_object, object_)],
        asserted_always_relations=[inside(object_, containing_object)],
    )
    liquid_template = Phase1SituationTemplate(
        object_variables=[object_, containing_object],
        asserted_always_relations=[inside(object_, containing_object)],
    )

    return _phase1_instances(
        "objects-in-other-objects",
        chain(
            *[
                sampled(
                    liquid_template,
                    max_to_sample=100,
                    chooser=_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                ),
                sampled(
                    solid_template,
                    max_to_sample=25,
                    chooser=_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                ),
            ]
        ),
    )


def _make_fly_curriculum():
    # fly under something which has an under
    bird = object_variable("bird_0", BIRD)
    object_0 = object_variable("object_0", THING)
    object_with_space_under = object_variable(
        "object_0", THING, required_properties=[HAS_SPACE_UNDER]
    )
    ground = object_variable("ground", GROUND)

    bare_fly = [
        Phase1SituationTemplate(
            object_variables=[bird, ground],
            actions=[
                Action(
                    FLY,
                    argument_roles_to_fillers=[(AGENT, bird)],
                    during=DuringAction(
                        objects_to_paths=[
                            (
                                bird,
                                SpatialPath(
                                    AWAY_FROM if up else TOWARD, reference_object=ground
                                ),
                            )
                        ]
                    ),
                )
            ],
        )
        for up in (True, False)
    ]

    # "a bird flies up"
    # "a bird flies down"
    fly_up_down = [
        Phase1SituationTemplate(
            object_variables=[bird, ground],
            actions=[
                Action(
                    FLY,
                    argument_roles_to_fillers=[(AGENT, bird)],
                    during=DuringAction(
                        objects_to_paths=[
                            (
                                bird,
                                SpatialPath(
                                    AWAY_FROM if up else TOWARD, reference_object=ground
                                ),
                            )
                        ]
                    ),
                )
            ],
            syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER],
        )
        for up in (True, False)
    ]

    # "a bird flies over a house"
    fly_over = Phase1SituationTemplate(
        object_variables=[bird, object_0],
        actions=[
            Action(
                FLY,
                argument_roles_to_fillers=[(AGENT, bird)],
                during=DuringAction(
                    at_some_point=[
                        Relation(
                            IN_REGION,
                            bird,
                            Region(
                                reference_object=object_with_space_under,
                                distance=DISTAL,
                                direction=GRAVITATIONAL_UP,
                            ),
                        )
                    ]
                ),
            )
        ],
    )

    # "a bird flies under a table"
    fly_under = Phase1SituationTemplate(
        object_variables=[bird, object_with_space_under],
        actions=[
            Action(
                FLY,
                argument_roles_to_fillers=[(AGENT, bird)],
                during=DuringAction(
                    at_some_point=[
                        Relation(
                            IN_REGION,
                            bird,
                            Region(
                                reference_object=object_with_space_under,
                                distance=DISTAL,
                                direction=GRAVITATIONAL_DOWN,
                            ),
                        )
                    ]
                ),
            )
        ],
    )

    return _phase1_instances(
        "flying",
        chain(
            *[
                flatten(
                    all_possible(
                        template, ontology=GAILA_PHASE_1_ONTOLOGY, chooser=_CHOOSER
                    )
                    for template in bare_fly
                ),
                flatten(
                    all_possible(
                        template, ontology=GAILA_PHASE_1_ONTOLOGY, chooser=_CHOOSER
                    )
                    for template in fly_up_down
                ),
                all_possible(
                    fly_under, ontology=GAILA_PHASE_1_ONTOLOGY, chooser=_CHOOSER
                ),
                sampled(
                    fly_over,
                    max_to_sample=25,
                    chooser=_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                ),
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
    _make_transfer_of_possession_curriculum(),
    _make_object_on_object_curriculum(),
    _make_object_in_other_object_curriculum(),
    _make_fly_curriculum(),
]
"""
One particular instantiation of the curriculum for GAILA Phase 1.
"""
