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
from adam.ontology import THING
from adam.ontology.during import DuringAction
from adam.ontology.ontology import Ontology
from adam.ontology.phase1_ontology import (
    AGENT,
    ANIMATE,
    BIGGER_THAN,
    BIRD,
    CAN_BE_SAT_ON_BY_PEOPLE,
    CAN_HAVE_THINGS_RESTING_ON_THEM,
    CAN_JUMP,
    DRINK,
    DRINK_CONTAINER_AUX,
    EAT,
    EDIBLE,
    FALL,
    FLY,
    GAILA_PHASE_1_ONTOLOGY,
    GIVE,
    GO,
    GOAL,
    GROUND,
    HAS,
    HAS_SPACE_UNDER,
    HOLLOW,
    INANIMATE,
    INANIMATE_OBJECT,
    IS_BODY_PART,
    JUMP,
    JUMP_INITIAL_SUPPORTER_AUX,
    LEARNER,
    LIQUID,
    MOVE,
    MOVE_GOAL,
    PATIENT,
    PERSON,
    PERSON_CAN_HAVE,
    PHASE_1_CURRICULUM_OBJECTS,
    PUT,
    ROLL,
    ROLLABLE,
    ROLL_SURFACE_AUXILIARY,
    SELF_MOVING,
    SIT,
    SIT_GOAL,
    SIT_THING_SAT_ON,
    SPIN,
    TAKE,
    THEME,
    TRANSFER_OF_POSSESSION,
    _GO_GOAL,
    bigger_than,
    contacts,
    inside,
    is_recognized_particular,
    on,
    strictly_above,
    PUSH,
    PUSH_SURFACE_AUX,
    PUSH_GOAL,
)
from adam.ontology.phase1_spatial_relations import (
    AWAY_FROM,
    EXTERIOR_BUT_IN_CONTACT,
    GRAVITATIONAL_DOWN,
    GRAVITATIONAL_UP,
    INTERIOR,
    PROXIMAL,
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


_GROUND_OBJECT = object_variable("ground", GROUND)
_ARBITRARY_OBJECT = object_variable("object_0", THING)
_NOT_A_BODY_PART = object_variable(
    "not-body-part_object_0", THING, banned_properties=[IS_BODY_PART]
)

# Show each object once by itself

_LEARNER_OBJECT = object_variable("learner", LEARNER)

SINGLE_OBJECT_TEMPLATE = Phase1SituationTemplate(
    "single-object", salient_object_variables=[object_variable("object"), _LEARNER_OBJECT]
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
    "object-with-color", salient_object_variables=[_COLOR_OBJECT, _LEARNER_OBJECT]
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
        is_liquid = ontology.has_all_properties(object_type, [LIQUID])
        # don't want multiples of named people
        if not is_recognized_particular(ontology, object_type) and not is_liquid:
            for _ in range(samples_per_object):
                num_objects = chooser.choice(range(2, 4))
                yield HighLevelSemanticsSituation(
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    salient_objects=[
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

_OBJECT_ON_GROUND_TEMPLATE = Phase1SituationTemplate(
    "object-on-ground",
    salient_object_variables=[_GROUND_OBJECT, _NOT_A_BODY_PART],
    asserted_always_relations=[on(_NOT_A_BODY_PART, _GROUND_OBJECT)],
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
    "person-has-object",
    salient_object_variables=[_PERSON_0, _INANIMATE_OBJECT_0, _LEARNER_OBJECT],
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
    "any-object-intransitive",
    salient_object_variables=[_ARBITRARY_OBJECT],
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


def _make_fall_curriculum():
    arbitary_object = object_variable("object_0", THING)
    ground = object_variable("ground_0", GROUND)

    def _make_templates() -> Iterable[Phase1SituationTemplate]:
        # Any Object Falling
        for use_adverbial_path_modifier in (True, False):
            yield Phase1SituationTemplate(
                "object-falls",
                salient_object_variables=[arbitary_object],
                actions=[
                    Action(
                        action_type=FALL,
                        argument_roles_to_fillers=[(THEME, arbitary_object)],
                    )
                ],
                syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER]
                if use_adverbial_path_modifier
                else [],
            )

        # "ball fell on the ground"
        yield Phase1SituationTemplate(
            "falls-to-ground",
            salient_object_variables=[arbitary_object, ground],
            actions=[
                Action(
                    action_type=FALL,
                    argument_roles_to_fillers=[(THEME, arbitary_object)],
                    during=DuringAction(at_some_point=[on(arbitary_object, ground)]),
                )
            ],
        )

    return _phase1_instances(
        "falling objects",
        chain(
            *[
                all_possible(template, ontology=GAILA_PHASE_1_ONTOLOGY, chooser=_CHOOSER)
                for template in _make_templates()
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
                        "transfer-of-possession",
                        salient_object_variables=[giver, recipient, given_object],
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
        "object-on-surface",
        salient_object_variables=[object_, object_with_surface],
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
    liquid = object_variable(
        "liquid_0", required_properties=[LIQUID], banned_properties=[IS_BODY_PART]
    )
    containing_object = object_variable(
        "object_1", required_properties=[HOLLOW], banned_properties=[IS_BODY_PART]
    )
    solid_template = Phase1SituationTemplate(
        "solid-containment",
        salient_object_variables=[object_, containing_object],
        constraining_relations=[Relation(BIGGER_THAN, containing_object, object_)],
        asserted_always_relations=[inside(object_, containing_object)],
    )
    liquid_template = Phase1SituationTemplate(
        "liquid-containment",
        salient_object_variables=[liquid, containing_object],
        asserted_always_relations=[inside(liquid, containing_object)],
    )

    return _phase1_instances(
        "objects-in-other-objects",
        chain(
            *[
                sampled(
                    liquid_template,
                    max_to_sample=25,
                    chooser=_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                ),
                sampled(
                    solid_template,
                    max_to_sample=75,
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

    bare_fly = [
        Phase1SituationTemplate(
            "fly",
            salient_object_variables=[bird, _GROUND_OBJECT],
            actions=[
                Action(
                    FLY,
                    argument_roles_to_fillers=[(AGENT, bird)],
                    during=DuringAction(
                        objects_to_paths=[
                            (
                                bird,
                                SpatialPath(
                                    AWAY_FROM if up else TOWARD,
                                    reference_object=_GROUND_OBJECT,
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
            "fly-up-down",
            salient_object_variables=[bird, _GROUND_OBJECT],
            actions=[
                Action(
                    FLY,
                    argument_roles_to_fillers=[(AGENT, bird)],
                    during=DuringAction(
                        objects_to_paths=[
                            (
                                bird,
                                SpatialPath(
                                    AWAY_FROM if up else TOWARD,
                                    reference_object=_GROUND_OBJECT,
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
        "fly-over",
        salient_object_variables=[bird, object_0],
        actions=[
            Action(
                FLY,
                argument_roles_to_fillers=[(AGENT, bird)],
                during=DuringAction(at_some_point=[strictly_above(bird, object_0)]),
            )
        ],
    )

    # "a bird flies under a table"
    fly_under = Phase1SituationTemplate(
        "fly-under",
        salient_object_variables=[bird, object_with_space_under],
        actions=[
            Action(
                FLY,
                argument_roles_to_fillers=[(AGENT, bird)],
                during=DuringAction(
                    at_some_point=[strictly_above(object_with_space_under, bird)]
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


def _make_roll_curriculum():
    animate_0 = object_variable("object_0", THING, required_properties=[ANIMATE])
    rollable_0 = object_variable(
        "object_1", INANIMATE_OBJECT, required_properties=[ROLLABLE]
    )
    rolling_surface = object_variable(
        "surface", THING, required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
    )

    # rolls intransitively
    # rolls transitively
    # rolls on a surface
    intransitive_roll = Phase1SituationTemplate(
        "roll-intransitive",
        salient_object_variables=[animate_0, rolling_surface],
        actions=[
            Action(
                ROLL,
                argument_roles_to_fillers=[(AGENT, animate_0)],
                auxiliary_variable_bindings=[(ROLL_SURFACE_AUXILIARY, rolling_surface)],
                during=DuringAction(continuously=[on(animate_0, rolling_surface)]),
            )
        ],
        constraining_relations=[bigger_than(rolling_surface, animate_0)],
    )

    transitive_roll = Phase1SituationTemplate(
        "roll-transitive",
        salient_object_variables=[animate_0, rollable_0],
        actions=[
            Action(
                ROLL,
                argument_roles_to_fillers=[(AGENT, animate_0), (THEME, rollable_0)],
                during=DuringAction(continuously=[on(rollable_0, rolling_surface)]),
                auxiliary_variable_bindings=[(ROLL_SURFACE_AUXILIARY, rolling_surface)],
            )
        ],
        constraining_relations=[bigger_than(animate_0, rollable_0)],
    )

    transitive_roll_with_surface = Phase1SituationTemplate(
        "roll-transitive-with-salient-surface",
        salient_object_variables=[animate_0, rollable_0, rolling_surface],
        actions=[
            Action(
                ROLL,
                argument_roles_to_fillers=[(AGENT, animate_0), (THEME, rollable_0)],
                during=DuringAction(continuously=[on(rollable_0, rolling_surface)]),
                auxiliary_variable_bindings=[(ROLL_SURFACE_AUXILIARY, rolling_surface)],
            )
        ],
        constraining_relations=[
            bigger_than(rolling_surface, rollable_0),
            bigger_than(animate_0, rollable_0),
        ],
    )

    return _phase1_instances(
        "rolling",
        chain(
            *[
                sampled(
                    situation,
                    max_to_sample=25,
                    chooser=_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                )
                for situation in (
                    intransitive_roll,
                    transitive_roll,
                    transitive_roll_with_surface,
                )
            ]
        ),
    )


JUMPER = object_variable("jumper_0", THING, required_properties=[CAN_JUMP])
JUMPED_OVER = object_variable("jumped_over", THING)


def _make_jump_curriculum():
    # "A person jumps"
    jump_on_ground = Phase1SituationTemplate(
        "jump",
        salient_object_variables=[JUMPER],
        actions=[
            Action(
                JUMP,
                argument_roles_to_fillers=[(AGENT, JUMPER)],
                auxiliary_variable_bindings=[
                    (JUMP_INITIAL_SUPPORTER_AUX, _GROUND_OBJECT)
                ],
            )
        ],
    )

    # "A person jumps over a ball"
    jump_over_object = make_jump_over_object_template()

    return _phase1_instances(
        "jumping",
        chain(
            *[
                all_possible(
                    jump_on_ground, ontology=GAILA_PHASE_1_ONTOLOGY, chooser=_CHOOSER
                ),
                sampled(
                    jump_over_object,
                    max_to_sample=25,
                    chooser=_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                ),
            ]
        ),
    )


# public for use in a test
def make_jump_over_object_template():
    return Phase1SituationTemplate(
        "jump-over",
        salient_object_variables=[JUMPER, JUMPED_OVER],
        actions=[
            Action(
                JUMP,
                argument_roles_to_fillers=[(AGENT, JUMPER)],
                during=DuringAction(at_some_point=[strictly_above(JUMPER, JUMPED_OVER)]),
                auxiliary_variable_bindings=[
                    (JUMP_INITIAL_SUPPORTER_AUX, _GROUND_OBJECT)
                ],
            )
        ],
        constraining_relations=[Relation(BIGGER_THAN, JUMPER, JUMPED_OVER)],
    )


def _make_put_curriculum():
    putter = object_variable("putter_0", required_properties=[ANIMATE])
    object_put = object_variable(
        "object_0", required_properties=[INANIMATE], banned_properties=[IS_BODY_PART]
    )

    on_region_object = object_variable(
        "on_region_object",
        INANIMATE_OBJECT,
        required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM],
        banned_properties=[IS_BODY_PART],
    )
    in_region_object = object_variable(
        "in_region_object",
        INANIMATE_OBJECT,
        required_properties=[HOLLOW],
        banned_properties=[IS_BODY_PART],
    )

    # X puts Y on Z
    put_on_template = Phase1SituationTemplate(
        "put-on",
        salient_object_variables=[putter, object_put, on_region_object],
        actions=[
            Action(
                PUT,
                argument_roles_to_fillers=[
                    (AGENT, putter),
                    (THEME, object_put),
                    (
                        GOAL,
                        Region(
                            on_region_object,
                            distance=EXTERIOR_BUT_IN_CONTACT,
                            direction=GRAVITATIONAL_UP,
                        ),
                    ),
                ],
            )
        ],
        constraining_relations=[
            Relation(BIGGER_THAN, on_region_object, object_put),
            Relation(BIGGER_THAN, putter, object_put),
        ],
    )

    # X puts Y in Z
    put_in_template = Phase1SituationTemplate(
        "put-in",
        salient_object_variables=[putter, object_put, in_region_object],
        actions=[
            Action(
                PUT,
                argument_roles_to_fillers=[
                    (AGENT, putter),
                    (THEME, object_put),
                    (GOAL, Region(in_region_object, distance=INTERIOR)),
                ],
            )
        ],
        constraining_relations=[
            Relation(BIGGER_THAN, in_region_object, object_put),
            Relation(BIGGER_THAN, putter, object_put),
        ],
    )

    return _phase1_instances(
        "putting",
        chain(
            *[
                sampled(
                    put_on_template,
                    max_to_sample=25,
                    chooser=_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                ),
                sampled(
                    put_in_template,
                    max_to_sample=25,
                    chooser=_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                ),
            ]
        ),
    )


def _make_drink_curriculum():
    object_0 = object_variable(
        "object_0", required_properties=[HOLLOW], banned_properties=[IS_BODY_PART]
    )
    liquid_0 = object_variable("liquid_0", required_properties=[LIQUID])
    person_0 = object_variable("person_0", PERSON)

    drink_liquid = Phase1SituationTemplate(
        "drink",
        salient_object_variables=[liquid_0, person_0],
        actions=[
            Action(
                DRINK,
                argument_roles_to_fillers=[(AGENT, person_0), (THEME, liquid_0)],
                auxiliary_variable_bindings=[(DRINK_CONTAINER_AUX, object_0)],
            )
        ],
    )

    return _phase1_instances(
        "drinking",
        chain(
            *[
                all_possible(
                    drink_liquid, ontology=GAILA_PHASE_1_ONTOLOGY, chooser=_CHOOSER
                )
            ]
        ),
    )


def _make_eat_curriculum():
    object_to_eat = object_variable(
        "object_0",
        INANIMATE_OBJECT,
        required_properties=[EDIBLE],
        banned_properties=[LIQUID],
    )
    eater = object_variable("eater_0", THING, required_properties=[ANIMATE])

    # "Mom eats a cookie"
    eat_object = Phase1SituationTemplate(
        "eat-object",
        salient_object_variables=[object_to_eat, eater],
        actions=[
            Action(
                EAT, argument_roles_to_fillers=[(AGENT, eater), (PATIENT, object_to_eat)]
            )
        ],
    )

    # TODO: "eat it up"
    # https://github.com/isi-vista/adam/issues/267

    return _phase1_instances(
        "eating",
        chain(
            *[
                sampled(
                    eat_object,
                    max_to_sample=25,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=_CHOOSER,
                )
            ]
        ),
    )


def _make_sit_curriculum():
    sitter = object_variable("sitter_0", THING, required_properties=[ANIMATE])
    sit_surface = object_variable(
        "surface", THING, required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
    )
    seat = object_variable(
        "sitting-surface", INANIMATE_OBJECT, required_properties=[CAN_BE_SAT_ON_BY_PEOPLE]
    )

    def _make_templates() -> Iterable[Phase1SituationTemplate]:
        # we need two groups of templates because in general something can sit on
        # anything bigger than itself which has a surface,
        # but people also sit in chairs, etc., which are smaller than them.
        sittee_to_contraints = (
            (  # type: ignore
                "-on-big-thing",
                sit_surface,
                [bigger_than(sit_surface, sitter)],
            ),
            ("-on-seat", seat, []),
        )

        syntax_hints_options = (
            ("default", []),  # type: ignore
            ("adverbial-mod", [USE_ADVERBIAL_PATH_MODIFIER]),
        )

        for (name, sittee, constraints) in sittee_to_contraints:
            for (syntax_name, syntax_hints) in syntax_hints_options:
                yield Phase1SituationTemplate(
                    f"sit-intransitive-{name}-{syntax_name}",
                    salient_object_variables=[sitter],
                    actions=[
                        Action(
                            SIT,
                            argument_roles_to_fillers=[(AGENT, sitter)],
                            auxiliary_variable_bindings=[
                                (
                                    SIT_GOAL,
                                    Region(
                                        sittee,
                                        direction=GRAVITATIONAL_UP,
                                        distance=EXTERIOR_BUT_IN_CONTACT,
                                    ),
                                ),
                                (SIT_THING_SAT_ON, sittee),
                            ],
                        )
                    ],
                    constraining_relations=constraints,
                    syntax_hints=syntax_hints,
                )

                yield Phase1SituationTemplate(
                    f"sit-transitive-{name}-{syntax_name}",
                    salient_object_variables=[sitter, sittee],
                    actions=[
                        Action(
                            SIT,
                            argument_roles_to_fillers=[
                                (AGENT, sitter),
                                (
                                    GOAL,
                                    Region(
                                        sittee,
                                        direction=GRAVITATIONAL_UP,
                                        distance=EXTERIOR_BUT_IN_CONTACT,
                                    ),
                                ),
                            ],
                            auxiliary_variable_bindings=[(SIT_THING_SAT_ON, sittee)],
                        )
                    ],
                    constraining_relations=constraints,
                    syntax_hints=syntax_hints,
                )

    return _phase1_instances(
        "sitting",
        chain(
            *[
                all_possible(
                    situation_templates, chooser=_CHOOSER, ontology=GAILA_PHASE_1_ONTOLOGY
                )
                for situation_templates in _make_templates()
            ]
        ),
    )


def _make_take_curriculum():
    taker = object_variable("taker_0", required_properties=[ANIMATE])
    object_taken = object_variable(
        "object_taken_0",
        required_properties=[INANIMATE],
        banned_properties=[IS_BODY_PART],
    )

    # X puts Y on Z
    take_template = Phase1SituationTemplate(
        "take",
        salient_object_variables=[taker, object_taken],
        actions=[
            Action(
                TAKE, argument_roles_to_fillers=[(AGENT, taker), (THEME, object_taken)]
            )
        ],
        constraining_relations=[Relation(BIGGER_THAN, taker, object_taken)],
    )

    return _phase1_instances(
        "taking",
        chain(
            *[
                sampled(
                    take_template,
                    max_to_sample=25,
                    chooser=_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                )
            ]
        ),
    )


def _make_move_curriculum():
    self_mover_0 = object_variable(
        "self-mover_0", THING, required_properties=[SELF_MOVING]
    )

    other_mover_0 = object_variable("mover_0", THING, required_properties=[ANIMATE])
    movee_0 = object_variable(
        "movee_0",
        THING,
        required_properties=[INANIMATE],
        banned_properties=[IS_BODY_PART],
    )
    move_goal_reference = object_variable(
        "move-goal-reference",
        THING,
        required_properties=[INANIMATE],
        banned_properties=[IS_BODY_PART],
    )

    # since we lack other prepositions at the moment,
    # all movement has the goal of being near an arbitrary inanimate object
    aux_variable_bindings = [(MOVE_GOAL, Region(move_goal_reference, distance=PROXIMAL))]

    # bare move (e.g. "a box moves") is about half of uses in child speed
    bare_move_template = Phase1SituationTemplate(
        "bare-move",
        salient_object_variables=[self_mover_0],
        actions=[
            Action(
                MOVE,
                argument_roles_to_fillers=[(AGENT, self_mover_0)],
                auxiliary_variable_bindings=aux_variable_bindings,
            )
        ],
    )

    transitive_move_template = Phase1SituationTemplate(
        "transitive-move",
        salient_object_variables=[other_mover_0, movee_0],
        actions=[
            Action(
                MOVE,
                argument_roles_to_fillers=[(AGENT, other_mover_0), (THEME, movee_0)],
                during=DuringAction(continuously=[contacts(other_mover_0, movee_0)]),
                auxiliary_variable_bindings=aux_variable_bindings,
            )
        ],
        constraining_relations=[bigger_than(other_mover_0, movee_0)],
    )

    # TODO: version with explicit goal

    return _phase1_instances(
        "move",
        chain(
            *[
                sampled(
                    situation,
                    max_to_sample=25,
                    chooser=_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                )
                for situation in (bare_move_template, transitive_move_template)
            ]
        ),
    )


def _make_spin_curriculum():
    self_turner = object_variable("self-spinner_0", THING, required_properties=[ANIMATE])

    other_spinner = object_variable("spinner_0", THING, required_properties=[ANIMATE])
    spinee = object_variable(
        "spinee_0",
        THING,
        required_properties=[INANIMATE],
        banned_properties=[IS_BODY_PART],
    )

    bare_spin_template = Phase1SituationTemplate(
        "bare-spin",
        salient_object_variables=[self_turner],
        actions=[Action(SPIN, argument_roles_to_fillers=[(AGENT, self_turner)])],
    )

    transitive_spin_template = Phase1SituationTemplate(
        "transitive-spin",
        salient_object_variables=[other_spinner, spinee],
        actions=[
            Action(
                SPIN, argument_roles_to_fillers=[(AGENT, other_spinner), (THEME, spinee)]
            )
        ],
        constraining_relations=[bigger_than(other_spinner, spinee)],
    )

    return _phase1_instances(
        "spin",
        chain(
            *[
                sampled(
                    situation,
                    max_to_sample=25,
                    chooser=_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                )
                for situation in (bare_spin_template, transitive_spin_template)
            ]
        ),
    )


def _make_go_curriculum():
    goer = object_variable("goer", THING, required_properties=[ANIMATE])
    goal_reference = object_variable("go-goal", THING, required_properties=[HOLLOW])

    bare_go = Phase1SituationTemplate(
        "bare-go",
        salient_object_variables=[goer],
        actions=[
            Action(
                GO,
                argument_roles_to_fillers=[(AGENT, goer)],
                auxiliary_variable_bindings=[
                    (_GO_GOAL, Region(goal_reference, distance=PROXIMAL))
                ],
            )
        ],
    )

    go_in = Phase1SituationTemplate(
        "go-in",
        salient_object_variables=[goer, goal_reference],
        actions=[
            Action(
                GO,
                argument_roles_to_fillers=[
                    (AGENT, goer),
                    (GOAL, Region(goal_reference, distance=INTERIOR)),
                ],
            )
        ],
        constraining_relations=[bigger_than(goal_reference, goer)],
    )

    go_under = Phase1SituationTemplate(
        "go-under",
        salient_object_variables=[goer, goal_reference],
        actions=[
            Action(
                GO,
                argument_roles_to_fillers=[
                    (AGENT, goer),
                    (
                        GOAL,
                        Region(
                            goal_reference,
                            distance=PROXIMAL,
                            direction=GRAVITATIONAL_DOWN,
                        ),
                    ),
                ],
            )
        ],
        constraining_relations=[bigger_than(goal_reference, goer)],
    )

    return _phase1_instances(
        "go",
        chain(
            *[
                sampled(
                    situation,
                    max_to_sample=25,
                    chooser=_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                )
                for situation in (bare_go, go_in, go_under)
            ]
        ),
    )


def _make_push_curriculum():
    pusher = object_variable("pusher", THING, required_properties=[ANIMATE])
    pushee = object_variable("pushee", INANIMATE_OBJECT, banned_properties=[IS_BODY_PART])
    push_surface = object_variable(
        "push_surface", THING, required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
    )
    push_goal_reference = object_variable(
        "push_goal", INANIMATE_OBJECT, banned_properties=[IS_BODY_PART]
    )

    # push with implicit goal
    aux_bindings = [
        (PUSH_SURFACE_AUX, push_surface),
        (PUSH_GOAL, Region(push_goal_reference, distance=PROXIMAL)),
    ]

    # this shouldn't need to be expressed explicitly;
    # we should be able to derive it from the action description
    # https://github.com/isi-vista/adam/issues/239
    during = DuringAction(continuously=[on(pushee, push_surface)])
    push_unexpressed_goal = Phase1SituationTemplate(
        "push-unexpressed-goal",
        salient_object_variables=[pusher, pushee],
        actions=[
            Action(
                PUSH,
                argument_roles_to_fillers=[(AGENT, pusher), (THEME, pushee)],
                auxiliary_variable_bindings=aux_bindings,
                during=during,
            )
        ],
        constraining_relations=[
            bigger_than(push_surface, pusher),
            bigger_than(push_surface, push_goal_reference),
        ],
    )

    # push with implicit goal
    push_unexpressed_goal_expressed_surface = Phase1SituationTemplate(
        "push-unexpressed-goal",
        salient_object_variables=[pusher, pushee, push_surface],
        actions=[
            Action(
                PUSH,
                argument_roles_to_fillers=[(AGENT, pusher), (THEME, pushee)],
                auxiliary_variable_bindings=aux_bindings,
                during=during,
            )
        ],
        constraining_relations=[bigger_than(push_surface, pusher)],
    )

    # push explicit goal
    # push_explicit_goal = Phase1SituationTemplate(
    #     "push-explicit-goal",
    #     salient_object_variables=[pusher, push_surface],
    #     actions=[
    #         Action(
    #             PUSH,
    #             argument_roles_to_fillers=[(AGENT, pusher), (THEME, pushee)],
    #             auxiliary_variable_bindings=[(PUSH_SURFACE_AUX, push_surface)]),
    #     ],
    #     constraining_relations=[bigger_than(push_surface, pusher)],
    # )

    return _phase1_instances(
        "pushing",
        chain(
            *[
                sampled(
                    situation,
                    max_to_sample=25,
                    chooser=_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                )
                for situation in (
                    push_unexpressed_goal,
                    push_unexpressed_goal_expressed_surface,
                )
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
    _make_fall_curriculum(),
    _make_transfer_of_possession_curriculum(),
    _make_object_on_object_curriculum(),
    _make_object_in_other_object_curriculum(),
    _make_fly_curriculum(),
    _make_roll_curriculum(),
    _make_jump_curriculum(),
    _make_drink_curriculum(),
    _make_sit_curriculum(),
    _make_put_curriculum(),
    _make_eat_curriculum(),
    _make_take_curriculum(),
    _make_move_curriculum(),
    _make_spin_curriculum(),
    _make_go_curriculum(),
    _make_push_curriculum(),
]
"""
One particular instantiation of the curriculum for GAILA Phase 1.
"""
