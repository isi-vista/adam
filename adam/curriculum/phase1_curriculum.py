"""
Curricula for DARPA GAILA Phase 1
"""

from itertools import chain
from typing import Iterable, Sequence

from immutablecollections import immutableset
from more_itertools import flatten

from adam.axes import HorizontalAxisOfObject, FacingAddresseeAxis, AxesInfo
from adam.curriculum.curriculum_utils import (
    phase1_instances,
    PHASE1_CHOOSER,
    Phase1InstanceGroup,
    standard_object,
    GROUND_OBJECT_TEMPLATE,
)
from adam.language_specific.english.english_language_generator import (
    PREFER_DITRANSITIVE,
    USE_ADVERBIAL_PATH_MODIFIER,
)
from adam.ontology import THING, IS_SPEAKER, IS_ADDRESSEE
from adam.ontology.during import DuringAction
from adam.ontology.ontology import Ontology
from adam.ontology.phase1_ontology import (
    AGENT,
    ANIMATE,
    BIRD,
    BOX,
    CAN_BE_SAT_ON_BY_PEOPLE,
    CAN_HAVE_THINGS_RESTING_ON_THEM,
    CAN_JUMP,
    COME,
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
    PUSH,
    PUSH_GOAL,
    PUSH_SURFACE_AUX,
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
    THROW,
    THROW_GOAL,
    TRANSFER_OF_POSSESSION,
    _GO_GOAL,
    bigger_than,
    contacts,
    inside,
    on,
    strictly_above,
    PHASE_1_CURRICULUM_OBJECTS,
    is_recognized_particular,
    near,
    far,
    has,
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
    Direction,
    TO,
)
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    GAILA_PHASE_1_PERCEPTION_GENERATOR,
    HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator,
)
from adam.random_utils import RandomChooser
from adam.relation import flatten_relations
from adam.situation import Action, SituationObject
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    action_variable,
    all_possible,
    color_variable,
    object_variable,
    sampled,
    TemplateObjectVariable,
)


# Show each object once by itself
def _make_each_object_by_itself_curriculum(
    perception_generator: HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator = GAILA_PHASE_1_PERCEPTION_GENERATOR
) -> Phase1InstanceGroup:
    single_object_template = Phase1SituationTemplate(
        "single-object", salient_object_variables=[object_variable("object")]
    )

    return phase1_instances(
        "each object by itself",
        chain(
            *[
                all_possible(
                    single_object_template,
                    chooser=PHASE1_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                )
            ]
        ),
        perception_generator=perception_generator,
    )


# Show each object in 20 different colors


def _make_objects_with_colors_curriculum() -> Phase1InstanceGroup:
    color = color_variable("color")
    object_with_color = standard_object("object", added_properties=[color])

    object_with_color_template = Phase1SituationTemplate(
        "object-with-color", salient_object_variables=[object_with_color]
    )

    return phase1_instances(
        "objects with colors",
        chain(
            *[
                sampled(
                    object_with_color_template,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                    max_to_sample=20,
                )
            ]
        ),
    )


def _make_multiple_objects_curriculum() -> Phase1InstanceGroup:
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
                            SituationObject.instantiate_ontology_node(
                                ontology_node=object_type,
                                debug_handle=object_type.handle + f"_{idx}",
                                ontology=GAILA_PHASE_1_ONTOLOGY,
                            )
                            for idx in range(num_objects)
                        ],
                        axis_info=AxesInfo(),
                    )

    return phase1_instances(
        "multiples of the same object",
        build_object_multiples_situations(
            ontology=GAILA_PHASE_1_ONTOLOGY, chooser=PHASE1_CHOOSER
        ),
    )


def _make_object_on_ground_curriculum() -> Phase1InstanceGroup:
    object_0 = standard_object("object_0")
    liquid_0 = object_variable("liquid_0", THING, required_properties=[LIQUID])

    object_on_ground_template = Phase1SituationTemplate(
        "object-on-ground",
        salient_object_variables=[GROUND_OBJECT_TEMPLATE, object_0],
        asserted_always_relations=[on(object_0, GROUND_OBJECT_TEMPLATE)],
    )

    liquid_on_ground_template = Phase1SituationTemplate(
        "liquid-on-ground",
        salient_object_variables=[liquid_0, GROUND_OBJECT_TEMPLATE],
        asserted_always_relations=[on(liquid_0, GROUND_OBJECT_TEMPLATE)],
    )

    return phase1_instances(
        "object on ground",
        chain(
            *[
                all_possible(
                    object_on_ground_template,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                ),
                all_possible(
                    liquid_on_ground_template,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                ),
            ]
        ),
    )


def _make_person_has_object_curriculum() -> Phase1InstanceGroup:
    person_0 = object_variable("person", PERSON)
    inanimate_object_0 = standard_object(
        "inanimate-object", INANIMATE_OBJECT, required_properties=[PERSON_CAN_HAVE]
    )
    person_has_object_template = Phase1SituationTemplate(
        "person-has-object",
        salient_object_variables=[person_0, inanimate_object_0],
        asserted_always_relations=[has(person_0, inanimate_object_0)],
    )

    return phase1_instances(
        "person has object",
        chain(
            *[
                sampled(
                    person_has_object_template,
                    chooser=PHASE1_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    max_to_sample=100,
                )
            ]
        ),
    )


def _make_fall_curriculum() -> Phase1InstanceGroup:
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

    return phase1_instances(
        "falling objects",
        chain(
            *[
                all_possible(
                    template, ontology=GAILA_PHASE_1_ONTOLOGY, chooser=PHASE1_CHOOSER
                )
                for template in _make_templates()
            ]
        ),
    )


def _make_transfer_of_possession_curriculum() -> Phase1InstanceGroup:
    action_variable("transfer-verb", with_properties=[TRANSFER_OF_POSSESSION])
    giver = object_variable("person_0", PERSON)
    recipient = object_variable("person_1", PERSON)
    given_object = standard_object("give_object_0")

    return phase1_instances(
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
                    chooser=PHASE1_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                )
                for prefer_ditransitive in (True, False)
            ]
        ),
    )


def _make_object_on_object_curriculum() -> Phase1InstanceGroup:
    object_ = object_variable("object_0", INANIMATE_OBJECT)
    object_with_surface = object_variable(
        "object_1",
        INANIMATE_OBJECT,
        required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM],
    )
    situation_template = Phase1SituationTemplate(
        "object-on-surface",
        salient_object_variables=[object_, object_with_surface],
        constraining_relations=[bigger_than(object_with_surface, object_)],
        asserted_always_relations=[on(object_, object_with_surface)],
    )

    return phase1_instances(
        "objects-on-surfaces",
        sampled(
            situation_template,
            max_to_sample=100,
            chooser=PHASE1_CHOOSER,
            ontology=GAILA_PHASE_1_ONTOLOGY,
        ),
    )


def _make_object_beside_object_curriculum() -> Phase1InstanceGroup:
    smaller_beside_object = standard_object("object")
    larger_beside_object = standard_object("larger_beside_object")

    situation_template = Phase1SituationTemplate(
        "object-beside-object",
        salient_object_variables=[smaller_beside_object, larger_beside_object],
        constraining_relations=[bigger_than(larger_beside_object, smaller_beside_object)],
        asserted_always_relations=[
            near(
                smaller_beside_object,
                larger_beside_object,
                direction=Direction(
                    positive=True,
                    relative_to_axis=HorizontalAxisOfObject(
                        larger_beside_object, index=0
                    ),
                ),
            )
        ],
    )

    return phase1_instances(
        "objects-beside-objects",
        sampled(
            situation_template,
            max_to_sample=50,
            chooser=PHASE1_CHOOSER,
            ontology=GAILA_PHASE_1_ONTOLOGY,
        ),
    )


def _make_object_under_or_over_object_curriculum() -> Phase1InstanceGroup:
    object_under = standard_object("object_0")
    object_above = standard_object("object_1", required_properties=[HAS_SPACE_UNDER])
    bird = object_variable("bird_0", BIRD)
    object_under_bird = standard_object("object_under_bird_0")

    templates = [
        Phase1SituationTemplate(
            f"object-under-object",
            salient_object_variables=[object_above],
            constraining_relations=[bigger_than(object_above, object_under)],
            asserted_always_relations=[strictly_above(object_above, object_under)],
        ),
        Phase1SituationTemplate(
            f"object-over-object",
            salient_object_variables=[object_under_bird],
            asserted_always_relations=[strictly_above(bird, object_under_bird)],
        ),
    ]

    return phase1_instances(
        "objects-under-over-objects",
        chain(
            *[
                sampled(
                    template,
                    max_to_sample=100,
                    chooser=PHASE1_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                )
                for template in templates
            ]
        ),
    )


def _make_object_in_other_object_curriculum() -> Phase1InstanceGroup:
    object_ = standard_object("object_0")
    liquid = object_variable(
        "liquid_0", required_properties=[LIQUID], banned_properties=[IS_BODY_PART]
    )
    containing_object = standard_object("object_1", required_properties=[HOLLOW])
    liquid_containing_object = standard_object(
        "object_2", required_properties=[HOLLOW, PERSON_CAN_HAVE]
    )
    solid_template = Phase1SituationTemplate(
        "solid-containment",
        salient_object_variables=[object_, containing_object],
        constraining_relations=[bigger_than(containing_object, object_)],
        asserted_always_relations=[inside(object_, containing_object)],
    )
    liquid_template = Phase1SituationTemplate(
        "liquid-containment",
        salient_object_variables=[liquid, liquid_containing_object],
        asserted_always_relations=[inside(liquid, liquid_containing_object)],
    )

    return phase1_instances(
        "objects-in-other-objects",
        chain(
            *[
                sampled(
                    liquid_template,
                    max_to_sample=25,
                    chooser=PHASE1_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                ),
                sampled(
                    solid_template,
                    max_to_sample=75,
                    chooser=PHASE1_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                ),
            ]
        ),
    )


def _make_fly_curriculum() -> Phase1InstanceGroup:
    # fly under something which has an under
    bird = standard_object("bird_0", BIRD)
    object_0 = standard_object("object_0", THING)
    object_with_space_under = standard_object(
        "object_0", THING, required_properties=[HAS_SPACE_UNDER]
    )

    bare_fly = [
        Phase1SituationTemplate(
            "fly",
            salient_object_variables=[bird],
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
                                    reference_object=GROUND_OBJECT_TEMPLATE,
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
            salient_object_variables=[bird],
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
                                    reference_object=GROUND_OBJECT_TEMPLATE,
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

    return phase1_instances(
        "flying",
        chain(
            *[
                flatten(
                    all_possible(
                        template, ontology=GAILA_PHASE_1_ONTOLOGY, chooser=PHASE1_CHOOSER
                    )
                    for template in bare_fly
                ),
                flatten(
                    all_possible(
                        template, ontology=GAILA_PHASE_1_ONTOLOGY, chooser=PHASE1_CHOOSER
                    )
                    for template in fly_up_down
                ),
                all_possible(
                    fly_under, ontology=GAILA_PHASE_1_ONTOLOGY, chooser=PHASE1_CHOOSER
                ),
                sampled(
                    fly_over,
                    max_to_sample=25,
                    chooser=PHASE1_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                ),
            ]
        ),
    )


def _make_roll_curriculum() -> Phase1InstanceGroup:
    animate_0 = standard_object("object_0", THING, required_properties=[ANIMATE])
    rollable_0 = standard_object(
        "object_1", INANIMATE_OBJECT, required_properties=[ROLLABLE]
    )
    rolling_surface = standard_object(
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

    return phase1_instances(
        "rolling",
        chain(
            *[
                sampled(
                    situation,
                    max_to_sample=25,
                    chooser=PHASE1_CHOOSER,
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


def _make_speaker_addressee_curriculum() -> Phase1InstanceGroup:
    speaker = standard_object("speaker_0", PERSON, added_properties=[IS_SPEAKER])
    addressee = standard_object("addressee_0", PERSON, added_properties=[IS_ADDRESSEE])
    given_object = standard_object("given_object", INANIMATE_OBJECT)

    def _make_templates() -> Iterable[Phase1SituationTemplate]:
        for prefer_ditransitive in (True, False):
            # "you give Mom the cookie"
            yield Phase1SituationTemplate(
                "addressee-agent",
                salient_object_variables=[speaker, addressee, given_object],
                actions=[
                    Action(
                        GIVE,
                        argument_roles_to_fillers=[
                            (AGENT, addressee),
                            (GOAL, speaker),
                            (THEME, given_object),
                        ],
                    )
                ],
                syntax_hints=[PREFER_DITRANSITIVE] if prefer_ditransitive else [],
            )

            # "Mom gives you the cookie"
            yield Phase1SituationTemplate(
                "addressee-goal",
                salient_object_variables=[speaker, addressee, given_object],
                actions=[
                    Action(
                        GIVE,
                        argument_roles_to_fillers=[
                            (AGENT, speaker),
                            (GOAL, addressee),
                            (THEME, given_object),
                        ],
                    )
                ],
                syntax_hints=[PREFER_DITRANSITIVE] if prefer_ditransitive else [],
            )

    return phase1_instances(
        "addressee_curriculum",
        chain(
            *[
                flatten(
                    sampled(
                        template,
                        max_to_sample=25,
                        chooser=PHASE1_CHOOSER,
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                    )
                    for template in _make_templates()
                )
            ]
        ),
    )


def _make_jump_curriculum() -> Phase1InstanceGroup:
    jumper = standard_object("jumper_0", THING, required_properties=[CAN_JUMP])
    jumped_over = standard_object("jumped_over")

    # "A person jumps"
    jump_on_ground = Phase1SituationTemplate(
        "jump",
        salient_object_variables=[jumper],
        actions=[
            Action(
                JUMP,
                argument_roles_to_fillers=[(AGENT, jumper)],
                auxiliary_variable_bindings=[
                    (JUMP_INITIAL_SUPPORTER_AUX, GROUND_OBJECT_TEMPLATE)
                ],
            )
        ],
    )

    # "A person jumps over a ball"
    jump_over_object = Phase1SituationTemplate(
        "jump-over",
        salient_object_variables=[jumper, jumped_over],
        actions=[
            Action(
                JUMP,
                argument_roles_to_fillers=[(AGENT, jumper)],
                during=DuringAction(at_some_point=[strictly_above(jumper, jumped_over)]),
                auxiliary_variable_bindings=[
                    (JUMP_INITIAL_SUPPORTER_AUX, GROUND_OBJECT_TEMPLATE)
                ],
            )
        ],
        constraining_relations=[bigger_than(jumper, jumped_over)],
    )

    return phase1_instances(
        "jumping",
        chain(
            *[
                all_possible(
                    jump_on_ground,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                ),
                sampled(
                    jump_over_object,
                    max_to_sample=25,
                    chooser=PHASE1_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                ),
            ]
        ),
    )


def _make_put_curriculum() -> Phase1InstanceGroup:
    putter = standard_object("putter_0", THING, required_properties=[ANIMATE])
    object_put = standard_object("object_0", required_properties=[INANIMATE])

    on_region_object = standard_object(
        "on_region_object", required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
    )
    in_region_object = standard_object("in_region_object", required_properties=[HOLLOW])

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
            bigger_than(on_region_object, object_put),
            bigger_than(putter, object_put),
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
            bigger_than(in_region_object, object_put),
            bigger_than(putter, object_put),
        ],
    )

    return phase1_instances(
        "putting",
        chain(
            *[
                sampled(
                    put_on_template,
                    max_to_sample=25,
                    chooser=PHASE1_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                ),
                sampled(
                    put_in_template,
                    max_to_sample=25,
                    chooser=PHASE1_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                ),
            ]
        ),
    )


def _make_put_on_speaker_addressee_body_part_curriculum() -> Phase1InstanceGroup:
    speaker_putter = standard_object(
        "speaker_putter_0",
        THING,
        required_properties=[ANIMATE],
        added_properties=[IS_SPEAKER],
    )
    addressee_putter = standard_object(
        "addressee_putter_0",
        THING,
        required_properties=[ANIMATE],
        added_properties=[IS_ADDRESSEE],
    )
    object_put = standard_object("object_put_0", required_properties=[INANIMATE])

    body_part_of_putter = object_variable(
        "body_part_of_putter",
        required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM, IS_BODY_PART],
    )

    # X puts Y on BodyPart
    templates = [
        Phase1SituationTemplate(
            "put-on-body-part",
            salient_object_variables=[putter, object_put, body_part_of_putter],
            actions=[
                Action(
                    PUT,
                    argument_roles_to_fillers=[
                        (AGENT, putter),
                        (THEME, object_put),
                        (
                            GOAL,
                            Region(
                                body_part_of_putter,
                                distance=EXTERIOR_BUT_IN_CONTACT,
                                direction=GRAVITATIONAL_UP,
                            ),
                        ),
                    ],
                )
            ],
            constraining_relations=[
                bigger_than(body_part_of_putter, object_put),
                bigger_than(putter, object_put),
            ],
            asserted_always_relations=[has(putter, body_part_of_putter)],
        )
        for putter in [speaker_putter, addressee_putter]
    ]

    return phase1_instances(
        "putting-on-body-part-addressee-speaker",
        chain(
            *[
                sampled(
                    template,
                    max_to_sample=25,
                    chooser=PHASE1_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                )
                for template in templates
            ]
        ),
    )


def _make_drink_curriculum() -> Phase1InstanceGroup:
    object_0 = standard_object("object_0", required_properties=[HOLLOW])
    liquid_0 = object_variable("liquid_0", required_properties=[LIQUID])
    person_0 = standard_object("person_0", PERSON)

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

    return phase1_instances(
        "drinking",
        chain(
            *[
                all_possible(
                    drink_liquid, ontology=GAILA_PHASE_1_ONTOLOGY, chooser=PHASE1_CHOOSER
                )
            ]
        ),
    )


def _make_eat_curriculum() -> Phase1InstanceGroup:
    object_to_eat = standard_object("object_0", required_properties=[EDIBLE])
    eater = standard_object("eater_0", THING, required_properties=[ANIMATE])

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

    return phase1_instances(
        "eating",
        chain(
            *[
                sampled(
                    eat_object,
                    max_to_sample=25,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                )
            ]
        ),
    )


def _make_sit_curriculum() -> Phase1InstanceGroup:
    sitter = standard_object("sitter_0", THING, required_properties=[ANIMATE])
    sit_surface = standard_object(
        "surface", THING, required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
    )
    seat = standard_object(
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

    return phase1_instances(
        "sitting",
        chain(
            *[
                all_possible(
                    situation_templates,
                    chooser=PHASE1_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                )
                for situation_templates in _make_templates()
            ]
        ),
    )


def _make_take_curriculum() -> Phase1InstanceGroup:
    taker = standard_object("taker_0", THING, required_properties=[ANIMATE])
    object_taken = standard_object("object_taken_0", required_properties=[INANIMATE])

    # X puts Y on Z
    take_template = Phase1SituationTemplate(
        "take",
        salient_object_variables=[taker, object_taken],
        actions=[
            Action(
                TAKE, argument_roles_to_fillers=[(AGENT, taker), (THEME, object_taken)]
            )
        ],
        constraining_relations=[bigger_than(taker, object_taken)],
    )

    return phase1_instances(
        "taking",
        chain(
            *[
                sampled(
                    take_template,
                    max_to_sample=25,
                    chooser=PHASE1_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                )
            ]
        ),
    )


def _make_move_curriculum() -> Phase1InstanceGroup:
    self_mover_0 = standard_object(
        "self-mover_0", THING, required_properties=[SELF_MOVING]
    )

    other_mover_0 = standard_object("mover_0", THING, required_properties=[ANIMATE])
    movee_0 = standard_object("movee_0", THING, required_properties=[INANIMATE])
    move_goal_reference = standard_object(
        "move-goal-reference", THING, required_properties=[INANIMATE]
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

    return phase1_instances(
        "move",
        chain(
            *[
                sampled(
                    situation,
                    max_to_sample=25,
                    chooser=PHASE1_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                )
                for situation in (bare_move_template, transitive_move_template)
            ]
        ),
    )


def _make_spin_curriculum() -> Phase1InstanceGroup:
    self_turner = standard_object("self-spinner_0", THING, required_properties=[ANIMATE])

    other_spinner = standard_object("spinner_0", THING, required_properties=[ANIMATE])
    spinee = standard_object("spinee_0", THING, required_properties=[INANIMATE])

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

    return phase1_instances(
        "spin",
        chain(
            *[
                sampled(
                    situation,
                    max_to_sample=25,
                    chooser=PHASE1_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                )
                for situation in (bare_spin_template, transitive_spin_template)
            ]
        ),
    )


def _make_go_curriculum() -> Phase1InstanceGroup:
    goer = standard_object("goer", THING, required_properties=[ANIMATE])
    goal_reference = standard_object("go-goal", THING, required_properties=[HOLLOW])

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

    return phase1_instances(
        "go",
        chain(
            *[
                sampled(
                    situation,
                    max_to_sample=25,
                    chooser=PHASE1_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                )
                for situation in (bare_go, go_in, go_under)
            ]
        ),
    )


def _make_push_curriculum() -> Phase1InstanceGroup:
    pusher = standard_object("pusher", THING, required_properties=[ANIMATE])
    pushee = standard_object("pushee", INANIMATE_OBJECT)
    push_surface = standard_object(
        "push_surface", THING, required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
    )
    push_goal_reference = standard_object("push_goal", INANIMATE_OBJECT)

    # push with implicit goal
    aux_bindings = [
        (PUSH_SURFACE_AUX, push_surface),
        (PUSH_GOAL, Region(push_goal_reference, distance=PROXIMAL)),
    ]

    # this shouldn't need to be expressed explicitly;
    # we should be able to derive it from the action description
    # https://github.com/isi-vista/adam/issues/239
    # This declaration has mypy confused so we ignore it
    during = DuringAction(continuously=[on(pushee, push_surface)])  # type: ignore
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

    return phase1_instances(
        "pushing",
        chain(
            *[
                sampled(
                    situation,
                    max_to_sample=25,
                    chooser=PHASE1_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                )
                for situation in (
                    push_unexpressed_goal,
                    push_unexpressed_goal_expressed_surface,
                )
            ]
        ),
    )


def _make_throw_curriculum() -> Phase1InstanceGroup:
    thrower = standard_object("thrower_0", THING, required_properties=[ANIMATE])
    object_thrown = standard_object("object_0", required_properties=[INANIMATE])
    implicit_goal_reference = standard_object("implicit_throw_goal_object", BOX)

    # Dad throws a cookie on the ground
    throw_on_ground_template = Phase1SituationTemplate(
        "throw-on-ground",
        salient_object_variables=[thrower, object_thrown, GROUND_OBJECT_TEMPLATE],
        actions=[
            Action(
                THROW,
                argument_roles_to_fillers=[
                    (AGENT, thrower),
                    (THEME, object_thrown),
                    (
                        GOAL,
                        Region(
                            GROUND_OBJECT_TEMPLATE,
                            distance=EXTERIOR_BUT_IN_CONTACT,
                            direction=GRAVITATIONAL_UP,
                        ),
                    ),
                ],
            )
        ],
        constraining_relations=[bigger_than(thrower, object_thrown)],
    )

    # A baby throws a truck
    throw_template = Phase1SituationTemplate(
        "throw",
        salient_object_variables=[thrower, object_thrown],
        actions=[
            Action(
                THROW,
                argument_roles_to_fillers=[(AGENT, thrower), (THEME, object_thrown)],
                auxiliary_variable_bindings=[
                    (THROW_GOAL, Region(implicit_goal_reference, distance=PROXIMAL))
                ],
            )
        ],
        constraining_relations=[bigger_than(thrower, object_thrown)],
    )

    return phase1_instances(
        "throwing",
        chain(
            *[
                sampled(
                    throw_on_ground_template,
                    max_to_sample=25,
                    chooser=PHASE1_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                ),
                sampled(
                    throw_template,
                    max_to_sample=25,
                    chooser=PHASE1_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                ),
            ]
        ),
    )


def _make_come_down_template(
    agent: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    speaker: TemplateObjectVariable,
    ground: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    background_objects_mutable = [speaker, ground]
    background_objects_mutable.extend(background)
    background_objects = immutableset(background_objects_mutable)
    # TODO: Make sure the agent isn't in contact with the ground at the start
    # See: https://github.com/isi-vista/adam/issues/597
    return Phase1SituationTemplate(
        f"{agent.handle}-come-to-{goal_reference.handle}",
        salient_object_variables=[agent, goal_reference],
        background_object_variables=background_objects,
        actions=[
            Action(
                COME,
                argument_roles_to_fillers=[(AGENT, agent), (GOAL, goal_reference)],
                during=DuringAction(
                    objects_to_paths=[
                        (agent, SpatialPath(TOWARD, ground)),
                        (agent, SpatialPath(TO, goal_reference)),
                    ]
                ),
            )
        ],
        asserted_always_relations=flatten_relations(near(speaker, goal_reference)),
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER],
    )


def _make_come_curriculum() -> Phase1InstanceGroup:
    movee = standard_object("movee", required_properties=[SELF_MOVING])
    learner = standard_object("leaner_0", LEARNER)
    speaker = standard_object("speaker", PERSON, added_properties=[IS_SPEAKER])
    object_ = standard_object("object_0", THING)
    ground = standard_object("ground", root_node=GROUND)

    come_to_speaker = Phase1SituationTemplate(
        "come-to-speaker",
        salient_object_variables=[movee, speaker],
        actions=[
            Action(COME, argument_roles_to_fillers=[(AGENT, movee), (GOAL, speaker)])
        ],
    )

    come_to_learner = Phase1SituationTemplate(
        "come-to-leaner",
        salient_object_variables=[movee],
        actions=[
            Action(COME, argument_roles_to_fillers=[(AGENT, movee), (GOAL, learner)])
        ],
    )

    come_to_object = Phase1SituationTemplate(
        "come-to-object",
        salient_object_variables=[movee, object_],
        actions=[
            Action(COME, argument_roles_to_fillers=[(AGENT, movee), (GOAL, object_)])
        ],
    )

    return phase1_instances(
        "come",
        chain(
            *[
                all_possible(
                    come_to_speaker,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                ),
                all_possible(
                    come_to_learner,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                ),
                sampled(
                    come_to_object,
                    max_to_sample=25,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                ),
                sampled(
                    _make_come_down_template(
                        movee, object_, speaker, ground, immutableset()
                    ),
                    max_to_sample=25,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                ),
            ]
        ),
    )


def _make_behind_in_front_curriculum() -> Phase1InstanceGroup:
    front_behind_ground_object = standard_object("ground_object")
    front_behind_figure_object = standard_object("figure_object")
    front_behind_speaker = standard_object(
        "speaker_0", PERSON, added_properties=[IS_SPEAKER]
    )
    front_behind_addressee = standard_object(
        "addressee_0", PERSON, added_properties=[IS_ADDRESSEE]
    )

    def make_behind_in_front_templates() -> Iterable[Phase1SituationTemplate]:
        for in_front_of in (True, False):
            for distal in (True, False):
                suffix = "-in-front" if in_front_of else "-behind"
                direction = Direction(
                    positive=in_front_of,
                    relative_to_axis=FacingAddresseeAxis(front_behind_ground_object),
                )
                yield Phase1SituationTemplate(
                    f"front_behind_addressee-relative-{suffix}",
                    salient_object_variables=[
                        front_behind_figure_object,
                        front_behind_ground_object,
                    ],
                    background_object_variables=[
                        front_behind_speaker,
                        front_behind_addressee,
                    ],
                    asserted_always_relations=[
                        near(
                            front_behind_figure_object,
                            front_behind_ground_object,
                            direction=direction,
                        )
                        if distal
                        else far(
                            front_behind_figure_object,
                            front_behind_ground_object,
                            direction=direction,
                        )
                    ],
                    constraining_relations=[
                        bigger_than(
                            front_behind_ground_object, front_behind_figure_object
                        )
                    ],
                )

    return phase1_instances(
        "behind_in_front_curriculum",
        chain(
            *[
                flatten(
                    sampled(
                        template,
                        max_to_sample=25,
                        chooser=PHASE1_CHOOSER,
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                    )
                    for template in make_behind_in_front_templates()
                )
            ]
        ),
    )


def build_gaila_phase_1_curriculum() -> Sequence[Phase1InstanceGroup]:
    """
    One particular instantiation of the curriculum for GAILA Phase 1.
    """
    return [
        _make_each_object_by_itself_curriculum(),
        _make_objects_with_colors_curriculum(),
        _make_multiple_objects_curriculum(),
        _make_object_on_ground_curriculum(),
        _make_person_has_object_curriculum(),
        _make_fall_curriculum(),
        _make_transfer_of_possession_curriculum(),
        _make_object_on_object_curriculum(),
        _make_object_beside_object_curriculum(),
        _make_object_under_or_over_object_curriculum(),
        _make_object_in_other_object_curriculum(),
        _make_fly_curriculum(),
        _make_roll_curriculum(),
        _make_speaker_addressee_curriculum(),
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
        _make_throw_curriculum(),
        _make_put_on_speaker_addressee_body_part_curriculum(),
        _make_come_curriculum(),
        _make_behind_in_front_curriculum(),
    ]
