"""
Curricula for DARPA GAILA Phase 1
"""

from itertools import chain
from typing import Iterable, Sequence

from more_itertools import flatten

from adam.axes import AxesInfo, FacingAddresseeAxis, HorizontalAxisOfObject
from adam.curriculum.curriculum_utils import (
    GROUND_OBJECT_TEMPLATE,
    PHASE1_CHOOSER_FACTORY,
    Phase1InstanceGroup,
    phase1_instances,
    standard_object,
)
from adam.language_specific.english.english_language_generator import (
    IGNORE_HAS_AS_VERB,
    PREFER_DITRANSITIVE,
    USE_ADVERBIAL_PATH_MODIFIER,
    ATTRIBUTES_AS_X_IS_Y,
)
from adam.ontology import IS_ADDRESSEE, IS_SPEAKER, THING
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
    PHASE_1_CURRICULUM_OBJECTS,
    PUSH,
    PUSH_GOAL,
    PUSH_SURFACE_AUX,
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
    bigger_than,
    contacts,
    far,
    has,
    inside,
    is_recognized_particular,
    near,
    on,
    strictly_above,
)
from adam.ontology.phase1_spatial_relations import (
    AWAY_FROM,
    Direction,
    EXTERIOR_BUT_IN_CONTACT,
    GRAVITATIONAL_UP,
    PROXIMAL,
    Region,
    SpatialPath,
    TO,
    TOWARD,
)
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    GAILA_PHASE_1_PERCEPTION_GENERATOR,
    HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator,
)
from adam.random_utils import RandomChooser
from adam.relation import flatten_relations, negate
from adam.situation import Action, SituationObject
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.situation.templates.phase1_situation_templates import (
    _fly_over_template,
    _go_in_template,
    _go_to_template,
    _go_under_template,
    _jump_over_template,
    _put_in_template,
    _put_on_body_part_template,
    _put_on_template,
)
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    TemplateObjectVariable,
    action_variable,
    all_possible,
    color_variable,
    object_variable,
    sampled,
)
from immutablecollections import immutableset


# Show each object once by itself
def _make_each_object_by_itself_curriculum(
    perception_generator: HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator = GAILA_PHASE_1_PERCEPTION_GENERATOR
) -> Phase1InstanceGroup:
    single_object_template = Phase1SituationTemplate(
        "single-object", salient_object_variables=[object_variable("object")]
    )
    single_speaker_template = Phase1SituationTemplate(
        "single-speaker",
        salient_object_variables=[
            standard_object("speaker", PERSON, added_properties=[IS_SPEAKER])
        ],
    )
    single_addressee_template = Phase1SituationTemplate(
        "single-addressee",
        salient_object_variables=[
            standard_object("addressee", PERSON, added_properties=[IS_ADDRESSEE])
        ],
    )

    return phase1_instances(
        "each object by itself",
        chain(
            *[
                all_possible(
                    single_object_template,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                ),
                all_possible(
                    single_speaker_template,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                ),
                all_possible(
                    single_addressee_template,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                ),
            ]
        ),
        perception_generator=perception_generator,
    )


# Show each object in 20 different colors


def _object_with_color_template(
    object_with_color: TemplateObjectVariable,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        "object-with-color", salient_object_variables=[object_with_color]
    )


def _make_objects_with_colors_curriculum() -> Phase1InstanceGroup:
    color = color_variable("color")
    object_with_color = standard_object("object", added_properties=[color])

    return phase1_instances(
        "objects with colors",
        chain(
            *[
                sampled(
                    _object_with_color_template(object_with_color),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=20,
                )
            ]
        ),
    )


def _object_with_color_is_template(
    object_with_color: TemplateObjectVariable,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        "object-with-color",
        salient_object_variables=[object_with_color],
        syntax_hints=[ATTRIBUTES_AS_X_IS_Y],
    )


def _make_objects_with_colors_is_curriculum() -> Phase1InstanceGroup:
    color = color_variable("color")
    object_with_color = standard_object("object", added_properties=[color])

    return phase1_instances(
        "objects with colors",
        chain(
            *[
                sampled(
                    _object_with_color_is_template(object_with_color),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=20,
                )
            ]
        ),
    )


def _make_multiple_objects_curriculum() -> Phase1InstanceGroup:
    """
    We are deferring handling numeric quantifiers until Phase 2,
    so this curriculum is not actually executed in Phase 1.
    """

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
            ontology=GAILA_PHASE_1_ONTOLOGY, chooser=PHASE1_CHOOSER_FACTORY()
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
                    chooser=PHASE1_CHOOSER_FACTORY(),
                ),
                all_possible(
                    liquid_on_ground_template,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                ),
            ]
        ),
    )


def _x_has_y_template(
    person: TemplateObjectVariable,
    has_object: TemplateObjectVariable,
    *,
    background: Iterable[TemplateObjectVariable] = immutableset(),
    syntax_hints: Iterable[str] = immutableset(),
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{person.handle}-has-{has_object.handle}",
        salient_object_variables=[person, has_object],
        asserted_always_relations=flatten_relations(has(person, has_object)),
        background_object_variables=background,
        syntax_hints=syntax_hints,
    )


def _make_person_has_object_curriculum() -> Phase1InstanceGroup:
    person_0 = object_variable("person", PERSON)
    inanimate_object_0 = standard_object(
        "inanimate-object", INANIMATE_OBJECT, required_properties=[PERSON_CAN_HAVE]
    )

    return phase1_instances(
        "person has object",
        chain(
            *[
                sampled(
                    _x_has_y_template(person_0, inanimate_object_0),
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    max_to_sample=100,
                )
            ]
        ),
    )


def _make_my_your_object_curriculum(num_to_sample: int = 5) -> Phase1InstanceGroup:
    person_0 = standard_object("speaker", PERSON, added_properties=[IS_SPEAKER])
    person_1 = standard_object("addressee", PERSON, added_properties=[IS_ADDRESSEE])
    inanimate_object = standard_object(
        "object", INANIMATE_OBJECT, required_properties=[PERSON_CAN_HAVE]
    )

    owners = (person_0, person_1)

    return phase1_instances(
        "my-your-object",
        chain(
            *[
                sampled(
                    _x_has_y_template(
                        person,
                        inanimate_object,
                        background=[person_0] if person == person_1 else [],
                        syntax_hints=[IGNORE_HAS_AS_VERB],
                    ),
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    max_to_sample=num_to_sample,
                )
                for person in owners
            ]
        ),
    )


def make_fall_templates() -> Iterable[Phase1SituationTemplate]:
    arbitary_object = standard_object("object_0", THING)
    ground = object_variable("ground_0", GROUND)

    # Any Object Falling
    for object_ends_up_on_ground in (True, False):
        if object_ends_up_on_ground:
            after_action_relations = flatten_relations([on(arbitary_object, ground)])
        else:
            after_action_relations = flatten_relations(
                [negate(contacts(arbitary_object, ground))]
            )

        for use_adverbial_path_modifier in (True, False):
            yield Phase1SituationTemplate(
                "object-falls",
                salient_object_variables=[arbitary_object],
                background_object_variables=[ground],
                actions=[
                    Action(
                        action_type=FALL,
                        argument_roles_to_fillers=[(THEME, arbitary_object)],
                    )
                ],
                syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER]
                if use_adverbial_path_modifier
                else [],
                before_action_relations=[negate(contacts(arbitary_object, ground))],
                after_action_relations=after_action_relations,
            )

    # "ball fell on the ground"
    yield Phase1SituationTemplate(
        "falls-to-ground",
        salient_object_variables=[arbitary_object, ground],
        actions=[
            Action(action_type=FALL, argument_roles_to_fillers=[(THEME, arbitary_object)])
        ],
        after_action_relations=[on(arbitary_object, ground)],
    )


def _make_fall_curriculum() -> Phase1InstanceGroup:

    return phase1_instances(
        "falling objects",
        chain(
            *[
                all_possible(
                    template,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                )
                for template in make_fall_templates()
            ]
        ),
    )


def make_give_templates() -> Iterable[Phase1SituationTemplate]:
    action_variable("transfer-verb", with_properties=[TRANSFER_OF_POSSESSION])
    giver = object_variable("person_0", PERSON)
    recipient = object_variable("person_1", PERSON)
    given_object = standard_object("give_object_0")

    for prefer_ditransitive in (True, False):
        yield Phase1SituationTemplate(
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
        )


def _make_transfer_of_possession_curriculum() -> Phase1InstanceGroup:

    return phase1_instances(
        "transfer-of-possession",
        chain(
            *[
                sampled(
                    template,
                    max_to_sample=100,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                )
                for template in make_give_templates()
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
            chooser=PHASE1_CHOOSER_FACTORY(),
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
            chooser=PHASE1_CHOOSER_FACTORY(),
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
                    chooser=PHASE1_CHOOSER_FACTORY(),
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
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                ),
                sampled(
                    solid_template,
                    max_to_sample=75,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                ),
            ]
        ),
    )


def make_fly_templates() -> Iterable[Phase1SituationTemplate]:
    bird = standard_object("bird_0", BIRD)
    object_0 = standard_object("object_0", THING)
    # object_with_space_under = standard_object(
    #    "object_with_space_under", THING, required_properties=[HAS_SPACE_UNDER]
    # )
    syntax_hints_options = ([], [USE_ADVERBIAL_PATH_MODIFIER])  # type: ignore

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
            syntax_hints=syntax_hints,
        )
        for up in (True, False)
        for syntax_hints in syntax_hints_options
    ]
    # We have fly under disabled due to long run times
    # See https://github.com/isi-vista/adam/issues/672
    return bare_fly + [
        # _fly_under_template(bird, object_with_space_under, []),
        _fly_over_template(bird, object_0, [])
    ]


def _make_fly_curriculum() -> Phase1InstanceGroup:
    return phase1_instances(
        "flying",
        chain(
            flatten(
                [
                    all_possible(
                        template,
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                    )
                    for template in make_fly_templates()
                ]
            )
        ),
    )


def make_roll_templates() -> Iterable[Phase1SituationTemplate]:
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
        salient_object_variables=[animate_0],
        actions=[
            Action(
                ROLL,
                argument_roles_to_fillers=[(AGENT, animate_0)],
                auxiliary_variable_bindings=[(ROLL_SURFACE_AUXILIARY, rolling_surface)],
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
                auxiliary_variable_bindings=[(ROLL_SURFACE_AUXILIARY, rolling_surface)],
            )
        ],
        asserted_always_relations=[on(rollable_0, rolling_surface)],
        constraining_relations=[
            bigger_than(rolling_surface, rollable_0),
            bigger_than(animate_0, rollable_0),
        ],
    )
    return [intransitive_roll, transitive_roll, transitive_roll_with_surface]


def _make_roll_curriculum() -> Phase1InstanceGroup:
    return phase1_instances(
        "rolling",
        chain(
            *[
                sampled(
                    situation,
                    max_to_sample=25,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                )
                for situation in make_roll_templates()
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
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                    )
                    for template in _make_templates()
                )
            ]
        ),
    )


def make_jump_templates() -> Iterable[Phase1SituationTemplate]:
    jumper = standard_object("jumper_0", THING, required_properties=[CAN_JUMP])

    # "A person jumps"
    for use_adverbial_path_modifier in (True, False):
        yield Phase1SituationTemplate(
            "jump-on-ground",
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
            syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER]
            if use_adverbial_path_modifier
            else [],
        )


def _make_jump_curriculum() -> Phase1InstanceGroup:

    jumper = standard_object("jumper_0", THING, required_properties=[CAN_JUMP])
    jumped_over = standard_object("jumped_over")

    return phase1_instances(
        "jumping",
        chain(
            flatten(
                [
                    all_possible(
                        template,
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                    )
                    for template in make_jump_templates()
                ]
            ),
            flatten(
                [
                    sampled(
                        _jump_over_template(jumper, jumped_over, []),
                        max_to_sample=25,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                    )
                ]
            ),
        ),
    )


def make_put_templates() -> Iterable[Phase1SituationTemplate]:
    putter = standard_object("putter_0", THING, required_properties=[ANIMATE])
    object_put = standard_object("object_0", required_properties=[INANIMATE])
    on_region_object = standard_object(
        "on_region_object", required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
    )
    in_region_object = standard_object("in_region_object", required_properties=[HOLLOW])
    return [
        _put_on_template(putter, object_put, on_region_object, []),
        _put_in_template(putter, object_put, in_region_object, []),
    ]


def _make_put_curriculum() -> Phase1InstanceGroup:

    return phase1_instances(
        "putting",
        chain(
            flatten(
                [
                    sampled(
                        template,
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=25,
                    )
                    for template in make_put_templates()
                ]
            )
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

    return phase1_instances(
        "putting-on-body-part-addressee-speaker",
        chain(
            flatten(
                [
                    sampled(
                        _put_on_body_part_template(
                            putter, object_put, body_part_of_putter, []
                        ),
                        max_to_sample=25,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                    )
                    for putter in [speaker_putter, addressee_putter]
                ]
            )
        ),
    )


def make_drink_template() -> Phase1SituationTemplate:
    object_0 = standard_object("object_0", required_properties=[HOLLOW])
    liquid_0 = object_variable("liquid_0", required_properties=[LIQUID])
    person_0 = standard_object("person_0", PERSON)

    return Phase1SituationTemplate(
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


def _make_drink_curriculum() -> Phase1InstanceGroup:

    return phase1_instances(
        "drinking",
        chain(
            *[
                all_possible(
                    make_drink_template(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                )
            ]
        ),
    )


def make_eat_template() -> Phase1SituationTemplate:
    object_to_eat = standard_object("object_0", required_properties=[EDIBLE])
    eater = standard_object("eater_0", THING, required_properties=[ANIMATE])

    # "Mom eats a cookie"
    return Phase1SituationTemplate(
        "eat-object",
        salient_object_variables=[object_to_eat, eater],
        actions=[
            Action(
                EAT, argument_roles_to_fillers=[(AGENT, eater), (PATIENT, object_to_eat)]
            )
        ],
    )


def _make_eat_curriculum() -> Phase1InstanceGroup:
    # TODO: "eat it up"
    # https://github.com/isi-vista/adam/issues/267

    return phase1_instances(
        "eating",
        chain(
            *[
                sampled(
                    make_eat_template(),
                    max_to_sample=25,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                )
            ]
        ),
    )


def make_sit_templates() -> Iterable[Phase1SituationTemplate]:
    sitter = standard_object("sitter_0", THING, required_properties=[ANIMATE])
    sit_surface = standard_object(
        "surface", THING, required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
    )
    seat = standard_object(
        "sitting-surface", INANIMATE_OBJECT, required_properties=[CAN_BE_SAT_ON_BY_PEOPLE]
    )

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


def _make_sit_curriculum() -> Phase1InstanceGroup:

    return phase1_instances(
        "sitting",
        chain(
            *[
                all_possible(
                    situation_templates,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                )
                for situation_templates in make_sit_templates()
            ]
        ),
    )


def make_take_template() -> Phase1SituationTemplate:
    taker = standard_object("taker_0", THING, required_properties=[ANIMATE])
    object_taken = standard_object("object_taken_0", required_properties=[INANIMATE])

    # X puts Y on Z
    return Phase1SituationTemplate(
        "take",
        salient_object_variables=[taker, object_taken],
        actions=[
            Action(
                TAKE, argument_roles_to_fillers=[(AGENT, taker), (THEME, object_taken)]
            )
        ],
        constraining_relations=[bigger_than(taker, object_taken)],
    )


def _make_take_curriculum() -> Phase1InstanceGroup:

    return phase1_instances(
        "taking",
        chain(
            *[
                sampled(
                    make_take_template(),
                    max_to_sample=25,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                )
            ]
        ),
    )


def make_move_templates() -> Iterable[Phase1SituationTemplate]:
    self_mover_0 = standard_object(
        "self-mover_0", THING, required_properties=[SELF_MOVING]
    )

    other_mover_0 = standard_object("mover_0", THING, required_properties=[ANIMATE])
    movee_0 = standard_object("movee_0", THING, required_properties=[INANIMATE])
    move_goal_reference = standard_object(
        "move-goal-reference", THING, required_properties=[INANIMATE]
    )

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
    return [bare_move_template, transitive_move_template]


def _make_move_curriculum() -> Phase1InstanceGroup:

    return phase1_instances(
        "move",
        chain(
            *[
                sampled(
                    situation,
                    max_to_sample=25,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                )
                for situation in make_move_templates()
            ]
        ),
    )


def make_spin_templates() -> Iterable[Phase1SituationTemplate]:
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
    return [bare_spin_template, transitive_spin_template]


def _make_spin_curriculum() -> Phase1InstanceGroup:
    return phase1_instances(
        "spin",
        chain(
            *[
                sampled(
                    situation,
                    max_to_sample=25,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                )
                for situation in make_spin_templates()
            ]
        ),
    )


def make_go_templates() -> Iterable[Phase1SituationTemplate]:
    goer = standard_object("goer", THING, required_properties=[ANIMATE])
    goal_reference = standard_object("go-goal", THING)
    in_goal_reference = standard_object("go-in-goal", THING, required_properties=[HOLLOW])

    go_to = _go_to_template(goer, goal_reference, [])
    go_in = _go_in_template(goer, in_goal_reference, [])
    return [go_to, go_in]


def _make_go_curriculum() -> Phase1InstanceGroup:
    goer = standard_object("goer", THING, required_properties=[ANIMATE])
    under_goal_reference = standard_object(
        "go-under-goal", THING, required_properties=[HAS_SPACE_UNDER]
    )

    return phase1_instances(
        "go",
        chain(
            flatten(
                [
                    sampled(
                        situation,
                        max_to_sample=25,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                    )
                    for situation in make_go_templates()
                ]
            ),
            flatten(
                [
                    sampled(
                        _go_under_template(
                            goer, under_goal_reference, [], is_distal=is_distal
                        ),
                        max_to_sample=25,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                    )
                    for is_distal in (True, False)
                ]
            ),
        ),
    )


def make_push_templates() -> Iterable[Phase1SituationTemplate]:
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
    return [push_unexpressed_goal, push_unexpressed_goal_expressed_surface]


def _make_push_curriculum() -> Phase1InstanceGroup:

    return phase1_instances(
        "pushing",
        chain(
            *[
                sampled(
                    situation,
                    max_to_sample=25,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                )
                for situation in make_push_templates()
            ]
        ),
    )


def make_throw_templates() -> Iterable[Phase1SituationTemplate]:
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

    # Throw up, down
    throw_up_down_templates = [
        Phase1SituationTemplate(
            f"{thrower.handle}-throws-{object_thrown.handle}-up-down",
            salient_object_variables=[thrower, object_thrown],
            actions=[
                Action(
                    THROW,
                    argument_roles_to_fillers=[(AGENT, thrower), (THEME, object_thrown)],
                    auxiliary_variable_bindings=[
                        (THROW_GOAL, Region(implicit_goal_reference, distance=PROXIMAL))
                    ],
                    during=DuringAction(
                        objects_to_paths=[
                            (
                                object_thrown,
                                SpatialPath(
                                    AWAY_FROM, reference_object=GROUND_OBJECT_TEMPLATE
                                ),
                            )
                        ]
                        if is_up
                        else [
                            (
                                object_thrown,
                                SpatialPath(
                                    TOWARD, reference_object=GROUND_OBJECT_TEMPLATE
                                ),
                            )
                        ]
                    ),
                )
            ],
            constraining_relations=flatten_relations(bigger_than(thrower, object_thrown)),
            syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER],
        )
        for is_up in (True, False)
    ]
    return throw_up_down_templates + [throw_template] + [throw_on_ground_template]


def _make_throw_curriculum() -> Phase1InstanceGroup:

    return phase1_instances(
        "throwing",
        chain(
            flatten(
                [
                    sampled(
                        template,
                        max_to_sample=25,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                    )
                    for template in make_throw_templates()
                ]
            )
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
                    chooser=PHASE1_CHOOSER_FACTORY(),
                ),
                all_possible(
                    come_to_learner,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                ),
                sampled(
                    come_to_object,
                    max_to_sample=25,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                ),
                sampled(
                    _make_come_down_template(
                        movee, object_, speaker, ground, immutableset()
                    ),
                    max_to_sample=25,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
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
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                    )
                    for template in make_behind_in_front_templates()
                )
            ]
        ),
    )


def build_gaila_phase1_object_curriculum() -> Sequence[Phase1InstanceGroup]:
    """
    One particular instantiation of the object-learning parts of the curriculum for GAILA Phase 1.
    """
    return [
        _make_each_object_by_itself_curriculum(),
        #     We are deferring handling numeric quantifiers until Phase 2,
        #     so this curriculum is not actually executed in Phase 1.
        # _make_multiple_objects_curriculum(),
        _make_object_on_ground_curriculum(),
    ]


def build_gaila_phase1_attribute_curriculum() -> Sequence[Phase1InstanceGroup]:
    """
    One particular instantiation of the object-learning parts of the curriculum for GAILA Phase 1.
    """
    return [
        _make_objects_with_colors_curriculum(),
        _make_objects_with_colors_is_curriculum(),
        _make_my_your_object_curriculum(),
    ]


def build_gaila_phase1_relation_curriculum() -> Sequence[Phase1InstanceGroup]:
    """
    One particular instantiation of the object-learning parts of the curriculum for GAILA Phase 1.
    """
    return [
        _make_person_has_object_curriculum(),
        _make_object_on_object_curriculum(),
        _make_object_beside_object_curriculum(),
        _make_object_under_or_over_object_curriculum(),
        _make_object_in_other_object_curriculum(),
        _make_behind_in_front_curriculum(),
    ]


def build_gaila_phase1_verb_curriculum() -> Sequence[Phase1InstanceGroup]:
    """
    One particular instantiation of the object-learning parts of the curriculum for GAILA Phase 1.
    """
    return [
        _make_fall_curriculum(),
        _make_transfer_of_possession_curriculum(),
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
        # _make_put_on_speaker_addressee_body_part_curriculum(),
        _make_come_curriculum(),
    ]


def build_gaila_phase_1_curriculum() -> Sequence[Phase1InstanceGroup]:
    """
    One particular instantiation of the curriculum for GAILA Phase 1.
    """
    return list(
        chain(
            build_gaila_phase1_object_curriculum(),
            build_gaila_phase1_attribute_curriculum(),
            build_gaila_phase1_relation_curriculum(),
            build_gaila_phase1_verb_curriculum(),
        )
    )
