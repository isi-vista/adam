"""
Curricula for DARPA GAILA Phase 1
"""

from itertools import chain
from typing import Iterable, Sequence, List

from immutablecollections import immutableset
from more_itertools import flatten, first

from adam.axes import AxesInfo, FacingAddresseeAxis, HorizontalAxisOfObject
from adam.curriculum import ExplicitWithSituationInstanceGroup
from adam.curriculum.curriculum_utils import (
    GROUND_OBJECT_TEMPLATE,
    PHASE1_CHOOSER_FACTORY,
    Phase1InstanceGroup,
    phase1_instances,
    standard_object,
)
from adam.language import TokenSequenceLinguisticDescription
from adam.language_specific.english.english_language_generator import (
    IGNORE_HAS_AS_VERB,
    PREFER_DITRANSITIVE,
    USE_ADVERBIAL_PATH_MODIFIER,
    ATTRIBUTES_AS_X_IS_Y,
    IGNORE_COLORS,
)
from adam.ontology import IS_ADDRESSEE, IS_SPEAKER, THING, OntologyNode
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
    WALK_SURFACE_AUXILIARY,
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
    PASS,
    BABY,
    TRUCK,
    CAR,
    DOG,
    MOM,
    DAD,
    HOUSE,
    BALL,
    WALK,
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
from adam.relation import flatten_relations
from adam.relation_dsl import negate
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
    _fly_under_template,
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


# Show each object once by itself
def _make_each_object_by_itself_curriculum(
    perception_generator: HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator = GAILA_PHASE_1_PERCEPTION_GENERATOR
) -> Phase1InstanceGroup:
    color = color_variable("color")
    single_object_template = Phase1SituationTemplate(
        "single-object",
        salient_object_variables=[object_variable("object", added_properties=[color])],
        syntax_hints=[IGNORE_COLORS],
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
                    max_to_sample=80,
                )
            ]
        ),
    )


def _object_with_color_is_template(
    object_with_color: TemplateObjectVariable,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        "object-with-color-is",
        salient_object_variables=[object_with_color],
        syntax_hints=[ATTRIBUTES_AS_X_IS_Y],
    )


def _make_objects_with_colors_is_curriculum() -> Phase1InstanceGroup:
    color = color_variable("color")
    object_with_color = standard_object("object", added_properties=[color])

    return phase1_instances(
        "objects with colors-is",
        chain(
            *[
                sampled(
                    _object_with_color_is_template(object_with_color),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=80,
                )
            ]
        ),
    )


def _make_plural_objects_curriculum() -> Phase1InstanceGroup:
    def build_object_multiples_situations(
        ontology: Ontology, *, samples_per_object: int = 3, chooser: RandomChooser
    ) -> Iterable[HighLevelSemanticsSituation]:
        for object_type in PHASE_1_CURRICULUM_OBJECTS:
            # Exclude slow objects for now
            if object_type.handle in ["bird", "dog", "truck"]:
                continue
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


def _make_generic_statements_curriculum() -> Phase1InstanceGroup:
    # Hard-coded examples: we create dynamic instances and replace the linguistic description
    # The way we do this is explained here: https://github.com/isi-vista/adam/issues/771
    all_instances = []
    verbs_to_instances = {
        "eat": _make_eat_curriculum().instances(),  # E.g babies eat
        "drink": _make_drink_curriculum().instances(),
        "sit": _make_sit_curriculum().instances(),
        "jump": _make_jump_curriculum().instances(),
        "fly": _make_fly_curriculum().instances(),
    }
    for verb, instances in verbs_to_instances.items():
        for (situation, description, perception) in instances:
            subject = [
                token
                for token in description.as_token_sequence()
                if token not in ["a", "the"]
            ][0]
            all_instances.append(
                (
                    situation,
                    TokenSequenceLinguisticDescription((subject, "s", verb)),
                    perception,
                )
            )
    return ExplicitWithSituationInstanceGroup("generics instances", all_instances)


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


def _make_part_whole_curriculum() -> Phase1InstanceGroup:
    whole_object_to_parts = {
        BABY: ["head", "hand", "arm"],
        BIRD: ["head", "wing"],
        TRUCK: ["tire"],
        CAR: ["tire", "trailer"],
        DAD: ["head", "hand", "arm"],
        MOM: ["head", "hand", "arm"],
        DOG: ["head", "leg"],
        HOUSE: ["wall", "roof"],
    }
    all_instances = []
    for whole_object, parts in whole_object_to_parts.items():
        whole = object_variable("whole", whole_object)

        # Get the description sequence for "[whole] has a [part]" Using a part directly causes issues.
        seq = first(
            phase1_instances(
                "desc",
                situations=sampled(
                    _x_has_y_template(whole, object_variable("filler", BALL)),
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    max_to_sample=1,
                ),
            ).instances()
        )[1].as_token_sequence()

        for part in parts:
            # Replace the filler object with the part object description
            description = TokenSequenceLinguisticDescription(
                tuple([w if w != "ball" else part for w in seq])
            )

            # Get the situation and perception from just the [whole] object
            instances = phase1_instances(
                "desc",
                situations=sampled(
                    Phase1SituationTemplate(
                        f"{whole.handle}", salient_object_variables=[whole]
                    ),
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    max_to_sample=3,
                ),
            ).instances()
            for situation, _, perception in instances:
                all_instances.append((situation, description, perception))

    return ExplicitWithSituationInstanceGroup("part of instances", all_instances)


def _make_my_your_object_curriculum(num_to_sample: int = 20) -> Phase1InstanceGroup:
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


def falling_template(
    theme: TemplateObjectVariable,
    *,
    lands_on_ground: bool,
    syntax_hints: Iterable[str],
    spatial_properties: Iterable[OntologyNode] = immutableset(),
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        "object-falls",
        salient_object_variables=[theme],
        background_object_variables=[GROUND_OBJECT_TEMPLATE],
        actions=[
            Action(
                action_type=FALL,
                argument_roles_to_fillers=[(THEME, theme)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            theme,
                            SpatialPath(
                                None,
                                reference_object=GROUND_OBJECT_TEMPLATE,
                                properties=spatial_properties,
                            ),
                        )
                    ]
                ),
            )
        ],
        syntax_hints=syntax_hints,
        before_action_relations=[negate(contacts(theme, GROUND_OBJECT_TEMPLATE))],
        after_action_relations=flatten_relations([on(theme, GROUND_OBJECT_TEMPLATE)])
        if lands_on_ground
        else flatten_relations([negate(contacts(theme, GROUND_OBJECT_TEMPLATE))]),
    )


def fall_on_ground_template(
    theme: TemplateObjectVariable,
    *,
    spatial_properties: Iterable[OntologyNode] = immutableset(),
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        "falls-to-ground",
        salient_object_variables=[theme, GROUND_OBJECT_TEMPLATE],
        actions=[
            Action(
                action_type=FALL,
                argument_roles_to_fillers=[(THEME, theme)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            theme,
                            SpatialPath(
                                None,
                                reference_object=GROUND_OBJECT_TEMPLATE,
                                properties=spatial_properties,
                            ),
                        )
                    ]
                ),
            )
        ],
        before_action_relations=[negate(on(theme, GROUND_OBJECT_TEMPLATE))],
        after_action_relations=[on(theme, GROUND_OBJECT_TEMPLATE)],
    )


def make_fall_templates() -> Iterable[Phase1SituationTemplate]:
    arbitary_object = standard_object("object_0", THING)
    syntax_hints_options = ([], [USE_ADVERBIAL_PATH_MODIFIER])  # type: ignore

    # Any Object Falling
    object_falling = [
        falling_template(
            arbitary_object,
            lands_on_ground=object_ends_up_on_ground,
            syntax_hints=syntax_hints,
        )
        for object_ends_up_on_ground in (True, False)
        for syntax_hints in syntax_hints_options
    ]

    # "ball fell on the ground"
    return object_falling + [fall_on_ground_template(arbitary_object)]


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


def bare_fly(
    agent: TemplateObjectVariable,
    *,
    up: bool,
    syntax_hints: Iterable[str],
    spatial_properties: Iterable[OntologyNode] = immutableset(),
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        "fly",
        salient_object_variables=[agent],
        actions=[
            Action(
                FLY,
                argument_roles_to_fillers=[(AGENT, agent)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            agent,
                            SpatialPath(
                                AWAY_FROM if up else TOWARD,
                                reference_object=GROUND_OBJECT_TEMPLATE,
                                properties=spatial_properties,
                            ),
                        )
                    ]
                ),
            )
        ],
        syntax_hints=syntax_hints,
    )


def make_fly_templates() -> Iterable[Phase1SituationTemplate]:
    bird = standard_object("bird_0", BIRD)
    object_0 = standard_object("object_0", THING)
    object_with_space_under = standard_object(
        "object_with_space_under", THING, required_properties=[HAS_SPACE_UNDER]
    )
    syntax_hints_options = ([], [USE_ADVERBIAL_PATH_MODIFIER])  # type: ignore

    fly_templates = [
        bare_fly(bird, up=up, syntax_hints=syntax_hints)
        for up in (True, False)
        for syntax_hints in syntax_hints_options
    ]
    # We have fly under disabled due to long run times
    # See https://github.com/isi-vista/adam/issues/672
    return fly_templates + [
        _fly_under_template(bird, object_with_space_under, []),
        _fly_over_template(bird, object_0, []),
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


def intransitive_roll(
    agent: TemplateObjectVariable,
    surface: TemplateObjectVariable,
    *,
    spatial_properties: Iterable[OntologyNode] = immutableset(),
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        "roll-intransitive",
        salient_object_variables=[agent],
        actions=[
            Action(
                ROLL,
                argument_roles_to_fillers=[(AGENT, agent)],
                auxiliary_variable_bindings=[(ROLL_SURFACE_AUXILIARY, surface)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            agent,
                            SpatialPath(
                                None,
                                reference_object=surface,
                                properties=spatial_properties,
                            ),
                        )
                    ]
                ),
            )
        ],
        constraining_relations=[bigger_than(surface, agent)],
    )


def transitive_roll(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    surface: TemplateObjectVariable,
    *,
    spatial_properties: Iterable[OntologyNode] = immutableset(),
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        "roll-transitive",
        salient_object_variables=[agent, theme],
        actions=[
            Action(
                ROLL,
                argument_roles_to_fillers=[(AGENT, agent), (THEME, theme)],
                auxiliary_variable_bindings=[(ROLL_SURFACE_AUXILIARY, surface)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            theme,
                            SpatialPath(
                                None,
                                reference_object=surface,
                                properties=spatial_properties,
                            ),
                        )
                    ]
                ),
            )
        ],
        constraining_relations=[bigger_than(agent, theme)],
    )


def transitive_roll_with_surface(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    surface: TemplateObjectVariable,
    *,
    spatial_properties: Iterable[OntologyNode] = immutableset(),
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        "roll-transitive-with-salient-surface",
        salient_object_variables=[agent, theme, surface],
        actions=[
            Action(
                ROLL,
                argument_roles_to_fillers=[(AGENT, agent), (THEME, theme)],
                auxiliary_variable_bindings=[(ROLL_SURFACE_AUXILIARY, surface)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            theme,
                            SpatialPath(
                                None,
                                reference_object=surface,
                                properties=spatial_properties,
                            ),
                        )
                    ]
                ),
            )
        ],
        asserted_always_relations=[on(theme, surface)],
        constraining_relations=[bigger_than([surface, agent], theme)],
    )


def make_roll_templates() -> Iterable[Phase1SituationTemplate]:
    animate_0 = standard_object("object_0", THING, required_properties=[ANIMATE])
    rollable_0 = standard_object(
        "object_1", INANIMATE_OBJECT, required_properties=[ROLLABLE]
    )
    rolling_surface = standard_object(
        "surface", THING, required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
    )

    return [
        # rolls intransitively
        intransitive_roll(animate_0, rolling_surface),
        # rolls transitively
        transitive_roll(animate_0, rollable_0, rolling_surface),
        # rolls on a surface
        transitive_roll_with_surface(animate_0, rollable_0, rolling_surface),
    ]


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


def make_transitive_roll_templates() -> Iterable[Phase1SituationTemplate]:
    animate_0 = standard_object("object_0", THING, required_properties=[ANIMATE])
    rollable_0 = standard_object(
        "object_1", INANIMATE_OBJECT, required_properties=[ROLLABLE]
    )
    rolling_surface = standard_object(
        "surface", THING, required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
    )

    return [
        # rolls transitively
        transitive_roll(animate_0, rollable_0, rolling_surface),
        # rolls on a surface
        transitive_roll_with_surface(animate_0, rollable_0, rolling_surface),
    ]


def _make_transitive_roll_curriculum() -> Phase1InstanceGroup:
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
                for situation in make_transitive_roll_templates()
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


def make_jump_template(
    agent: TemplateObjectVariable,
    *,
    use_adverbial_path_modifier: bool,
    spatial_properties: Iterable[OntologyNode] = immutableset(),
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        "jump-on-ground",
        salient_object_variables=[agent],
        actions=[
            Action(
                JUMP,
                argument_roles_to_fillers=[(AGENT, agent)],
                auxiliary_variable_bindings=[
                    (JUMP_INITIAL_SUPPORTER_AUX, GROUND_OBJECT_TEMPLATE)
                ],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            agent,
                            SpatialPath(
                                None,
                                reference_object=GROUND_OBJECT_TEMPLATE,
                                properties=spatial_properties,
                            ),
                        )
                    ]
                ),
            )
        ],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER] if use_adverbial_path_modifier else [],
    )


def make_pass_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal: TemplateObjectVariable,
    *,
    use_adverbial_path_modifier: bool,
    spatial_properties: Iterable[OntologyNode] = immutableset(),
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle} tosses {theme.handle}",
        salient_object_variables=[agent, theme, goal],
        actions=[
            Action(
                PASS,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (THEME, theme),
                    (GOAL, Region(goal, distance=PROXIMAL)),
                ],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            agent,
                            SpatialPath(
                                None,
                                reference_object=GROUND_OBJECT_TEMPLATE,
                                properties=spatial_properties,
                            ),
                        )
                    ]
                )
                if spatial_properties
                else None,
            )
        ],
        constraining_relations=[bigger_than(agent, theme)],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER] if use_adverbial_path_modifier else [],
    )


def make_jump_templates():
    jumper = standard_object("jumper_0", THING, required_properties=[CAN_JUMP])
    for use_adverbial_path_modifier in (True, False):
        yield make_jump_template(
            jumper, use_adverbial_path_modifier=use_adverbial_path_modifier
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
                        # "A person jumps"
                        make_jump_template(
                            jumper,
                            use_adverbial_path_modifier=use_adverbial_path_modifier,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                    )
                    for use_adverbial_path_modifier in (True, False)
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


def make_eat_template(
    agent: TemplateObjectVariable,
    patient: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable] = immutableset(),
) -> Phase1SituationTemplate:

    # "Mom eats a cookie"
    return Phase1SituationTemplate(
        "eat-object",
        salient_object_variables=[patient, agent],
        background_object_variables=background,
        actions=[
            Action(EAT, argument_roles_to_fillers=[(AGENT, agent), (PATIENT, patient)])
        ],
    )


def _make_eat_curriculum(
    num_to_sample: int = 25, *, noise_objects: int = 0
) -> Phase1InstanceGroup:
    # TODO: "eat it up"
    # https://github.com/isi-vista/adam/issues/267

    object_to_eat = standard_object("object_0", required_properties=[EDIBLE])
    eater = standard_object("eater_0", THING, required_properties=[ANIMATE])
    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )

    return phase1_instances(
        "eating",
        chain(
            *[
                sampled(
                    make_eat_template(eater, object_to_eat, background),
                    max_to_sample=num_to_sample,
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


def make_take_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    *,
    use_adverbial_path_modifier: bool,
    spatial_properties: Iterable[OntologyNode] = None,
) -> Phase1SituationTemplate:
    # X grabs Y
    ground = GROUND_OBJECT_TEMPLATE
    return Phase1SituationTemplate(
        f"{agent.handle}-take-{theme.handle}",
        salient_object_variables=[agent, theme],
        actions=[
            Action(
                TAKE,
                argument_roles_to_fillers=[(AGENT, agent), (THEME, theme)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            agent,
                            SpatialPath(
                                None,
                                reference_object=ground,
                                properties=spatial_properties,
                            ),
                        )
                    ]
                )
                if spatial_properties
                else None,
            )
        ],
        constraining_relations=[bigger_than(agent, theme)],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER] if use_adverbial_path_modifier else [],
    )


def make_walk_run_template(
    agent: TemplateObjectVariable,
    *,
    use_adverbial_path_modifier: bool,
    spatial_properties: Iterable[OntologyNode] = None,
) -> Phase1SituationTemplate:
    # X walks
    ground = GROUND_OBJECT_TEMPLATE
    return Phase1SituationTemplate(
        f"{agent.handle} walk",
        salient_object_variables=[agent],
        background_object_variables=[ground],
        actions=[
            Action(
                WALK,
                auxiliary_variable_bindings=[(WALK_SURFACE_AUXILIARY, ground)],
                argument_roles_to_fillers=[(AGENT, agent)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            agent,
                            SpatialPath(
                                None,
                                reference_object=ground,
                                properties=spatial_properties,
                            ),
                        )
                    ]
                ),
            )
        ],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER] if use_adverbial_path_modifier else [],
    )


def _make_take_curriculum() -> Phase1InstanceGroup:
    return phase1_instances(
        "taking",
        chain(
            *[
                sampled(
                    make_take_template(
                        agent=standard_object(
                            "taker_0", THING, required_properties=[ANIMATE]
                        ),
                        theme=standard_object(
                            "object_taken_0", required_properties=[INANIMATE]
                        ),
                        use_adverbial_path_modifier=False,
                    ),
                    max_to_sample=25,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                )
            ]
        ),
    )


def bare_move_template(
    agent: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    *,
    spatial_properties: Iterable[OntologyNode] = immutableset(),
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        "bare-move",
        salient_object_variables=[agent],
        actions=[
            Action(
                MOVE,
                argument_roles_to_fillers=[(AGENT, agent)],
                auxiliary_variable_bindings=[
                    (MOVE_GOAL, Region(goal_reference, distance=PROXIMAL))
                ],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            agent,
                            SpatialPath(
                                None,
                                reference_object=goal_reference,
                                properties=spatial_properties,
                            ),
                        )
                    ]
                ),
            )
        ],
    )


def transitive_move_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    *,
    spatial_properties: Iterable[OntologyNode] = immutableset(),
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        "transitive-move",
        salient_object_variables=[agent, theme],
        actions=[
            Action(
                MOVE,
                argument_roles_to_fillers=[(AGENT, agent), (THEME, theme)],
                during=DuringAction(
                    continuously=[contacts(agent, theme)],
                    objects_to_paths=[
                        (
                            theme,
                            SpatialPath(
                                None,
                                reference_object=goal_reference,
                                properties=spatial_properties,
                            ),
                        )
                    ],
                ),
                auxiliary_variable_bindings=[
                    (MOVE_GOAL, Region(goal_reference, distance=PROXIMAL))
                ],
            )
        ],
        constraining_relations=[bigger_than(agent, theme)],
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

    return [
        # bare move (e.g. "a box moves") is about half of uses in child speed
        bare_move_template(self_mover_0, move_goal_reference),
        # Transitive Move
        transitive_move_template(other_mover_0, movee_0, move_goal_reference),
    ]


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


def make_push_templates(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    push_surface: TemplateObjectVariable,
    push_goal: TemplateObjectVariable,
    *,
    use_adverbial_path_modifier: bool,
    spatial_properties: Iterable[OntologyNode] = immutableset(),
) -> List[Phase1SituationTemplate]:
    # push with implicit goal
    aux_bindings = [
        (PUSH_SURFACE_AUX, push_surface),
        (PUSH_GOAL, Region(push_goal, distance=PROXIMAL)),
    ]
    push_unexpressed_goal = Phase1SituationTemplate(
        "push-unexpressed-goal",
        salient_object_variables=[agent, theme],
        actions=[
            Action(
                PUSH,
                argument_roles_to_fillers=[(AGENT, agent), (THEME, theme)],
                auxiliary_variable_bindings=aux_bindings,
                during=DuringAction(
                    continuously=[on(theme, push_surface)],
                    objects_to_paths=[
                        (
                            agent,
                            SpatialPath(
                                None,
                                reference_object=push_goal,
                                properties=spatial_properties,
                            ),
                        )
                    ],
                )
                if spatial_properties
                else DuringAction(continuously=[on(theme, push_surface)]),  # type: ignore
            )
        ],
        constraining_relations=[
            bigger_than(push_surface, agent),
            bigger_than(push_surface, push_goal),
        ],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER] if use_adverbial_path_modifier else [],
    )

    # push with implicit goal
    push_unexpressed_goal_expressed_surface = Phase1SituationTemplate(
        "push-unexpressed-goal",
        salient_object_variables=[agent, theme, push_surface],
        actions=[
            Action(
                PUSH,
                argument_roles_to_fillers=[(AGENT, agent), (THEME, theme)],
                auxiliary_variable_bindings=aux_bindings,
                during=DuringAction(
                    continuously=[on(theme, push_surface)],
                    objects_to_paths=[
                        (
                            agent,
                            SpatialPath(
                                None,
                                reference_object=push_goal,
                                properties=spatial_properties,
                            ),
                        )
                    ],
                )
                if spatial_properties
                else DuringAction(continuously=[on(theme, push_surface)]),  # type: ignore
            )
        ],
        constraining_relations=[bigger_than(push_surface, theme)],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER] if use_adverbial_path_modifier else [],
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
                for situation in make_push_templates(
                    agent=standard_object("pusher", THING, required_properties=[ANIMATE]),
                    theme=standard_object("pushee", INANIMATE_OBJECT),
                    push_surface=standard_object(
                        "push_surface",
                        THING,
                        required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM],
                    ),
                    push_goal=standard_object("push_goal", INANIMATE_OBJECT),
                    use_adverbial_path_modifier=False,
                )
            ]
        ),
    )


def throw_on_ground_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    *,
    spatial_properties: Iterable[OntologyNode] = immutableset(),
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        "throw-on-ground",
        salient_object_variables=[agent, theme, GROUND_OBJECT_TEMPLATE],
        actions=[
            Action(
                THROW,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (THEME, theme),
                    (
                        GOAL,
                        Region(
                            GROUND_OBJECT_TEMPLATE,
                            distance=EXTERIOR_BUT_IN_CONTACT,
                            direction=GRAVITATIONAL_UP,
                        ),
                    ),
                ],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            theme,
                            SpatialPath(
                                None,
                                reference_object=GROUND_OBJECT_TEMPLATE,
                                properties=spatial_properties,
                            ),
                        )
                    ]
                ),
            )
        ],
        constraining_relations=[bigger_than(agent, theme)],
    )


def throw_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal: TemplateObjectVariable,
    *,
    spatial_properties: Iterable[OntologyNode] = immutableset(),
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        "throw",
        salient_object_variables=[agent, theme],
        actions=[
            Action(
                THROW,
                argument_roles_to_fillers=[(AGENT, agent), (THEME, theme)],
                auxiliary_variable_bindings=[
                    (THROW_GOAL, Region(goal, distance=PROXIMAL))
                ],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            theme,
                            SpatialPath(
                                None, reference_object=goal, properties=spatial_properties
                            ),
                        )
                    ]
                ),
            )
        ],
        constraining_relations=[bigger_than(agent, theme)],
    )


def throw_up_down_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal: TemplateObjectVariable,
    *,
    is_up: bool,
    spatial_properties: Iterable[OntologyNode] = immutableset(),
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-throws-{theme.handle}-up-down",
        salient_object_variables=[agent, theme],
        actions=[
            Action(
                THROW,
                argument_roles_to_fillers=[(AGENT, agent), (THEME, theme)],
                auxiliary_variable_bindings=[
                    (THROW_GOAL, Region(goal, distance=PROXIMAL))
                ],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            theme,
                            SpatialPath(
                                AWAY_FROM,
                                reference_object=GROUND_OBJECT_TEMPLATE,
                                properties=spatial_properties,
                            ),
                        )
                    ]
                    if is_up
                    else [
                        (
                            theme,
                            SpatialPath(
                                TOWARD,
                                reference_object=GROUND_OBJECT_TEMPLATE,
                                properties=spatial_properties,
                            ),
                        )
                    ]
                ),
            )
        ],
        constraining_relations=flatten_relations(bigger_than(agent, theme)),
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER],
    )


def throw_to_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal: TemplateObjectVariable,
    *,
    spatial_properties: Iterable[OntologyNode] = None,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        "throw-to",
        salient_object_variables=[agent, theme, goal],
        actions=[
            Action(
                THROW,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (THEME, theme),
                    (GOAL, Region(goal, distance=PROXIMAL)),
                ],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            theme,
                            SpatialPath(
                                None, reference_object=goal, properties=spatial_properties
                            ),
                        )
                    ]
                ),
            )
        ],
        constraining_relations=[bigger_than(agent, theme)],
    )


def make_throw_templates() -> Iterable[Phase1SituationTemplate]:
    thrower = standard_object("thrower_0", THING, required_properties=[ANIMATE])
    catcher = standard_object("catcher_0", THING, required_properties=[ANIMATE])
    object_thrown = standard_object("object_0", required_properties=[INANIMATE])
    implicit_goal_reference = standard_object("implicit_throw_goal_object", BOX)

    return [
        # Dad throws a cookie on the ground
        throw_on_ground_template(thrower, object_thrown),
        # A baby throws a truck
        throw_template(thrower, object_thrown, implicit_goal_reference),
        # Throw up
        throw_up_down_template(
            thrower, object_thrown, implicit_goal_reference, is_up=True
        ),
        # Throw down
        throw_up_down_template(
            thrower, object_thrown, implicit_goal_reference, is_up=False
        ),
        # Throw To
        throw_to_template(thrower, object_thrown, catcher),
    ]


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


def _make_pass_curriculum() -> Phase1InstanceGroup:
    return phase1_instances(
        "passing",
        sampled(
            make_pass_template(
                agent=standard_object("thrower_0", THING, required_properties=[ANIMATE]),
                theme=standard_object("object_0", required_properties=[INANIMATE]),
                goal=standard_object("catcher_0", THING, required_properties=[ANIMATE]),
                use_adverbial_path_modifier=False,
            ),
            max_to_sample=25,
            chooser=PHASE1_CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
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
        before_action_relations=[negate(contacts(agent, ground))],
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


def build_gaila_plurals_curriculum() -> Sequence[Phase1InstanceGroup]:
    return [_make_plural_objects_curriculum()]


def build_gaila_generics_curriculum() -> Sequence[Phase1InstanceGroup]:
    return [_make_generic_statements_curriculum()]


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
        _make_pass_curriculum(),
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
