from immutablecollections import ImmutableSet

from itertools import chain
from typing import Iterable

from immutablecollections import immutableset
from more_itertools import flatten

from adam.axes import HorizontalAxisOfObject, FacingAddresseeAxis
from adam.curriculum.curriculum_utils import (
    standard_object,
    phase1_instances,
    PHASE1_CHOOSER,
    Phase1InstanceGroup,
)
from adam.language_specific.english.english_language_generator import (
    USE_ADVERBIAL_PATH_MODIFIER,
)
from adam.ontology import THING
from adam.ontology.during import DuringAction
from adam.ontology.phase1_ontology import (
    AGENT,
    bigger_than,
    GOAL,
    GAILA_PHASE_1_ONTOLOGY,
    HOLLOW,
    SIT,
    ANIMATE,
    INANIMATE_OBJECT,
    CAN_HAVE_THINGS_RESTING_ON_THEM,
    SIT_THING_SAT_ON,
    CAN_BE_SAT_ON_BY_PEOPLE,
    GRAVITATIONAL_UP,
    EXTERIOR_BUT_IN_CONTACT,
    HAS_SPACE_UNDER,
    PUSH,
    THEME,
    PUSH_SURFACE_AUX,
    on,
)
from adam.ontology.phase1_spatial_relations import (
    Region,
    PROXIMAL,
    INTERIOR,
    Direction,
    GRAVITATIONAL_DOWN,
    DISTAL,
)
from adam.situation import Action
from adam.situation.templates.phase1_templates import (
    TemplateObjectVariable,
    Phase1SituationTemplate,
    sampled,
)

BOOL_SET = immutableset([True, False])

# PUSH templates


def _push_to_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    surface: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-pushes-{theme.handle}-to-{goal_reference.handle}",
        salient_object_variables=[agent, theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                PUSH,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (THEME, theme),
                    (GOAL, Region(goal_reference, distance=PROXIMAL)),
                ],
                auxiliary_variable_bindings=[(PUSH_SURFACE_AUX, surface)],
                during=DuringAction(continuously=[on(theme, surface)]),
            )
        ],
        constraining_relations=[
            bigger_than(surface, agent),
            bigger_than(surface, goal_reference),
        ],
    )


def _push_in_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    surface: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-pushes-{theme.handle}-in-{goal_reference.handle}",
        salient_object_variables=[agent, theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                PUSH,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (THEME, theme),
                    (GOAL, Region(goal_reference, distance=INTERIOR)),
                ],
                auxiliary_variable_bindings=[(PUSH_SURFACE_AUX, surface)],
                during=DuringAction(continuously=[on(theme, surface)]),
            )
        ],
        constraining_relations=[
            bigger_than(surface, agent),
            bigger_than(surface, goal_reference),
            bigger_than(goal_reference, theme),
        ],
    )


def _push_under_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    surface: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    is_distal: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-pushes-{theme.handle}-under-{goal_reference.handle}",
        salient_object_variables=[agent, theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                PUSH,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (THEME, theme),
                    (
                        GOAL,
                        Region(
                            goal_reference,
                            distance=DISTAL if is_distal else PROXIMAL,
                            direction=GRAVITATIONAL_DOWN,
                        ),
                    ),
                ],
                auxiliary_variable_bindings=[(PUSH_SURFACE_AUX, surface)],
                during=DuringAction(continuously=[on(theme, surface)]),
            )
        ],
        constraining_relations=[
            bigger_than(surface, agent),
            bigger_than(surface, goal_reference),
            bigger_than(goal_reference, theme),
        ],
    )


def _push_beside_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    surface: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    is_right: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-pushes-{theme.handle}-beside-{goal_reference.handle}",
        salient_object_variables=[agent, theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                PUSH,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (THEME, theme),
                    (
                        GOAL,
                        Region(
                            goal_reference,
                            distance=PROXIMAL,
                            direction=Direction(
                                positive=is_right,
                                relative_to_axis=HorizontalAxisOfObject(
                                    goal_reference, index=0
                                ),
                            ),
                        ),
                    ),
                ],
                auxiliary_variable_bindings=[(PUSH_SURFACE_AUX, surface)],
                during=DuringAction(continuously=[on(theme, surface)]),
            )
        ],
        constraining_relations=[
            bigger_than(surface, agent),
            bigger_than(surface, goal_reference),
        ],
    )


def _push_in_front_of_behind_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    surface: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    is_distal: bool,
    is_in_front: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-pushes-{theme.handle}-in-front-of-behind-{goal_reference.handle}",
        salient_object_variables=[agent, theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                PUSH,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (THEME, theme),
                    (
                        GOAL,
                        Region(
                            goal_reference,
                            distance=DISTAL if is_distal else PROXIMAL,
                            direction=Direction(
                                positive=is_in_front,
                                relative_to_axis=FacingAddresseeAxis(goal_reference),
                            ),
                        ),
                    ),
                ],
                auxiliary_variable_bindings=[(PUSH_SURFACE_AUX, surface)],
                during=DuringAction(continuously=[on(theme, surface)]),
            )
        ],
        constraining_relations=[
            bigger_than(surface, agent),
            bigger_than(surface, goal_reference),
        ],
    )


# SIT templates


def _sit_on_template(
    agent: TemplateObjectVariable,
    surface: TemplateObjectVariable,
    seat: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    syntax_hints: ImmutableSet[str],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-sits-on-{seat.handle}",
        salient_object_variables=[agent, seat],
        background_object_variables=background,
        actions=[
            Action(
                SIT,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (
                        GOAL,
                        Region(
                            seat,
                            direction=GRAVITATIONAL_UP,
                            distance=EXTERIOR_BUT_IN_CONTACT,
                        ),
                    ),
                ],
                auxiliary_variable_bindings=[(SIT_THING_SAT_ON, seat)],
            )
        ],
        constraining_relations=[bigger_than(surface, seat)],
        syntax_hints=syntax_hints,
    )


def _sit_in_template(
    agent: TemplateObjectVariable,
    surface: TemplateObjectVariable,
    seat: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    syntax_hints: ImmutableSet[str],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-sits-(down)-in-{seat.handle}",
        salient_object_variables=[agent, seat],
        background_object_variables=background,
        actions=[
            Action(
                SIT,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (GOAL, Region(seat, distance=INTERIOR)),
                ],
                auxiliary_variable_bindings=[(SIT_THING_SAT_ON, seat)],
            )
        ],
        constraining_relations=[bigger_than(surface, seat), bigger_than(seat, agent)],
        syntax_hints=syntax_hints,
    )


# Make curricula


def _make_push_with_prepositions(
    num_samples: int = 5, *, noise_objects: int = 0
) -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[ANIMATE])
    theme = standard_object("theme", INANIMATE_OBJECT)
    goal_reference = standard_object("goal_reference", INANIMATE_OBJECT)
    goal_in = standard_object("goal_in", INANIMATE_OBJECT, required_properties=[HOLLOW])
    goal_under = standard_object(
        "goal_under", INANIMATE_OBJECT, required_properties=[HAS_SPACE_UNDER]
    )
    surface = standard_object(
        "surface", THING, required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
    )
    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )
    to_in_templates = [
        _push_to_template(agent, theme, goal_reference, surface, background),
        _push_in_template(agent, theme, goal_in, surface, background),
    ]

    return phase1_instances(
        "Push + PP",
        chain(
            # to, in
            flatten(
                [
                    sampled(
                        template,
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                    for template in to_in_templates
                ]
            ),
            # beside
            flatten(
                [
                    sampled(
                        _push_beside_template(
                            agent,
                            theme,
                            goal_reference,
                            surface,
                            background,
                            is_right=is_right,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                    for is_right in BOOL_SET
                ]
            ),
            # under
            flatten(
                [
                    sampled(
                        _push_under_template(
                            agent,
                            theme,
                            goal_under,
                            surface,
                            background,
                            is_distal=is_distal,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                    for is_distal in BOOL_SET
                ]
            ),
            # in front of, behind
            flatten(
                [
                    sampled(
                        _push_in_front_of_behind_template(
                            agent,
                            theme,
                            goal_reference,
                            surface,
                            background,
                            is_distal=is_distal,
                            is_in_front=is_in_front,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                    for is_distal in BOOL_SET
                    for is_in_front in BOOL_SET
                ]
            ),
        ),
    )


def _make_sit_with_prepositions(
    num_samples: int = 5, *, noise_objects: int = 0
) -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[ANIMATE])
    seat = standard_object(
        "seat", INANIMATE_OBJECT, required_properties=[CAN_BE_SAT_ON_BY_PEOPLE]
    )
    seat_in = standard_object("seat_in", INANIMATE_OBJECT, required_properties=[HOLLOW])
    surface = standard_object(
        "surface", THING, required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
    )
    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )
    syntax_hints_options = ([], [USE_ADVERBIAL_PATH_MODIFIER])  # type: ignore

    return phase1_instances(
        "Sit + PP",
        chain(
            # on
            flatten(
                [
                    sampled(
                        _sit_on_template(agent, seat, surface, background, syntax_hints),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                    for syntax_hints in syntax_hints_options
                ]
            ),
            # in
            flatten(
                [
                    sampled(
                        _sit_in_template(
                            agent, seat_in, surface, background, syntax_hints
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                    for syntax_hints in syntax_hints_options
                ]
            ),
        ),
    )


def make_verb_with_dynamic_prepositions_curriculum(
    num_samples: int = 5, *, num_noise_objects: int = 0
):
    return [
        _make_push_with_prepositions(num_samples, noise_objects=num_noise_objects),
        _make_sit_with_prepositions(num_samples, noise_objects=num_noise_objects),
    ]
