from immutablecollections import ImmutableSet

from itertools import chain
from typing import Iterable

from immutablecollections import immutableset
from more_itertools import flatten

from adam.axes import (
    HorizontalAxisOfObject,
    FacingAddresseeAxis,
    GRAVITATIONAL_AXIS_FUNCTION,
)
from adam.curriculum.curriculum_utils import (
    standard_object,
    phase1_instances,
    PHASE1_CHOOSER,
    Phase1InstanceGroup,
    make_background,
)
from adam.language_specific.english.english_language_generator import (
    USE_ADVERBIAL_PATH_MODIFIER,
)
from adam.ontology import THING
from adam.ontology.during import DuringAction
from adam.ontology.phase1_ontology import (
    AGENT,
    FALL,
    GOAL,
    GAILA_PHASE_1_ONTOLOGY,
    HOLLOW,
    SIT,
    SIT_THING_SAT_ON,
    CAN_BE_SAT_ON_BY_PEOPLE,
    EXTERIOR_BUT_IN_CONTACT,
    HAS_SPACE_UNDER,
    PUSH,
    THEME,
    PUSH_SURFACE_AUX,
    ANIMATE,
    INANIMATE_OBJECT,
    CAN_HAVE_THINGS_RESTING_ON_THEM,
    GO,
    ROLL,
    _GO_GOAL,
    ROLL_SURFACE_AUXILIARY,
    ROLLABLE,
    GROUND,
    above,
    on,
    bigger_than,
    near,
    far,
    inside,
)
from adam.ontology.phase1_spatial_relations import (
    Region,
    PROXIMAL,
    INTERIOR,
    Direction,
    GRAVITATIONAL_DOWN,
    DISTAL,
    GRAVITATIONAL_UP,
    SpatialPath,
    VIA,
)
from adam.relation import flatten_relations
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


def _go_to_template(
    agent: TemplateObjectVariable,
    goal_object: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"go_to-{agent.handle}-to-{goal_object.handle}",
        salient_object_variables=[agent, goal_object],
        background_object_variables=background,
        actions=[
            Action(
                GO,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (GOAL, Region(goal_object, distance=PROXIMAL)),
                ],
            )
        ],
        gazed_objects=[agent],
    )


def _go_in_template(
    agent: TemplateObjectVariable,
    goal_object: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"go_in-{agent.handle}-in-{goal_object.handle}",
        salient_object_variables=[agent, goal_object],
        background_object_variables=background,
        actions=[
            Action(
                GO,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (GOAL, Region(goal_object, distance=INTERIOR)),
                ],
            )
        ],
        constraining_relations=[bigger_than(goal_object, agent)],
    )


def _go_beside_template(
    agent: TemplateObjectVariable,
    goal_object: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_right: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"go_beside-{agent.handle}-beside-{goal_object.handle}",
        salient_object_variables=[agent, goal_object],
        background_object_variables=background,
        actions=[
            Action(
                GO,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (
                        GOAL,
                        Region(
                            goal_object,
                            distance=PROXIMAL,
                            direction=Direction(
                                positive=is_right,
                                relative_to_axis=HorizontalAxisOfObject(
                                    goal_object, index=0
                                ),
                            ),
                        ),
                    ),
                ],
            )
        ],
    )


def _go_behind_in_front_template(
    agent: TemplateObjectVariable,
    goal_object: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_distal: bool,
    is_behind: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"go_behind-{agent.handle}-behind-{goal_object.handle}",
        salient_object_variables=[agent, goal_object],
        background_object_variables=background,
        actions=[
            Action(
                GO,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (
                        GOAL,
                        Region(
                            goal_object,
                            distance=DISTAL if is_distal else PROXIMAL,
                            direction=Direction(
                                positive=False if is_behind else True,
                                relative_to_axis=FacingAddresseeAxis(goal_object),
                            ),
                        ),
                    ),
                ],
            )
        ],
    )


def _go_over_under_template(
    agent: TemplateObjectVariable,
    goal_object: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_distal: bool,
    is_over: bool,
) -> Phase1SituationTemplate:
    handle = "over" if is_over else "under"
    return Phase1SituationTemplate(
        f"go_{handle}-{agent.handle}-{handle}-{goal_object.handle}",
        salient_object_variables=[agent, goal_object],
        background_object_variables=background,
        actions=[
            Action(
                GO,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (
                        GOAL,
                        Region(
                            goal_object,
                            distance=DISTAL if is_distal else PROXIMAL,
                            direction=GRAVITATIONAL_UP if is_over else GRAVITATIONAL_DOWN,
                        ),
                    ),
                ],
            )
        ],
    )


def _go_behind_in_front_path_template(
    agent: TemplateObjectVariable,
    goal_object: TemplateObjectVariable,
    path_object: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_behind: bool,
) -> Phase1SituationTemplate:
    additional_background = [goal_object, path_object]
    additional_background.extend(background)
    total_background = immutableset(additional_background)
    handle = "behind" if is_behind else "in-front-of"
    return Phase1SituationTemplate(
        f"go_{handle}-{agent.handle}-{handle}-{goal_object.handle}-via-{path_object.handle}",
        salient_object_variables=[agent],
        background_object_variables=total_background,
        actions=[
            Action(
                GO,
                argument_roles_to_fillers=[(AGENT, agent)],
                auxiliary_variable_bindings=[(_GO_GOAL, goal_object)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            path_object,
                            SpatialPath(
                                operator=VIA,
                                reference_object=path_object,
                                reference_axis=FacingAddresseeAxis(path_object),
                                orientation_changed=True,
                            ),
                        )
                    ],
                    # TODO: ADD 'at_some_point' condition for in_front or behind regional conditions
                    # See: https://github.com/isi-vista/adam/issues/583
                ),
            )
        ],
        gazed_objects=[agent],
    )


def _go_over_under_path_template(
    agent: TemplateObjectVariable,
    goal_object: TemplateObjectVariable,
    path_object: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_over: bool,
) -> Phase1SituationTemplate:
    additional_background = [goal_object, path_object]
    additional_background.extend(background)
    total_background = immutableset(additional_background)
    handle = "over" if is_over else "under"
    return Phase1SituationTemplate(
        f"go_{handle}-{agent.handle}-{handle}-{goal_object.handle}-via-{path_object.handle}",
        salient_object_variables=[agent],
        background_object_variables=total_background,
        actions=[
            Action(
                GO,
                argument_roles_to_fillers=[(AGENT, agent)],
                auxiliary_variable_bindings=[(_GO_GOAL, goal_object)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            path_object,
                            SpatialPath(
                                operator=VIA,
                                reference_object=path_object,
                                reference_axis=GRAVITATIONAL_AXIS_FUNCTION,
                            ),
                        )
                    ],
                    at_some_point=[
                        above(agent, path_object)
                        if is_over
                        else above(path_object, agent)
                    ],
                ),
            )
        ],
        gazed_objects=[agent],
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
        constraining_relations=[bigger_than(surface, seat), bigger_than(seat, agent)],
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


def _x_roll_beside_y_template(
    agent: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    surface: TemplateObjectVariable,
    *,
    is_right: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-rolls-beside-{goal_reference.handle}",
        salient_object_variables=[agent, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                ROLL,
                argument_roles_to_fillers=[(AGENT, agent)],
                auxiliary_variable_bindings=[(ROLL_SURFACE_AUXILIARY, surface)],
            )
        ],
        after_action_relations=flatten_relations(
            near(
                agent,
                goal_reference,
                direction=Direction(
                    positive=is_right,
                    relative_to_axis=HorizontalAxisOfObject(goal_reference, index=0),
                ),
            )
        ),
        gazed_objects=[agent],
    )


def _x_roll_behind_in_front_y_template(
    agent: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    surface: TemplateObjectVariable,
    *,
    is_distal: bool,
    is_behind: bool,
) -> Phase1SituationTemplate:
    direction = Direction(
        positive=True if is_behind else False,
        relative_to_axis=FacingAddresseeAxis(goal_reference),
    )
    return Phase1SituationTemplate(
        f"{agent.handle}-rolls-behind-{goal_reference.handle}",
        salient_object_variables=[agent, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                ROLL,
                argument_roles_to_fillers=[(AGENT, agent)],
                auxiliary_variable_bindings=[(ROLL_SURFACE_AUXILIARY, surface)],
            )
        ],
        after_action_relations=flatten_relations(
            far(agent, goal_reference, direction=direction)
            if is_distal
            else near(agent, goal_reference, direction=direction)
        ),
        gazed_objects=[agent],
    )


def _x_roll_under_y_template(
    agent: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    surface: TemplateObjectVariable,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-rolls-under-{goal_reference.handle}",
        salient_object_variables=[agent, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                ROLL,
                argument_roles_to_fillers=[(AGENT, agent)],
                auxiliary_variable_bindings=[(ROLL_SURFACE_AUXILIARY, surface)],
            )
        ],
        after_action_relations=flatten_relations(above(goal_reference, agent)),
        gazed_objects=[agent],
    )


def _x_roll_y_in_z_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    surface: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-rolls-{theme.handle}-in-{goal_reference.handle}",
        salient_object_variables=[agent, theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                ROLL,
                argument_roles_to_fillers=[(AGENT, agent), (THEME, theme)],
                auxiliary_variable_bindings=[(ROLL_SURFACE_AUXILIARY, surface)],
            )
        ],
        constraining_relations=[bigger_than([agent, goal_reference], theme)],
        after_action_relations=flatten_relations(inside(theme, goal_reference)),
        gazed_objects=[theme],
    )


def _x_roll_y_beside_z_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    surface: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_right: bool,
) -> Phase1SituationTemplate:
    direction = Direction(
        positive=True if is_right else False,
        relative_to_axis=HorizontalAxisOfObject(goal_reference, index=0),
    )
    return Phase1SituationTemplate(
        f"{agent.handle}-rolls-{theme.handle}-beside-{goal_reference.handle}",
        salient_object_variables=[agent, theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                ROLL,
                argument_roles_to_fillers=[(AGENT, agent), (THEME, theme)],
                auxiliary_variable_bindings=[(ROLL_SURFACE_AUXILIARY, surface)],
            )
        ],
        constraining_relations=[bigger_than(agent, theme)],
        after_action_relations=flatten_relations(
            near(theme, goal_reference, direction=direction)
        ),
        gazed_objects=[theme],
    )


def _x_roll_y_behind_in_front_z_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    surface: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_distal: bool,
    is_behind: bool,
) -> Phase1SituationTemplate:
    value = "behind" if is_behind else "in-front-of"
    direction = Direction(
        positive=True if is_behind else False,
        relative_to_axis=FacingAddresseeAxis(goal_reference),
    )

    return Phase1SituationTemplate(
        f"{agent.handle}-rolls-{theme.handle}-{value}-{goal_reference.handle}",
        salient_object_variables=[agent, theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                ROLL,
                argument_roles_to_fillers=[(AGENT, agent), (THEME, theme)],
                auxiliary_variable_bindings=[(ROLL_SURFACE_AUXILIARY, surface)],
            )
        ],
        constraining_relations=[bigger_than(agent, theme)],
        after_action_relations=flatten_relations(
            far(theme, goal_reference, direction=direction)
            if is_distal
            else near(theme, goal_reference, direction=direction)
        ),
        gazed_objects=[theme],
    )


def _x_rolls_y_over_under_z_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    surface: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_over: bool,
) -> Phase1SituationTemplate:
    value = "over" if is_over else "under"
    return Phase1SituationTemplate(
        f"{agent.handle}-rolls-{theme.handle}-{value}-{goal_reference}",
        salient_object_variables=[agent, theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                ROLL,
                argument_roles_to_fillers=[(AGENT, agent), (THEME, theme)],
                auxiliary_variable_bindings=[(ROLL_SURFACE_AUXILIARY, surface)],
            )
        ],
        constraining_relations=[bigger_than(agent, theme)],
        after_action_relations=flatten_relations(
            above(theme, goal_reference) if is_over else above(goal_reference, theme)
        ),
        gazed_objects=[theme],
    )


# FALL templates


def _fall_on_template(
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: ImmutableSet[TemplateObjectVariable],
    *,
    syntax_hints: ImmutableSet[str],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{theme.handle}-falls-(down)-on-{goal_reference.handle}",
        salient_object_variables=[theme, goal_reference],
        background_object_variables=background,
        actions=[Action(FALL, argument_roles_to_fillers=[(THEME, theme)])],
        after_action_relations=flatten_relations(on(theme, goal_reference)),
        constraining_relations=[bigger_than(goal_reference, theme)],
        syntax_hints=syntax_hints,
    )


def _fall_in_template(
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: ImmutableSet[TemplateObjectVariable],
    *,
    syntax_hints: ImmutableSet[str],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{theme.handle}-falls-(down)-in-{goal_reference.handle}",
        salient_object_variables=[theme, goal_reference],
        background_object_variables=background,
        actions=[Action(FALL, argument_roles_to_fillers=[(THEME, theme)])],
        after_action_relations=flatten_relations(inside(theme, goal_reference)),
        constraining_relations=[bigger_than(goal_reference, theme)],
        syntax_hints=syntax_hints,
    )


def _fall_beside_template(
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: ImmutableSet[TemplateObjectVariable],
    *,
    syntax_hints: ImmutableSet[str],
    is_right: bool,
) -> Phase1SituationTemplate:
    direction = Direction(
        positive=is_right,
        relative_to_axis=HorizontalAxisOfObject(goal_reference, index=0),
    )
    return Phase1SituationTemplate(
        f"{theme.handle}-falls-(down)-beside-{goal_reference.handle}",
        salient_object_variables=[theme, goal_reference],
        background_object_variables=background,
        actions=[Action(FALL, argument_roles_to_fillers=[(THEME, theme)])],
        after_action_relations=flatten_relations(
            near(theme, goal_reference, direction=direction)
        ),
        syntax_hints=syntax_hints,
    )


def _fall_in_front_of_behind_template(
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: ImmutableSet[TemplateObjectVariable],
    *,
    syntax_hints: ImmutableSet[str],
    is_distal: bool,
    is_in_front: bool,
) -> Phase1SituationTemplate:
    direction = Direction(
        positive=is_in_front, relative_to_axis=FacingAddresseeAxis(goal_reference)
    )
    return Phase1SituationTemplate(
        f"{theme.handle}-falls-(down)-in-front-of-behind-{goal_reference.handle}",
        salient_object_variables=[theme, goal_reference],
        background_object_variables=background,
        actions=[Action(FALL, argument_roles_to_fillers=[(THEME, theme)])],
        after_action_relations=flatten_relations(
            far(theme, goal_reference, direction=direction)
            if is_distal
            else near(theme, goal_reference, direction=direction)
        ),
        syntax_hints=syntax_hints,
    )


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


def _make_go_with_prepositions(num_samples: int = 5, *, noise_objects: int = 0):
    agent = standard_object("agent", THING, required_properties=[ANIMATE])
    goal_object = standard_object("goal_object")
    goal_object_hollow = standard_object(
        "goal_object_hollow", required_properties=[HOLLOW]
    )
    path_object = standard_object(
        "path_object",
        required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM, HAS_SPACE_UNDER],
    )

    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )

    return phase1_instances(
        "Go + PP",
        chain(
            # To
            flatten(
                [
                    sampled(
                        _go_to_template(agent, goal_object, background),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                ]
            ),
            # In
            flatten(
                [
                    sampled(
                        _go_in_template(agent, goal_object_hollow, background),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                ]
            ),
            # Beside
            flatten(
                [
                    sampled(
                        _go_beside_template(
                            agent, goal_object, background, is_right=is_right
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                    for is_right in BOOL_SET
                ]
            ),
            # Behind & In Front Of
            flatten(
                [
                    sampled(
                        _go_behind_in_front_template(
                            agent,
                            goal_object,
                            background,
                            is_distal=is_distal,
                            is_behind=is_behind,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                    for is_distal in BOOL_SET
                    for is_behind in BOOL_SET
                ]
            ),
            # Over & Under
            flatten(
                [
                    sampled(
                        _go_over_under_template(
                            agent,
                            goal_object,
                            background,
                            is_distal=is_distal,
                            is_over=is_over,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                    for is_distal in BOOL_SET
                    for is_over in BOOL_SET
                ]
            ),
            # Behind & In Front Of Paths
            flatten(
                [
                    sampled(
                        _go_behind_in_front_path_template(
                            agent,
                            goal_object,
                            path_object,
                            background,
                            is_behind=is_behind,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                    for is_behind in BOOL_SET
                ]
            ),
            # Over & Under Paths
            flatten(
                [
                    sampled(
                        _go_over_under_path_template(
                            agent, goal_object, path_object, background, is_over=is_over
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                    for is_over in BOOL_SET
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


def _make_roll_with_prepositions(num_samples: int = 5, *, noise_objects: int = 0):
    agent = standard_object("agent", THING, required_properties=[ANIMATE])
    goal_object = standard_object("goal_object")
    goal_object_hollow = standard_object(
        "goal_object_hollow", required_properties=[HOLLOW]
    )
    theme = standard_object("rollee", required_properties=[ROLLABLE])
    ground = standard_object("ground", root_node=GROUND)
    roll_surface = standard_object(
        "rollable_surface", required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
    )
    noise_objects_immutable: Iterable[TemplateObjectVariable] = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )
    surfaces: Iterable[TemplateObjectVariable] = immutableset([ground, roll_surface])
    all_objects_mutable = [ground, roll_surface]
    all_objects_mutable.extend(noise_objects_immutable)
    all_object: Iterable[TemplateObjectVariable] = immutableset(all_objects_mutable)

    return phase1_instances(
        "Roll + PP",
        chain(
            # X rolls beside Y
            flatten(
                [
                    sampled(
                        _x_roll_beside_y_template(
                            agent,
                            goal_object,
                            make_background([roll_surface], all_object),
                            ground,
                            is_right=is_right,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                    for is_right in BOOL_SET
                ]
            ),
            # X rolls behind/In front of Y
            flatten(
                [
                    sampled(
                        _x_roll_behind_in_front_y_template(
                            agent,
                            goal_object,
                            make_background([roll_surface], all_object),
                            ground,
                            is_distal=is_distal,
                            is_behind=is_behind,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                    for is_distal in BOOL_SET
                    for is_behind in BOOL_SET
                ]
            ),
            # X rolls under Y
            flatten(
                [
                    sampled(
                        _x_roll_under_y_template(
                            agent,
                            goal_object,
                            make_background([roll_surface], all_object),
                            ground,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                ]
            ),
            # X rolls Y in Z
            flatten(
                [
                    sampled(
                        _x_roll_y_in_z_template(
                            agent,
                            theme,
                            goal_object_hollow,
                            ground,
                            make_background([roll_surface], all_object),
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                ]
            ),
            # X rolls Y beside Z
            flatten(
                [
                    sampled(
                        _x_roll_y_beside_z_template(
                            agent,
                            theme,
                            goal_object,
                            ground,
                            make_background([roll_surface], all_object),
                            is_right=is_right,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                    for is_right in BOOL_SET
                ]
            ),
            # X rolls Y behind/In front of Z
            flatten(
                [
                    sampled(
                        _x_roll_y_behind_in_front_z_template(
                            agent,
                            theme,
                            goal_object,
                            ground,
                            make_background([roll_surface], all_object),
                            is_distal=is_distal,
                            is_behind=is_behind,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                    for is_distal in BOOL_SET
                    for is_behind in BOOL_SET
                ]
            ),
            # X rolls Y over/under Z
            flatten(
                [
                    sampled(
                        _x_rolls_y_over_under_z_template(
                            agent,
                            theme,
                            goal_object,
                            ground,
                            make_background([ground], all_object),
                            is_over=is_over,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                    for is_over in BOOL_SET
                ]
            ),
            # X rolls (Y) over/under Z - As Goal
            flatten(
                [
                    sampled(
                        _x_rolls_y_over_under_z_template(
                            agent,
                            theme,
                            surface,
                            surface,
                            noise_objects_immutable,
                            is_over=is_over,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                    for is_over in BOOL_SET
                    for surface in surfaces
                ]
            ),
        ),
    )


def _make_fall_with_prepositions(
    num_samples: int = 5, *, noise_objects: int = 0
) -> Phase1InstanceGroup:
    theme = standard_object("theme", THING)
    goal_reference = standard_object("goal_reference", THING)
    goal_on = standard_object(
        "goal_on", THING, required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
    )
    goal_in = standard_object("goal_in", THING, required_properties=[HOLLOW])
    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )
    syntax_hints_options = ([], [USE_ADVERBIAL_PATH_MODIFIER])  # type: ignore
    return phase1_instances(
        "Fall + PP",
        chain(
            # on
            flatten(
                [
                    sampled(
                        _fall_on_template(
                            theme, goal_on, background, syntax_hints=syntax_hints
                        ),
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
                        _fall_in_template(
                            theme, goal_in, background, syntax_hints=syntax_hints
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                    for syntax_hints in syntax_hints_options
                ]
            ),
            # beside
            flatten(
                [
                    sampled(
                        _fall_beside_template(
                            theme,
                            goal_reference,
                            background,
                            syntax_hints=syntax_hints,
                            is_right=is_right,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                    for syntax_hints in syntax_hints_options
                    for is_right in BOOL_SET
                ]
            ),
            # in front of, behind
            flatten(
                [
                    sampled(
                        _fall_in_front_of_behind_template(
                            theme,
                            goal_reference,
                            background,
                            syntax_hints=syntax_hints,
                            is_distal=is_distal,
                            is_in_front=is_in_front,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                    for syntax_hints in syntax_hints_options
                    for is_distal in BOOL_SET
                    for is_in_front in BOOL_SET
                ]
            ),
        ),
    )


def make_verb_with_dynamic_prepositions_curriculum(
    num_samples: int = 5, *, num_noise_objects: int = 0
):
    return [
        _make_push_with_prepositions(num_samples, noise_objects=num_noise_objects),
        _make_go_with_prepositions(num_samples, noise_objects=num_noise_objects),
        _make_sit_with_prepositions(num_samples, noise_objects=num_noise_objects),
        _make_roll_with_prepositions(num_samples, noise_objects=num_noise_objects),
        _make_fall_with_prepositions(num_samples, noise_objects=num_noise_objects),
    ]
