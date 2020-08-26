from immutablecollections import immutableset
from itertools import chain
from typing import Iterable, Sequence, Optional
from adam.language.language_generator import LanguageGenerator
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.language.dependency import LinearizedDependencyTree
from more_itertools import flatten

from adam.axes import (
    HorizontalAxisOfObject,
    FacingAddresseeAxis,
    GRAVITATIONAL_AXIS_FUNCTION,
)
from adam.curriculum.curriculum_utils import (
    standard_object,
    phase1_instances,
    PHASE1_CHOOSER_FACTORY,
    Phase1InstanceGroup,
    make_background,
    body_part_object,
    GROUND_OBJECT_TEMPLATE,
    make_noise_objects,
)
from adam.language_specific.english.english_language_generator import (
    USE_ADVERBIAL_PATH_MODIFIER,
    IGNORE_GOAL,
)
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
    HAS_SPACE_UNDER,
    PUSH,
    TO,
    THEME,
    PUSH_SURFACE_AUX,
    ANIMATE,
    INANIMATE_OBJECT,
    CAN_HAVE_THINGS_RESTING_ON_THEM,
    GO,
    ROLL,
    ROLL_SURFACE_AUXILIARY,
    ROLLABLE,
    GROUND,
    above,
    on,
    bigger_than,
    near,
    far,
    inside,
    TAKE,
    PUT,
    PERSON,
    THROW,
    THROW_GOAL,
    strictly_above,
    MOVE,
    contacts,
    SELF_MOVING,
    JUMP_INITIAL_SUPPORTER_AUX,
    CAN_JUMP,
    JUMP,
    FLY,
    CAN_FLY,
    PUSH_GOAL,
    COME,
)
from adam.ontology import THING, IS_SPEAKER, IS_ADDRESSEE, IN_REGION
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
    EXTERIOR_BUT_IN_CONTACT,
    TOWARD,
    AWAY_FROM,
)
from adam.relation import flatten_relations, Relation
from adam.relation_dsl import negate
from adam.situation import Action
from adam.situation.templates.phase1_situation_templates import (
    _fly_over_template,
    _fly_under_template,
    _jump_over_template,
    _put_in_template,
    _put_on_template,
    _put_on_body_part_template,
    _go_in_template,
    _go_under_template,
    _go_to_template,
)
from adam.situation.templates.phase1_templates import (
    TemplateObjectVariable,
    Phase1SituationTemplate,
    sampled,
)

BOOL_SET = immutableset([True, False])

# TODO: fix https://github.com/isi-vista/adam/issues/917 which causes us to have to specify that we don't wish to include ME_HACK and YOU_HACK in our curriculum design
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
        after_action_relations=[near(theme, goal_reference)],
        constraining_relations=flatten_relations(
            bigger_than(surface, [agent, goal_reference])
        ),
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
        after_action_relations=[near(theme, goal_reference)],
        constraining_relations=flatten_relations(
            [
                bigger_than(surface, [agent, goal_reference]),
                bigger_than(goal_reference, theme),
            ]
        ),
    )


def _push_under_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    surface: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
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
        after_action_relations=[near(theme, goal_reference)],
        constraining_relations=flatten_relations(
            [
                bigger_than(surface, [agent, goal_reference]),
                bigger_than(goal_reference, theme),
            ]
        ),
    )


def _push_beside_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    surface: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
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
        after_action_relations=[near(theme, goal_reference)],
        constraining_relations=flatten_relations(
            bigger_than(surface, [agent, goal_reference])
        ),
    )


def _push_in_front_of_behind_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    surface: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
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
        after_action_relations=[near(theme, goal_reference)],
        constraining_relations=flatten_relations(
            bigger_than(surface, [agent, goal_reference])
        ),
    )


def _push_towards_away_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    spatial_reference: TemplateObjectVariable,
    surface: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_ending_proximal: bool,
    is_towards: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-pushes-{theme.handle}-toward/away-{spatial_reference.handle}",
        salient_object_variables=[agent, theme, spatial_reference],
        background_object_variables=background,
        actions=[
            Action(
                PUSH,
                argument_roles_to_fillers=[(AGENT, agent), (THEME, theme)],
                auxiliary_variable_bindings=[
                    (PUSH_SURFACE_AUX, surface),
                    (
                        PUSH_GOAL,
                        Region(
                            spatial_reference,
                            distance=PROXIMAL if is_ending_proximal else DISTAL,
                        ),
                    ),
                ],
                during=DuringAction(
                    continuously=[on(theme, surface)],
                    objects_to_paths=[
                        (
                            theme,
                            SpatialPath(
                                operator=TOWARD if is_towards else AWAY_FROM,
                                reference_object=spatial_reference,
                                # The reference axis should explicitly be the axis
                                # on which movement is occuring. Implying this axis
                                # May not always be 100% correct because a person
                                # doesn't always face the way they are walking
                                reference_axis=HorizontalAxisOfObject(agent, 1),
                            ),
                        )
                    ],
                ),
            )
        ],
        before_action_relations=[
            far([agent, theme], spatial_reference)
            if is_ending_proximal
            else near([agent, theme], spatial_reference)
        ],
    )


def _push_out_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    spatial_reference: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    surface: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_distal: bool,
) -> Phase1SituationTemplate:
    inside_relation = inside([agent, theme], spatial_reference)
    return Phase1SituationTemplate(
        f"{agent.handle}-push-{theme.handle}-out-of-{spatial_reference.handle}",
        salient_object_variables=[agent, theme, spatial_reference],
        background_object_variables=background,
        actions=[
            Action(
                PUSH,
                argument_roles_to_fillers=[(AGENT, agent), (THEME, theme)],
                auxiliary_variable_bindings=[
                    (
                        PUSH_GOAL,
                        Region(
                            goal_reference, distance=DISTAL if is_distal else PROXIMAL
                        ),
                    ),
                    (PUSH_SURFACE_AUX, surface),
                ],
            )
        ],
        before_action_relations=flatten_relations(inside_relation),
        after_action_relations=flatten_relations(
            [relation.negated_copy() for relation in inside_relation]
        ),
        constraining_relations=flatten_relations(
            [bigger_than(spatial_reference, [theme, agent]), bigger_than(agent, theme)]
        ),
        syntax_hints=[IGNORE_GOAL],
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
        after_action_relations=[near(agent, goal_object)],
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
        after_action_relations=[near(agent, goal_object)],
    )


def _go_over_template(
    agent: TemplateObjectVariable,
    goal_object: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_distal: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"go_over-{agent.handle}-over-{goal_object.handle}",
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
                            direction=GRAVITATIONAL_UP,
                        ),
                    ),
                ],
            )
        ],
        after_action_relations=[near(agent, goal_object)],
    )


def _go_behind_in_front_path_template(
    agent: TemplateObjectVariable,
    goal_object: TemplateObjectVariable,
    path_object: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_behind: bool,
    is_near_path: bool,
    is_near_goal: bool,
) -> Phase1SituationTemplate:
    additional_background = [goal_object, path_object]
    additional_background.extend(background)
    total_background = immutableset(additional_background)
    handle = "behind" if is_behind else "in-front-of"
    return Phase1SituationTemplate(
        f"go_{handle}-{agent.handle}-{handle}-{goal_object.handle}-via-{path_object.handle}",
        salient_object_variables=[agent, goal_object],
        background_object_variables=total_background,
        actions=[
            Action(
                GO,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (
                        GOAL,
                        Region(
                            goal_object, distance=PROXIMAL if is_near_goal else DISTAL
                        ),
                    ),
                ],
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
                    # HACK - This is a hack for 'in front of' and 'behind'
                    at_some_point=[
                        near(
                            agent,
                            goal_object,
                            direction=Direction(
                                positive=False if is_behind else True,
                                relative_to_axis=FacingAddresseeAxis(agent),
                            ),
                        )
                        if is_near_path
                        else far(
                            agent,
                            goal_object,
                            direction=Direction(
                                positive=False if is_behind else True,
                                relative_to_axis=FacingAddresseeAxis(agent),
                            ),
                        )
                    ],
                ),
            )
        ],
        gazed_objects=[agent],
        syntax_hints=[IGNORE_GOAL],
    )


def _go_over_under_path_template(
    agent: TemplateObjectVariable,
    goal_object: TemplateObjectVariable,
    path_object: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_over: bool,
    is_near_goal: bool,
) -> Phase1SituationTemplate:
    additional_background = [goal_object, path_object]
    additional_background.extend(background)
    total_background = immutableset(additional_background)
    handle = "over" if is_over else "under"
    return Phase1SituationTemplate(
        f"go_{handle}-{agent.handle}-{handle}-{goal_object.handle}-via-{path_object.handle}",
        salient_object_variables=[agent, path_object],
        background_object_variables=total_background,
        actions=[
            Action(
                GO,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (
                        GOAL,
                        Region(
                            goal_object, distance=PROXIMAL if is_near_goal else DISTAL
                        ),
                    ),
                ],
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
        syntax_hints=[IGNORE_GOAL],
    )


def _go_towards_away_template(
    agent: TemplateObjectVariable,
    spatial_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_toward: bool,
    is_near_goal: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-go-toward_away-{spatial_reference.handle}",
        salient_object_variables=[agent, spatial_reference],
        background_object_variables=background,
        actions=[
            Action(
                GO,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (
                        GOAL,
                        Region(
                            spatial_reference,
                            distance=PROXIMAL if is_near_goal else DISTAL,
                        ),
                    ),
                ],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            agent,
                            SpatialPath(
                                operator=TOWARD if is_toward else AWAY_FROM,
                                reference_object=spatial_reference,
                                reference_axis=HorizontalAxisOfObject(agent, 1),
                            ),
                        )
                    ]
                ),
            )
        ],
        before_action_relations=[
            far(agent, spatial_reference) if is_toward else near(agent, spatial_reference)
        ],
        syntax_hints=[IGNORE_GOAL],
    )


def _x_go_out_y_template(
    agent: TemplateObjectVariable,
    spatial_reference: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_distal: bool,
) -> Phase1SituationTemplate:
    computed_background = [goal_reference]
    computed_background.extend(background)
    inside_relation = inside(agent, spatial_reference)
    return Phase1SituationTemplate(
        f"{agent.handle}-go-out-of-{spatial_reference.handle}",
        salient_object_variables=[agent, spatial_reference],
        background_object_variables=computed_background,
        actions=[
            Action(
                GO,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (
                        GOAL,
                        Region(
                            goal_reference, distance=DISTAL if is_distal else PROXIMAL
                        ),
                    ),
                ],
            )
        ],
        before_action_relations=flatten_relations(inside_relation),
        after_action_relations=flatten_relations(
            relation.negated_copy() for relation in inside_relation
        ),
        constraining_relations=flatten_relations(bigger_than(spatial_reference, agent)),
        syntax_hints=[IGNORE_GOAL],
    )


# THROW templates


def _throw_to_template(
    # "Mom throws a ball to a chair"
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-throws-{theme.handle}-to-{goal_reference.handle}",
        salient_object_variables=[agent, theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                THROW,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (THEME, theme),
                    (GOAL, Region(goal_reference, distance=PROXIMAL)),
                ],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            theme,
                            SpatialPath(
                                TO,
                                reference_object=Region(
                                    goal_reference, distance=PROXIMAL
                                ),
                            ),
                        )
                    ]
                ),
            )
        ],
        after_action_relations=[near(theme, goal_reference)],
        constraining_relations=flatten_relations(bigger_than(agent, theme)),
        gazed_objects=[theme],
    )


def _throw_in_template(
    # "Dad throws a ball (and it lands) in a box"
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-throws-{theme.handle}-in-{goal_reference.handle}",
        salient_object_variables=[agent, theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                THROW,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (THEME, theme),
                    (GOAL, Region(goal_reference, distance=INTERIOR)),
                ],
            )
        ],
        constraining_relations=flatten_relations(
            bigger_than([agent, goal_reference], theme)
        ),
        after_action_relations=[near(theme, goal_reference)],
        gazed_objects=[theme],
    )


def _throw_on_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-throws-{theme.handle}-on-{goal_reference.handle}",
        salient_object_variables=[agent, theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                THROW,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (THEME, theme),
                    (
                        GOAL,
                        Region(
                            goal_reference,
                            direction=GRAVITATIONAL_UP,
                            distance=EXTERIOR_BUT_IN_CONTACT,
                        ),
                    ),
                ],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            theme,
                            SpatialPath(
                                None, reference_object=goal_reference, properties=[]
                            ),
                        )
                    ]
                ),
            )
        ],
        after_action_relations=[on(theme, goal_reference)],
        constraining_relations=flatten_relations(bigger_than(agent, theme)),
        gazed_objects=[theme],
    )


def _throw_beside_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_right: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-throws-{theme.handle}-beside-{goal_reference.handle}",
        salient_object_variables=[agent, theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                THROW,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (THEME, theme),
                    (
                        GOAL,
                        Region(
                            goal_reference,
                            direction=Direction(
                                positive=is_right,
                                relative_to_axis=HorizontalAxisOfObject(
                                    goal_reference, index=0
                                ),
                            ),
                            distance=PROXIMAL,
                        ),
                    ),
                ],
            )
        ],
        constraining_relations=flatten_relations(bigger_than(agent, theme)),
        after_action_relations=[near(theme, goal_reference)],
        gazed_objects=[theme],
    )


def _throw_in_front_of_behind_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_distal: bool,
    is_in_front: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-throws-{theme.handle}-in-front-of-behind-{goal_reference.handle}",
        salient_object_variables=[agent, theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                THROW,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (THEME, theme),
                    (
                        GOAL,
                        Region(
                            goal_reference,
                            direction=Direction(
                                positive=is_in_front,
                                relative_to_axis=FacingAddresseeAxis(goal_reference),
                            ),
                            distance=DISTAL if is_distal else PROXIMAL,
                        ),
                    ),
                ],
            )
        ],
        after_action_relations=[near(theme, goal_reference)],
        constraining_relations=flatten_relations(bigger_than(agent, theme)),
        gazed_objects=[theme],
    )


def _throw_under_template(
    # "A baby throws a cup under a table"
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_distal: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-throws-{theme.handle}-under-{goal_reference.handle}",
        salient_object_variables=[agent, theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                THROW,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (THEME, theme),
                    (
                        GOAL,
                        Region(
                            goal_reference,
                            direction=GRAVITATIONAL_DOWN,
                            distance=DISTAL if is_distal else PROXIMAL,
                        ),
                    ),
                ],
            )
        ],
        after_action_relations=[near(theme, goal_reference)],
        constraining_relations=flatten_relations(
            bigger_than([agent, goal_reference], theme)
        ),
        gazed_objects=[theme],
    )


def _throw_path_over_template(
    # "A baby throws a truck over a table (and lands on the other side)"
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    object_in_path: TemplateObjectVariable,
    implicit_goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-throws-{theme.handle}-with-path-over-{object_in_path.handle}",
        salient_object_variables=[agent, theme, object_in_path],
        background_object_variables=background,
        actions=[
            Action(
                THROW,
                argument_roles_to_fillers=[(AGENT, agent), (THEME, theme)],
                auxiliary_variable_bindings=[
                    (THROW_GOAL, Region(implicit_goal_reference, distance=PROXIMAL))
                ],
                during=DuringAction(
                    at_some_point=[strictly_above(theme, object_in_path)]
                ),
            )
        ],
        # hack of ordering relation for English language generator
        after_action_relations=[near(implicit_goal_reference, theme)],
        constraining_relations=flatten_relations(bigger_than(agent, theme)),
        gazed_objects=[theme],
    )


def _throw_path_under_template(
    # "A baby throws a truck under a table (and lands on the other side)"
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    object_in_path: TemplateObjectVariable,
    implicit_goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_distal: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-throws-{theme.handle}-with-path-under-{object_in_path.handle}",
        salient_object_variables=[agent, theme, object_in_path],
        background_object_variables=background,
        actions=[
            Action(
                THROW,
                argument_roles_to_fillers=[(AGENT, agent), (THEME, theme)],
                auxiliary_variable_bindings=[
                    (
                        THROW_GOAL,
                        Region(
                            implicit_goal_reference,
                            distance=DISTAL if is_distal else PROXIMAL,
                        ),
                    )
                ],
                during=DuringAction(
                    at_some_point=[strictly_above(object_in_path, theme)]
                ),
            )
        ],
        after_action_relations=[near(theme, implicit_goal_reference)],
        constraining_relations=flatten_relations(
            bigger_than([agent, object_in_path], theme)
        ),
        gazed_objects=[theme],
    )


def _x_throws_y_to_z_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    recipient: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-throws-{theme.handle}-to-{recipient.handle}",
        salient_object_variables=[agent, theme, recipient],
        background_object_variables=background,
        actions=[
            Action(
                THROW,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (THEME, theme),
                    # hack since English language generator doesn't generate correct language for GOAL_MANIPULATOR currently
                    (GOAL, Region(recipient, distance=PROXIMAL)),
                ],
            )
        ],
        after_action_relations=[near(theme, recipient)],
        constraining_relations=flatten_relations(bigger_than([agent, recipient], theme)),
        gazed_objects=[theme],
    )


def _throw_towards_away_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    spatial_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_towards: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-throws-{theme.handle}-toward/away_from-{spatial_reference.handle}",
        salient_object_variables=[agent, theme, spatial_reference],
        background_object_variables=background,
        actions=[
            Action(
                THROW,
                argument_roles_to_fillers=[(AGENT, agent), (THEME, theme)],
                auxiliary_variable_bindings=[
                    (
                        THROW_GOAL,
                        Region(
                            spatial_reference, distance=PROXIMAL if is_towards else DISTAL
                        ),
                    )
                ],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            theme,
                            SpatialPath(
                                operator=TOWARD if is_towards else AWAY_FROM,
                                reference_object=spatial_reference,
                                reference_axis=HorizontalAxisOfObject(theme, 1),
                            ),
                        )
                    ]
                ),
            )
        ],
        constraining_relations=flatten_relations(bigger_than(agent, theme)),
        gazed_objects=[theme],
    )


# SIT templates


def _sit_on_template(
    agent: TemplateObjectVariable,
    surface: TemplateObjectVariable,
    seat: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    syntax_hints: Iterable[str],
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
        constraining_relations=flatten_relations(
            [bigger_than(surface, seat), bigger_than(seat, agent)]
        ),
        syntax_hints=syntax_hints,
    )


def _sit_in_template(
    agent: TemplateObjectVariable,
    surface: TemplateObjectVariable,
    seat: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    syntax_hints: Iterable[str],
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
        constraining_relations=flatten_relations(
            [bigger_than(surface, seat), bigger_than(seat, agent)]
        ),
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
        constraining_relations=flatten_relations(
            [bigger_than([agent, goal_reference], theme)]
        ),
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
        constraining_relations=flatten_relations([bigger_than(agent, theme)]),
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
        constraining_relations=flatten_relations([bigger_than(agent, theme)]),
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
        constraining_relations=flatten_relations([bigger_than(agent, theme)]),
        after_action_relations=flatten_relations(
            above(theme, goal_reference) if is_over else above(goal_reference, theme)
        ),
        gazed_objects=[theme],
    )


def _x_rolls_towards_away_from_y_template(
    agent: TemplateObjectVariable,
    spatial_reference: TemplateObjectVariable,
    surface: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_toward: bool,
) -> Phase1SituationTemplate:
    value = "towards" if is_toward else "away from"
    return Phase1SituationTemplate(
        f"{agent.handle}-rolls-{value}-{spatial_reference.handle}",
        salient_object_variables=[agent, spatial_reference],
        background_object_variables=background,
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
                                operator=TOWARD if is_toward else AWAY_FROM,
                                reference_object=spatial_reference,
                                reference_axis=HorizontalAxisOfObject(agent, 1),
                            ),
                        )
                    ]
                ),
            )
        ],
        before_action_relations=flatten_relations(
            far(agent, spatial_reference) if is_toward else near(agent, spatial_reference)
        ),
        after_action_relations=flatten_relations(
            near(agent, spatial_reference) if is_toward else immutableset()
        ),
    )


def _x_rolls_y_towards_away_from_z_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    spatial_reference: TemplateObjectVariable,
    surface: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_toward: bool,
) -> Phase1SituationTemplate:
    value = "towards" if is_toward else "away from"
    return Phase1SituationTemplate(
        f"{agent.handle}-rolls-{theme.handle}-{value}-{spatial_reference.handle}",
        salient_object_variables=[agent, spatial_reference, theme],
        background_object_variables=background,
        actions=[
            Action(
                ROLL,
                argument_roles_to_fillers=[(AGENT, agent), (THEME, theme)],
                auxiliary_variable_bindings=[(ROLL_SURFACE_AUXILIARY, surface)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            agent,
                            SpatialPath(
                                operator=TOWARD if is_toward else AWAY_FROM,
                                reference_object=spatial_reference,
                                reference_axis=HorizontalAxisOfObject(agent, 1),
                            ),
                        )
                    ]
                ),
            )
        ],
        constraining_relations=flatten_relations(bigger_than(agent, theme)),
        before_action_relations=flatten_relations(
            far([agent, theme], spatial_reference)
            if is_toward
            else near([agent, theme], spatial_reference)
        ),
        gazed_objects=[theme],
    )


def _x_rolls_out_z_template(
    agent: TemplateObjectVariable,
    spatial_reference: TemplateObjectVariable,
    surface: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    inside_relation = inside(agent, spatial_reference)
    return Phase1SituationTemplate(
        f"{agent.handle}-rolls-out-of-{spatial_reference.handle}",
        salient_object_variables=[agent, spatial_reference],
        background_object_variables=background,
        actions=[
            Action(
                ROLL,
                argument_roles_to_fillers=[(AGENT, agent)],
                auxiliary_variable_bindings=[(ROLL_SURFACE_AUXILIARY, surface)],
            )
        ],
        before_action_relations=flatten_relations(inside_relation),
        after_action_relations=flatten_relations(
            [relation.negated_copy() for relation in inside_relation]
        ),
        constraining_relations=flatten_relations(bigger_than(spatial_reference, agent)),
    )


def _x_rolls_y_out_of_z_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    spatial_reference: TemplateObjectVariable,
    surface: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    inside_relation = inside([agent, theme], spatial_reference)
    return Phase1SituationTemplate(
        f"{agent.handle}-rolls-{theme.handle}-out-of-{spatial_reference.handle}",
        salient_object_variables=[agent, theme, spatial_reference],
        background_object_variables=background,
        actions=[
            Action(
                ROLL,
                argument_roles_to_fillers=[(AGENT, agent), (THEME, theme)],
                auxiliary_variable_bindings=[(ROLL_SURFACE_AUXILIARY, surface)],
            )
        ],
        before_action_relations=flatten_relations(inside_relation),
        after_action_relations=flatten_relations(
            [relation.negated_copy() for relation in inside_relation]
        ),
        constraining_relations=flatten_relations(
            bigger_than(spatial_reference, [agent, theme])
        ),
        syntax_hints=[IGNORE_GOAL],
    )


# TAKE templates


def _take_to_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-takes-{theme.handle}-to-{goal_reference.handle}",
        salient_object_variables=[agent, theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(TAKE, argument_roles_to_fillers=[(AGENT, agent), (THEME, theme)])
        ],
        after_action_relations=flatten_relations(near(theme, goal_reference)),
        constraining_relations=flatten_relations(bigger_than(agent, theme)),
    )


# FALL templates


def _fall_on_template(
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    syntax_hints: Iterable[str],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{theme.handle}-falls-(down)-on-{goal_reference.handle}",
        salient_object_variables=[theme, goal_reference],
        background_object_variables=background,
        actions=[Action(FALL, argument_roles_to_fillers=[(THEME, theme)])],
        after_action_relations=flatten_relations(on(theme, goal_reference)),
        constraining_relations=flatten_relations(bigger_than(goal_reference, theme)),
        syntax_hints=syntax_hints,
    )


def _fall_in_template(
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    syntax_hints: Iterable[str],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{theme.handle}-falls-(down)-in-{goal_reference.handle}",
        salient_object_variables=[theme, goal_reference],
        background_object_variables=background,
        actions=[Action(FALL, argument_roles_to_fillers=[(THEME, theme)])],
        after_action_relations=flatten_relations(inside(theme, goal_reference)),
        constraining_relations=flatten_relations(bigger_than(goal_reference, theme)),
        syntax_hints=syntax_hints,
    )


def _fall_beside_template(
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    syntax_hints: Iterable[str],
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
    background: Iterable[TemplateObjectVariable],
    *,
    syntax_hints: Iterable[str],
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


def _fall_toward_away_from_template(
    theme: TemplateObjectVariable,
    spatial_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    syntax_hints: Iterable[str],
    is_toward: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{theme.handle}-falls-towards-away_from-{spatial_reference.handle}",
        salient_object_variables=[theme, spatial_reference],
        background_object_variables=background,
        actions=[
            Action(
                FALL,
                argument_roles_to_fillers=[(THEME, theme)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            theme,
                            SpatialPath(
                                operator=TOWARD if is_toward else AWAY_FROM,
                                reference_object=spatial_reference,
                                reference_axis=HorizontalAxisOfObject(theme, 1),
                            ),
                        )
                    ]
                ),
            )
        ],
        before_action_relations=flatten_relations(
            far(theme, spatial_reference) if is_toward else near(theme, spatial_reference)
        ),
        after_action_relations=flatten_relations(
            near(theme, spatial_reference) if is_toward else immutableset()
        ),
        syntax_hints=syntax_hints,
    )


# PUT templates


def _put_under_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_distal: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-puts-{theme.handle}-under-{goal_reference.handle}",
        salient_object_variables=[agent, theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                PUT,
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
            )
        ],
        constraining_relations=flatten_relations(
            bigger_than([agent, goal_reference], theme)
        ),
    )


def _put_beside_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_right: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-puts-{theme.handle}-beside-{goal_reference.handle}",
        salient_object_variables=[agent, theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                PUT,
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
            )
        ],
        constraining_relations=flatten_relations([bigger_than(agent, theme)]),
    )


def _put_in_front_of_behind_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_distal: bool,
    is_in_front: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-puts-{theme.handle}-in-front-of-behind-{goal_reference.handle}",
        salient_object_variables=[agent, theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                PUT,
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
            )
        ],
        constraining_relations=flatten_relations([bigger_than(agent, theme)]),
    )


def _x_move_beside_y_template(
    # "A baby moves beside a car"
    agent: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_right: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-moves-beside-{goal_reference.handle}",
        salient_object_variables=[agent, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                MOVE,
                argument_roles_to_fillers=[
                    (AGENT, agent),
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
            )
        ],
        after_action_relations=[near(agent, goal_reference)],
    )


def _x_move_in_front_of_behind_y_template(
    # "Mom moves in front of a house"
    agent: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_distal: bool,
    is_in_front: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-moves-in-front-of-behind-{goal_reference.handle}",
        salient_object_variables=[agent, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                MOVE,
                argument_roles_to_fillers=[
                    (AGENT, agent),
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
            )
        ],
        after_action_relations=[near(agent, goal_reference)],
    )


def _x_move_under_y_template(
    # "A dog moves under a chair"
    agent: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_distal: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-moves-under-{goal_reference.handle}",
        salient_object_variables=[agent, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                MOVE,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (
                        GOAL,
                        Region(
                            goal_reference,
                            direction=GRAVITATIONAL_DOWN,
                            distance=DISTAL if is_distal else PROXIMAL,
                        ),
                    ),
                ],
            )
        ],
        constraining_relations=flatten_relations(bigger_than(goal_reference, agent)),
        after_action_relations=[near(agent, goal_reference)],
    )


def _x_move_y_in_z_template(
    # "Dad moves a book in a box"
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-moves-{theme.handle}-in-{goal_reference.handle}",
        salient_object_variables=[agent, theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                MOVE,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (THEME, theme),
                    (GOAL, Region(goal_reference, distance=INTERIOR)),
                ],
                during=DuringAction(
                    continuously=flatten_relations(contacts(agent, theme))
                ),
            )
        ],
        constraining_relations=flatten_relations(bigger_than(goal_reference, theme)),
        after_action_relations=[near(agent, goal_reference)],
    )


def _x_move_y_on_z_template(
    # "Mom moves a cookie on a table"
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-moves-{theme.handle}-on-{goal_reference.handle}",
        salient_object_variables=[agent, theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                MOVE,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (THEME, theme),
                    (
                        GOAL,
                        Region(
                            goal_reference,
                            distance=EXTERIOR_BUT_IN_CONTACT,
                            direction=GRAVITATIONAL_UP,
                        ),
                    ),
                ],
                during=DuringAction(
                    continuously=flatten_relations(contacts(agent, theme))
                ),
            )
        ],
        constraining_relations=flatten_relations(bigger_than(goal_reference, theme)),
        after_action_relations=[near(theme, goal_reference)],
    )


def _x_move_y_under_z_template(
    # "A baby moves a car under a chair"
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_distal: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-moves-{theme.handle}-under-{goal_reference.handle}",
        salient_object_variables=[agent, theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                MOVE,
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
                during=DuringAction(
                    continuously=flatten_relations(contacts(agent, theme))
                ),
            )
        ],
        constraining_relations=flatten_relations(bigger_than(goal_reference, theme)),
        after_action_relations=[near(theme, goal_reference)],
    )


def _x_move_y_beside_z_template(
    # "A dog moves a cookie beside a baby"
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_right: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-moves-{theme.handle}-beside-{goal_reference.handle}",
        salient_object_variables=[agent, theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                MOVE,
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
                during=DuringAction(
                    continuously=flatten_relations(contacts(agent, theme))
                ),
            )
        ],
        after_action_relations=[near(theme, goal_reference)],
    )


def _x_move_y_in_front_of_behind_z_template(
    # "Dad moves a chair behind a house"
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_distal: bool,
    is_in_front: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-moves-{theme.handle}-in-front-of-behind-{goal_reference.handle}",
        salient_object_variables=[agent, theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                MOVE,
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
                during=DuringAction(
                    continuously=flatten_relations(contacts(agent, theme))
                ),
            )
        ],
        after_action_relations=[near(theme, goal_reference)],
    )


def _x_moves_towards_away_from_z_template(
    agent: TemplateObjectVariable,
    spatial_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_toward: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-moves-towards-away-{spatial_reference.handle}",
        salient_object_variables=[agent, spatial_reference],
        background_object_variables=background,
        actions=[
            Action(
                MOVE,
                argument_roles_to_fillers=[(AGENT, agent), (GOAL, spatial_reference)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            agent,
                            SpatialPath(
                                operator=TOWARD if is_toward else AWAY_FROM,
                                reference_object=spatial_reference,
                                reference_axis=HorizontalAxisOfObject(agent, 1),
                            ),
                        )
                    ]
                ),
            )
        ],
    )


def _x_moves_y_towards_away_from_z_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    spatial_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_toward: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-moves-{theme.handle}-towards-away-{spatial_reference.handle}",
        salient_object_variables=[agent, theme, spatial_reference],
        background_object_variables=background,
        actions=[
            Action(
                MOVE,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (THEME, theme),
                    (GOAL, spatial_reference),
                ],
                during=DuringAction(
                    continuously=flatten_relations(contacts(agent, theme)),
                    objects_to_paths=[
                        (
                            theme,
                            SpatialPath(
                                operator=TOWARD if is_toward else AWAY_FROM,
                                reference_object=spatial_reference,
                                reference_axis=HorizontalAxisOfObject(theme, 1),
                            ),
                        ),
                        (
                            agent,
                            SpatialPath(
                                operator=TOWARD if is_toward else AWAY_FROM,
                                reference_object=spatial_reference,
                                reference_axis=HorizontalAxisOfObject(agent, 1),
                            ),
                        ),
                    ],
                ),
            )
        ],
    )


def _x_moves_out_of_z_template(
    agent: TemplateObjectVariable,
    spatial_reference: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_distal: bool,
) -> Phase1SituationTemplate:
    inside_relation = inside(agent, spatial_reference)
    return Phase1SituationTemplate(
        f"{agent.handle}-moves-out-of-{spatial_reference.handle}",
        salient_object_variables=[agent, spatial_reference],
        background_object_variables=background,
        actions=[
            Action(
                MOVE,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (
                        GOAL,
                        Region(
                            goal_reference, distance=DISTAL if is_distal else PROXIMAL
                        ),
                    ),
                ],
            )
        ],
        before_action_relations=flatten_relations(inside_relation),
        after_action_relations=flatten_relations(
            [relation.negated_copy() for relation in inside_relation]
        ),
        constraining_relations=flatten_relations(bigger_than(spatial_reference, agent)),
        syntax_hints=[IGNORE_GOAL],
    )


def _x_moves_y_out_of_z_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    spatial_reference: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_distal: bool,
) -> Phase1SituationTemplate:
    inside_relations = inside([agent, theme], spatial_reference)
    return Phase1SituationTemplate(
        f"{agent.handle}-moves-{theme.handle}-out-of-{spatial_reference.handle}",
        salient_object_variables=[agent, theme, spatial_reference],
        background_object_variables=background,
        actions=[
            Action(
                MOVE,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (THEME, theme),
                    (
                        GOAL,
                        Region(
                            goal_reference, distance=DISTAL if is_distal else PROXIMAL
                        ),
                    ),
                ],
            )
        ],
        before_action_relations=flatten_relations(inside_relations),
        after_action_relations=flatten_relations(
            [relation.negated_copy() for relation in inside_relations]
        ),
        constraining_relations=flatten_relations(
            [bigger_than(spatial_reference, [agent, theme]), bigger_than(agent, theme)]
        ),
        syntax_hints=[IGNORE_GOAL],
    )


# JUMP templates


def _jump_in_template(
    # "A dog jumps in a box"
    agent: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-jumps-in-{goal_reference.handle}",
        salient_object_variables=[agent, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                JUMP,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (GOAL, Region(goal_reference, distance=INTERIOR)),
                ],
                auxiliary_variable_bindings=[
                    (JUMP_INITIAL_SUPPORTER_AUX, GROUND_OBJECT_TEMPLATE)
                ],
            )
        ],
        constraining_relations=flatten_relations(bigger_than(goal_reference, agent)),
    )


def _jump_on_template(
    # "Mom jumps on a chair"
    agent: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-jumps-on-{goal_reference.handle}",
        salient_object_variables=[agent, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                JUMP,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (
                        GOAL,
                        Region(
                            goal_reference,
                            distance=EXTERIOR_BUT_IN_CONTACT,
                            direction=GRAVITATIONAL_UP,
                        ),
                    ),
                ],
                auxiliary_variable_bindings=[
                    (JUMP_INITIAL_SUPPORTER_AUX, GROUND_OBJECT_TEMPLATE)
                ],
            )
        ],
        constraining_relations=flatten_relations(bigger_than(goal_reference, agent)),
    )


def _jump_beside_template(
    # "Dad jumps beside a dog"
    agent: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_right: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-jumps-beside-{goal_reference.handle}",
        salient_object_variables=[agent, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                JUMP,
                argument_roles_to_fillers=[
                    (AGENT, agent),
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
                auxiliary_variable_bindings=[
                    (JUMP_INITIAL_SUPPORTER_AUX, GROUND_OBJECT_TEMPLATE)
                ],
            )
        ],
    )


def _jump_in_front_of_behind_template(
    # "A baby jumps in front of a ball"
    agent: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_distal: bool,
    is_in_front: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-jumps-in-front-of-behind-{goal_reference.handle}",
        salient_object_variables=[agent, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                JUMP,
                argument_roles_to_fillers=[
                    (AGENT, agent),
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
                auxiliary_variable_bindings=[
                    (JUMP_INITIAL_SUPPORTER_AUX, GROUND_OBJECT_TEMPLATE)
                ],
            )
        ],
    )


# FLY templates


def _fly_in_template(
    # A bird flies in a house
    agent: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-flies-in-{goal_reference.handle}",
        salient_object_variables=[agent, goal_reference],
        background_object_variables=background,
        actions=[Action(FLY, argument_roles_to_fillers=[(AGENT, agent)])],
        after_action_relations=flatten_relations(inside(agent, goal_reference)),
        constraining_relations=flatten_relations(bigger_than(goal_reference, agent)),
    )


def _fly_beside_template(
    # "A bird flies (along) beside a table"
    agent: TemplateObjectVariable,
    object_passed: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_right: bool,
) -> Phase1SituationTemplate:
    object_region = Region(
        object_passed,
        distance=PROXIMAL,
        direction=Direction(
            positive=is_right,
            relative_to_axis=HorizontalAxisOfObject(object_passed, index=0),
        ),
    )
    return Phase1SituationTemplate(
        f"{agent.handle}-flies-beside-{object_passed.handle}",
        salient_object_variables=[agent, object_passed],
        background_object_variables=background,
        actions=[
            Action(
                FLY,
                argument_roles_to_fillers=[(AGENT, agent)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            agent,
                            SpatialPath(
                                VIA,
                                reference_object=object_region,
                                reference_axis=HorizontalAxisOfObject(
                                    object_passed, index=0
                                ),
                            ),
                        )
                    ],
                    at_some_point=flatten_relations(
                        [Relation(IN_REGION, agent, object_region)]
                    ),
                ),
            )
        ],
    )


def _fly_in_front_of_behind_template(
    # "A bird flies (along) behind a truck"
    agent: TemplateObjectVariable,
    object_passed: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_distal: bool,
    is_in_front: bool,
) -> Phase1SituationTemplate:
    object_region = Region(
        object_passed,
        distance=DISTAL if is_distal else PROXIMAL,
        direction=Direction(
            positive=is_in_front, relative_to_axis=FacingAddresseeAxis(object_passed)
        ),
    )
    return Phase1SituationTemplate(
        f"{agent.handle}-flies-in-front-of-behind-{object_passed.handle}",
        salient_object_variables=[agent, object_passed],
        background_object_variables=background,
        actions=[
            Action(
                FLY,
                argument_roles_to_fillers=[(AGENT, agent)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            agent,
                            SpatialPath(
                                VIA,
                                reference_object=object_region,
                                reference_axis=FacingAddresseeAxis(object_passed),
                            ),
                        )
                    ],
                    at_some_point=flatten_relations(
                        [Relation(IN_REGION, agent, object_region)]
                    ),
                ),
            )
        ],
    )


def _fly_towards_away_template(
    agent: TemplateObjectVariable,
    spatial_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_toward: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-flies-towards-away-from-{spatial_reference.handle}",
        salient_object_variables=[agent, spatial_reference],
        background_object_variables=background,
        actions=[
            Action(
                FLY,
                argument_roles_to_fillers=[(AGENT, agent)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            agent,
                            SpatialPath(
                                operator=TOWARD if is_toward else AWAY_FROM,
                                reference_object=spatial_reference,
                                reference_axis=HorizontalAxisOfObject(agent, 1),
                            ),
                        )
                    ]
                ),
            )
        ],
    )


def _fly_out_template(
    agent: TemplateObjectVariable,
    spatial_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    inside_relation = inside(agent, spatial_reference)
    return Phase1SituationTemplate(
        f"{agent.handle}-fly-out-of-{spatial_reference.handle}",
        salient_object_variables=[agent, spatial_reference],
        background_object_variables=background,
        actions=[Action(FLY, argument_roles_to_fillers=[(AGENT, agent)])],
        before_action_relations=flatten_relations(inside_relation),
        after_action_relations=flatten_relations(
            [relation.negated_copy() for relation in inside_relation]
        ),
        constraining_relations=flatten_relations(bigger_than(spatial_reference, agent)),
    )


# Come Templates


def _make_come_out_of_template(
    agent: TemplateObjectVariable,
    object_containing_agent: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    backgrounds_objects = [goal_reference]
    backgrounds_objects.extend(background)
    return Phase1SituationTemplate(
        f"{agent.handle}-come-out-of-{object_containing_agent.handle}",
        salient_object_variables=[agent, object_containing_agent],
        background_object_variables=backgrounds_objects,
        actions=[
            Action(
                COME, argument_roles_to_fillers=[(AGENT, agent), (GOAL, goal_reference)]
            )
        ],
        before_action_relations=flatten_relations(inside(agent, object_containing_agent)),
        after_action_relations=flatten_relations(
            [negate(inside(agent, object_containing_agent))]
        ),
        constraining_relations=flatten_relations(
            bigger_than(object_containing_agent, agent)
        ),
        syntax_hints=[IGNORE_GOAL],
    )


# Push


def _make_push_with_prepositions(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    agent = standard_object(
        "agent",
        THING,
        required_properties=[ANIMATE],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )
    theme = standard_object("theme", INANIMATE_OBJECT)
    goal_reference = standard_object("goal_reference", INANIMATE_OBJECT)
    goal_in = standard_object("goal_in", INANIMATE_OBJECT, required_properties=[HOLLOW])
    goal_under = standard_object(
        "goal_under", INANIMATE_OBJECT, required_properties=[HAS_SPACE_UNDER]
    )
    surface = standard_object(
        "surface", THING, required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
    )

    background = make_noise_objects(noise_objects)

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
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
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
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
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
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
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
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_distal in BOOL_SET
                    for is_in_front in BOOL_SET
                ]
            ),
            # Towards, Away
            flatten(
                [
                    sampled(
                        _push_towards_away_template(
                            agent,
                            theme,
                            goal_reference,
                            surface,
                            background,
                            is_ending_proximal=is_ending_proximal,
                            is_towards=is_towards,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_ending_proximal in BOOL_SET
                    for is_towards in BOOL_SET
                ]
            ),
            # Out
            flatten(
                [
                    sampled(
                        _push_out_template(
                            agent,
                            theme,
                            goal_in,
                            goal_reference,
                            surface,
                            background,
                            is_distal=is_distal,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_distal in BOOL_SET
                ]
            ),
        ),
        language_generator=language_generator,
    )


def _make_go_with_prepositions(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    agent = standard_object(
        "agent",
        THING,
        required_properties=[ANIMATE],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )
    goal_object = standard_object(
        "goal_object", banned_properties=[IS_SPEAKER, IS_ADDRESSEE]
    )

    goal_object_hollow = standard_object(
        "goal_object_hollow", required_properties=[HOLLOW], banned_properties=[ANIMATE]
    )
    goal_object_with_space_under = standard_object(
        "goal_object_with_space_under",
        required_properties=[HAS_SPACE_UNDER],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )
    path_object = standard_object(
        "path_object",
        required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM, HAS_SPACE_UNDER],
    )
    background = make_noise_objects(noise_objects)

    return phase1_instances(
        "Go + PP",
        chain(
            # To
            flatten(
                [
                    sampled(
                        _go_to_template(agent, goal_object, background),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                ]
            ),
            # In
            flatten(
                [
                    sampled(
                        _go_in_template(agent, goal_object_hollow, background),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
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
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
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
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_distal in BOOL_SET
                    for is_behind in BOOL_SET
                ]
            ),
            # Over
            flatten(
                [
                    sampled(
                        _go_over_template(
                            agent, goal_object, background, is_distal=is_distal
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_distal in BOOL_SET
                ]
            ),
            # Under
            flatten(
                [
                    sampled(
                        _go_under_template(
                            agent,
                            goal_object_with_space_under,
                            background,
                            is_distal=is_distal,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_distal in BOOL_SET
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
                            is_near_path=is_near_path,
                            is_near_goal=is_near_goal,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_behind in BOOL_SET
                    for is_near_path in BOOL_SET
                    for is_near_goal in BOOL_SET
                ]
            ),
            # Over & Under Paths
            flatten(
                [
                    sampled(
                        _go_over_under_path_template(
                            agent,
                            goal_object,
                            path_object,
                            background,
                            is_over=is_over,
                            is_near_goal=is_near_goal,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_over in BOOL_SET
                    for is_near_goal in BOOL_SET
                ]
            ),
            # Toward & Away Paths
            flatten(
                [
                    sampled(
                        _go_towards_away_template(
                            agent,
                            goal_object,
                            background,
                            is_toward=is_toward,
                            is_near_goal=is_near_goal,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_toward in BOOL_SET
                    for is_near_goal in BOOL_SET
                ]
            ),
            # Out
            flatten(
                [
                    sampled(
                        _x_go_out_y_template(
                            agent,
                            goal_object_hollow,
                            goal_object,
                            background,
                            is_distal=is_distal,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_distal in BOOL_SET
                ]
            ),
        ),
        language_generator=language_generator,
    )


def _make_sit_with_prepositions(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[ANIMATE])
    seat = standard_object(
        "seat",
        INANIMATE_OBJECT,
        required_properties=[CAN_BE_SAT_ON_BY_PEOPLE],
        banned_properties=[GROUND],
    )
    seat_in = standard_object("seat_in", INANIMATE_OBJECT, required_properties=[HOLLOW])
    surface = standard_object(
        "surface", THING, required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
    )
    background = make_noise_objects(noise_objects)
    syntax_hints_options: Sequence[Sequence[str]] = [[], [USE_ADVERBIAL_PATH_MODIFIER]]

    return phase1_instances(
        "Sit + PP",
        chain(
            # on
            flatten(
                [
                    sampled(
                        _sit_on_template(
                            agent, seat, surface, background, syntax_hints=syntax_hints
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for syntax_hints in syntax_hints_options
                ]
            ),
            # in
            flatten(
                [
                    sampled(
                        _sit_in_template(
                            agent, seat_in, surface, background, syntax_hints=syntax_hints
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for syntax_hints in syntax_hints_options
                ]
            ),
        ),
        language_generator=language_generator,
    )


def _make_roll_with_prepositions(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    agent = standard_object(
        "agent",
        THING,
        required_properties=[ANIMATE],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )

    goal_object = standard_object("goal_object")
    goal_object_hollow = standard_object(
        "goal_object_hollow",
        required_properties=[HOLLOW],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )
    theme = standard_object("rollee", required_properties=[ROLLABLE])
    ground = standard_object("ground", root_node=GROUND)
    roll_surface = standard_object(
        "rollable_surface", required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
    )
    noise_objects_immutable: Iterable[TemplateObjectVariable] = immutableset(
        make_noise_objects(noise_objects)
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
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
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
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
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
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
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
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
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
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
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
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
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
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
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
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_over in BOOL_SET
                    for surface in surfaces
                ]
            ),
            # X rolls toward/away from Z
            flatten(
                [
                    sampled(
                        _x_rolls_towards_away_from_y_template(
                            agent,
                            goal_object,
                            surface,
                            noise_objects_immutable,
                            is_toward=is_toward,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_toward in BOOL_SET
                    for surface in surfaces
                ]
            ),
            # X rolls y toward/away from Z
            flatten(
                [
                    sampled(
                        _x_rolls_y_towards_away_from_z_template(
                            agent,
                            theme,
                            goal_object,
                            surface,
                            noise_objects_immutable,
                            is_toward=is_toward,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_toward in BOOL_SET
                    for surface in surfaces
                ]
            ),
            # X rolls out of Z
            flatten(
                [
                    sampled(
                        _x_rolls_out_z_template(
                            agent, goal_object_hollow, surface, noise_objects_immutable
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for surface in surfaces
                ]
            ),
            # X rolls Y out of Z
            flatten(
                [
                    sampled(
                        _x_rolls_y_out_of_z_template(
                            agent,
                            theme,
                            goal_object_hollow,
                            surface,
                            noise_objects_immutable,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for surface in surfaces
                ]
            ),
        ),
        language_generator=language_generator,
    )


def _make_take_with_prepositions(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[ANIMATE])
    theme = standard_object("theme", INANIMATE_OBJECT)
    goal_reference = standard_object("goal_reference", INANIMATE_OBJECT)
    background = make_noise_objects(noise_objects)

    return phase1_instances(
        "Take + PP",
        # To
        flatten(
            [
                sampled(
                    _take_to_template(agent, theme, goal_reference, background),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples if num_samples else 5,
                )
            ]
        ),
        language_generator=language_generator,
    )


def _make_fall_with_prepositions(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    theme = standard_object("theme", THING, banned_properties=[IS_SPEAKER, IS_ADDRESSEE])
    goal_reference = standard_object(
        "goal_reference", THING, banned_properties=[IS_SPEAKER, IS_ADDRESSEE]
    )
    goal_on = standard_object(
        "goal_on", THING, required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
    )

    goal_in = standard_object(
        "goal_in",
        THING,
        required_properties=[HOLLOW],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )

    background = make_noise_objects(noise_objects)

    syntax_hints_options: Sequence[Sequence[str]] = [[], [USE_ADVERBIAL_PATH_MODIFIER]]

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
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
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
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
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
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
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
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for syntax_hints in syntax_hints_options
                    for is_distal in BOOL_SET
                    for is_in_front in BOOL_SET
                ]
            ),
            # Toward, Away from
            flatten(
                [
                    sampled(
                        _fall_toward_away_from_template(
                            theme,
                            goal_reference,
                            background,
                            syntax_hints=syntax_hints,
                            is_toward=is_toward,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for syntax_hints in syntax_hints_options
                    for is_toward in BOOL_SET
                ]
            ),
        ),
        language_generator=language_generator,
    )


def _make_put_with_prepositions(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[ANIMATE])
    speaker_agent = standard_object(
        "speaker_agent",
        PERSON,
        required_properties=[ANIMATE],
        added_properties=[IS_SPEAKER],
    )
    addressee_agent = standard_object(
        "addressee_agent",
        PERSON,
        required_properties=[ANIMATE],
        added_properties=[IS_ADDRESSEE],
    )
    theme = standard_object("theme", INANIMATE_OBJECT)
    goal_reference = standard_object("goal_reference", INANIMATE_OBJECT)
    goal_in = standard_object("goal_in", INANIMATE_OBJECT, required_properties=[HOLLOW])
    goal_under = standard_object(
        "goal_under", INANIMATE_OBJECT, required_properties=[HAS_SPACE_UNDER]
    )
    body_part_goal = body_part_object(
        "body_part_goal", required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
    )
    background = make_noise_objects(noise_objects)
    on_in_templates = [
        _put_on_template(agent, theme, goal_reference, background),
        _put_in_template(agent, theme, goal_in, background),
    ]
    special_agents = [speaker_agent, addressee_agent]

    return phase1_instances(
        "Put + PP",
        chain(
            # on, in
            flatten(
                [
                    sampled(
                        template,
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for template in on_in_templates
                ]
            ),
            # on body part
            flatten(
                [
                    sampled(
                        _put_on_body_part_template(
                            speaker_addressee, theme, body_part_goal, background
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for speaker_addressee in special_agents
                ]
            ),
            # beside
            flatten(
                [
                    sampled(
                        _put_beside_template(
                            agent, theme, goal_reference, background, is_right=is_right
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_right in BOOL_SET
                ]
            ),
            # under
            flatten(
                [
                    sampled(
                        _put_under_template(
                            agent, theme, goal_under, background, is_distal=is_distal
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_distal in BOOL_SET
                ]
            ),
            # in front of, behind
            flatten(
                [
                    sampled(
                        _put_in_front_of_behind_template(
                            agent,
                            theme,
                            goal_reference,
                            background,
                            is_distal=is_distal,
                            is_in_front=is_in_front,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_distal in BOOL_SET
                    for is_in_front in BOOL_SET
                ]
            ),
        ),
        language_generator=language_generator,
    )


def _make_move_with_prepositions(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    agent = standard_object(
        "agent",
        THING,
        required_properties=[SELF_MOVING],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )
    manipulating_agent = standard_object(
        "manipulating_agent",
        THING,
        required_properties=[ANIMATE],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )
    theme = standard_object("theme", INANIMATE_OBJECT)
    goal_reference = standard_object("goal_reference", INANIMATE_OBJECT)
    goal_in = standard_object("goal_in", INANIMATE_OBJECT, required_properties=[HOLLOW])
    goal_on = standard_object(
        "goal_on", INANIMATE_OBJECT, required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
    )
    goal_under = standard_object(
        "goal_under",
        INANIMATE_OBJECT,
        required_properties=[HAS_SPACE_UNDER],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )

    background = make_noise_objects(noise_objects)

    situation_templates = [
        _x_move_y_in_z_template(manipulating_agent, theme, goal_in, background),
        _x_move_y_on_z_template(manipulating_agent, theme, goal_on, background),
    ]

    return phase1_instances(
        "Move + PP",
        chain(
            # move beside
            flatten(
                [
                    sampled(
                        _x_move_beside_y_template(
                            agent, goal_reference, background, is_right=is_right
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_right in BOOL_SET
                ]
            ),
            # move in front, behind
            flatten(
                [
                    sampled(
                        _x_move_in_front_of_behind_y_template(
                            agent,
                            goal_reference,
                            background,
                            is_distal=is_distal,
                            is_in_front=is_in_front,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_distal in BOOL_SET
                    for is_in_front in BOOL_SET
                ]
            ),
            # move under
            flatten(
                [
                    sampled(
                        _x_move_under_y_template(
                            agent, goal_under, background, is_distal=is_distal
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_distal in BOOL_SET
                ]
            ),
            # move toward or away from
            flatten(
                [
                    sampled(
                        _x_moves_towards_away_from_z_template(
                            agent, goal_reference, background, is_toward=is_toward
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_toward in BOOL_SET
                ]
            ),
            # move something in, on
            flatten(
                [
                    sampled(
                        template,
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for template in situation_templates
                ]
            ),
            # move something under
            flatten(
                [
                    sampled(
                        _x_move_y_under_z_template(
                            manipulating_agent,
                            theme,
                            goal_under,
                            background,
                            is_distal=is_distal,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_distal in BOOL_SET
                ]
            ),
            # move something beside
            flatten(
                [
                    sampled(
                        _x_move_y_beside_z_template(
                            manipulating_agent,
                            theme,
                            goal_reference,
                            background,
                            is_right=is_right,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_right in BOOL_SET
                ]
            ),
            # move something in front of, behind
            flatten(
                [
                    sampled(
                        _x_move_y_in_front_of_behind_z_template(
                            manipulating_agent,
                            theme,
                            goal_reference,
                            background,
                            is_distal=is_distal,
                            is_in_front=is_in_front,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_distal in BOOL_SET
                    for is_in_front in BOOL_SET
                ]
            ),
            # move something toward, away_from
            flatten(
                [
                    sampled(
                        _x_moves_y_towards_away_from_z_template(
                            manipulating_agent,
                            theme,
                            goal_reference,
                            background,
                            is_toward=is_toward,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_toward in BOOL_SET
                ]
            ),
            # x moves out of z
            flatten(
                [
                    sampled(
                        _x_moves_out_of_z_template(
                            agent,
                            goal_in,
                            goal_reference,
                            background,
                            is_distal=is_distal,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_distal in BOOL_SET
                ]
            ),
            # x moves y out of z
            flatten(
                [
                    sampled(
                        _x_moves_y_out_of_z_template(
                            manipulating_agent,
                            theme,
                            goal_in,
                            goal_reference,
                            background,
                            is_distal=is_distal,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_distal in BOOL_SET
                ]
            ),
        ),
        language_generator=language_generator,
    )


def _make_throw_with_prepositions(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[ANIMATE])
    theme = standard_object("theme", INANIMATE_OBJECT)
    thrower = standard_object("thrower", PERSON)
    recipient = standard_object("recipient", PERSON)
    goal_reference = standard_object("goal_reference", INANIMATE_OBJECT)
    goal_in = standard_object("goal_in", THING, required_properties=[HOLLOW])
    goal_on = standard_object(
        "goal_on", THING, required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
    )
    goal_under = standard_object(
        "goal_under", THING, required_properties=[HAS_SPACE_UNDER]
    )
    implicit_goal_reference = standard_object("goal_reference", INANIMATE_OBJECT)
    background = make_noise_objects(noise_objects)
    situation_templates = [
        _throw_to_template(agent, theme, goal_reference, background),
        _throw_in_template(agent, theme, goal_in, background),
        _throw_on_template(agent, theme, goal_on, background),
        _x_throws_y_to_z_template(thrower, theme, recipient, background),
    ]

    return phase1_instances(
        "Throw + PP",
        chain(
            # to, in, on, to recipient
            flatten(
                [
                    sampled(
                        template,
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for template in situation_templates
                ]
            ),
            # beside
            flatten(
                [
                    sampled(
                        _throw_beside_template(
                            agent, theme, goal_reference, background, is_right=is_right
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_right in BOOL_SET
                ]
            ),
            # in front of, behind
            flatten(
                [
                    sampled(
                        _throw_in_front_of_behind_template(
                            agent,
                            theme,
                            goal_reference,
                            background,
                            is_distal=is_distal,
                            is_in_front=is_in_front,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_distal in BOOL_SET
                    for is_in_front in BOOL_SET
                ]
            ),
            # under
            flatten(
                [
                    sampled(
                        _throw_under_template(
                            agent, theme, goal_under, background, is_distal=is_distal
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_distal in BOOL_SET
                ]
            ),
            # path over
            flatten(
                [
                    sampled(
                        _throw_path_over_template(
                            agent,
                            theme,
                            goal_reference,
                            implicit_goal_reference,
                            background,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                ]
            ),
            # path under
            flatten(
                [
                    sampled(
                        _throw_path_under_template(
                            agent,
                            theme,
                            goal_under,
                            implicit_goal_reference,
                            background,
                            is_distal=is_distal,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_distal in BOOL_SET
                ]
            ),
            # Towards & Away
            flatten(
                [
                    sampled(
                        _throw_towards_away_template(
                            agent,
                            theme,
                            goal_reference,
                            background,
                            is_towards=is_towards,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_towards in BOOL_SET
                ]
            ),
        ),
        language_generator=language_generator,
    )


def _make_jump_with_prepositions(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    agent = standard_object(
        "agent",
        THING,
        required_properties=[CAN_JUMP],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )
    goal_reference = standard_object(
        "goal_reference", THING, banned_properties=[IS_SPEAKER, IS_ADDRESSEE]
    )
    goal_in = standard_object(
        "goal_reference", THING, required_properties=[HOLLOW], banned_properties=[ANIMATE]
    )
    goal_on = standard_object(
        "goal_reference", THING, required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
    )

    background = make_noise_objects(noise_objects)

    templates = [
        _jump_in_template(agent, goal_in, background),
        _jump_on_template(agent, goal_on, background),
        _jump_over_template(agent, goal_reference, background),
    ]

    return phase1_instances(
        "Jump + PP",
        chain(
            # in, on, over
            flatten(
                [
                    sampled(
                        template,
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for template in templates
                ]
            ),
            # beside
            flatten(
                [
                    sampled(
                        _jump_beside_template(
                            agent, goal_reference, background, is_right=is_right
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_right in BOOL_SET
                ]
            ),
            # in front of, behind
            flatten(
                [
                    sampled(
                        _jump_in_front_of_behind_template(
                            agent,
                            goal_reference,
                            background,
                            is_distal=is_distal,
                            is_in_front=is_in_front,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_distal in BOOL_SET
                    for is_in_front in BOOL_SET
                ]
            ),
        ),
        language_generator=language_generator,
    )


def _make_fly_with_prepositions(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[CAN_FLY])
    goal_reference = standard_object(
        "goal_reference", THING, banned_properties=[IS_SPEAKER, IS_ADDRESSEE]
    )
    goal_in = standard_object(
        "goal_in",
        THING,
        required_properties=[HOLLOW],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )
    goal_under = standard_object(
        "goal_under", THING, required_properties=[HAS_SPACE_UNDER]
    )

    background = make_noise_objects(noise_objects)

    return phase1_instances(
        "Fly + PP",
        chain(
            # in
            flatten(
                [
                    sampled(
                        _fly_in_template(agent, goal_in, background),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                ]
            ),
            # beside
            flatten(
                [
                    sampled(
                        _fly_beside_template(
                            agent, goal_reference, background, is_right=is_right
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_right in BOOL_SET
                ]
            ),
            # in front of, behind
            flatten(
                [
                    sampled(
                        _fly_in_front_of_behind_template(
                            agent,
                            goal_reference,
                            background,
                            is_distal=is_distal,
                            is_in_front=is_in_front,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_distal in BOOL_SET
                    for is_in_front in BOOL_SET
                ]
            ),
            # over
            flatten(
                [
                    sampled(
                        _fly_over_template(agent, goal_reference, background),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                ]
            ),
            # under
            flatten(
                [
                    sampled(
                        _fly_under_template(agent, goal_under, background),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                ]
            ),
            # toward or away from
            flatten(
                [
                    sampled(
                        _fly_towards_away_template(
                            agent, goal_reference, background, is_toward=is_toward
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for is_toward in BOOL_SET
                ]
            ),
            # out
            flatten(
                [
                    sampled(
                        _fly_out_template(agent, goal_in, background),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                ]
            ),
        ),
        language_generator=language_generator,
    )


# Come


def _make_come_with_prepositions(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    agent = standard_object("agent", required_properties=[SELF_MOVING])
    object_with_agent_inside = standard_object(
        "object-agent-inside", required_properties=[HOLLOW]
    )
    goal_object = standard_object("goal")
    background = make_noise_objects(num_noise_objects)

    return phase1_instances(
        "Come + PP",
        chain(
            # Come Out Of
            flatten(
                [
                    sampled(
                        _make_come_out_of_template(
                            agent, object_with_agent_inside, goal_object, background
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                ]
            )
        ),
        language_generator=language_generator,
    )


def make_verb_with_dynamic_prepositions_curriculum(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Sequence[Phase1InstanceGroup]:
    return [
        _make_push_with_prepositions(num_samples, num_noise_objects, language_generator),
        _make_go_with_prepositions(num_samples, num_noise_objects, language_generator),
        _make_throw_with_prepositions(num_samples, num_noise_objects, language_generator),
        _make_sit_with_prepositions(num_samples, num_noise_objects, language_generator),
        _make_roll_with_prepositions(num_samples, num_noise_objects, language_generator),
        _make_take_with_prepositions(num_samples, num_noise_objects, language_generator),
        _make_fall_with_prepositions(num_samples, num_noise_objects, language_generator),
        _make_put_with_prepositions(num_samples, num_noise_objects, language_generator),
        _make_move_with_prepositions(num_samples, num_noise_objects, language_generator),
        _make_jump_with_prepositions(num_samples, num_noise_objects, language_generator),
        _make_fly_with_prepositions(num_samples, num_noise_objects, language_generator),
        _make_come_with_prepositions(num_samples, num_noise_objects, language_generator),
    ]
