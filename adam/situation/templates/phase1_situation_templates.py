"""
This contains methods that create common situation templates.
"""
from typing import Iterable

from adam.curriculum.curriculum_utils import GROUND_OBJECT_TEMPLATE
from adam.ontology.during import DuringAction
from adam.ontology.phase1_ontology import (
    strictly_above,
    AGENT,
    FLY,
    bigger_than,
    JUMP_INITIAL_SUPPORTER_AUX,
    JUMP,
    GOAL,
    THEME,
    PUT,
    has,
    GO,
)
from adam.ontology.phase1_spatial_relations import (
    INTERIOR,
    Region,
    GRAVITATIONAL_UP,
    EXTERIOR_BUT_IN_CONTACT,
    GRAVITATIONAL_DOWN,
    PROXIMAL,
    DISTAL,
)
from adam.relation import flatten_relations
from adam.situation import Action
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    TemplateObjectVariable,
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
        constraining_relations=flatten_relations(bigger_than(goal_object, agent)),
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


def _go_under_template(
    agent: TemplateObjectVariable,
    goal_object: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_distal: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"go_under-{agent.handle}-under-{goal_object.handle}",
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
                            direction=GRAVITATIONAL_DOWN,
                        ),
                    ),
                ],
            )
        ],
        constraining_relations=flatten_relations(bigger_than(goal_object, agent)),
    )


def _put_on_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-puts-{theme.handle}-on-{goal_reference.handle}",
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
                            distance=EXTERIOR_BUT_IN_CONTACT,
                            direction=GRAVITATIONAL_UP,
                        ),
                    ),
                ],
            )
        ],
        constraining_relations=flatten_relations(
            bigger_than([agent, goal_reference], theme)
        ),
    )


def _put_on_body_part_template(
    # X puts Y on body part
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-puts-{theme.handle}-on-{goal_reference.handle}",
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
                            distance=EXTERIOR_BUT_IN_CONTACT,
                            direction=GRAVITATIONAL_UP,
                        ),
                    ),
                ],
            )
        ],
        constraining_relations=flatten_relations(
            bigger_than([agent, goal_reference], theme)
        ),
        asserted_always_relations=flatten_relations(has(agent, goal_reference)),
    )


def _put_in_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-puts-{theme.handle}-in-{goal_reference.handle}",
        salient_object_variables=[agent, theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                PUT,
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
    )


def _jump_over_template(
    # "Mom jumps over a ball"
    agent: TemplateObjectVariable,
    object_in_path: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-jumps-over-{object_in_path.handle}",
        salient_object_variables=[agent, object_in_path],
        background_object_variables=background,
        actions=[
            Action(
                JUMP,
                argument_roles_to_fillers=[(AGENT, agent)],
                during=DuringAction(
                    at_some_point=flatten_relations(strictly_above(agent, object_in_path))
                ),
                auxiliary_variable_bindings=[
                    (JUMP_INITIAL_SUPPORTER_AUX, GROUND_OBJECT_TEMPLATE)
                ],
            )
        ],
        constraining_relations=flatten_relations(bigger_than(agent, object_in_path)),
    )


def _fly_over_template(
    # A bird flies over a ball
    agent: TemplateObjectVariable,
    object_in_path: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-flies-over-{object_in_path.handle}",
        salient_object_variables=[agent, object_in_path],
        background_object_variables=background,
        actions=[
            Action(
                FLY,
                argument_roles_to_fillers=[(AGENT, agent)],
                during=DuringAction(
                    at_some_point=flatten_relations(strictly_above(agent, object_in_path))
                ),
            )
        ],
    )


def _fly_under_template(
    # A bird flies under a chair
    agent: TemplateObjectVariable,
    object_in_path: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-flies-under-{object_in_path.handle}",
        salient_object_variables=[agent, object_in_path],
        background_object_variables=background,
        actions=[
            Action(
                FLY,
                argument_roles_to_fillers=[(AGENT, agent)],
                during=DuringAction(
                    at_some_point=flatten_relations(strictly_above(object_in_path, agent))
                ),
            )
        ],
        constraining_relations=flatten_relations(bigger_than(object_in_path, agent)),
    )
