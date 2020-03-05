"""
This contains methods that create situation templates.
"""
from typing import Iterable

from adam.ontology.during import DuringAction
from adam.ontology.phase1_ontology import strictly_above, AGENT, FLY, bigger_than
from adam.relation import flatten_relations
from adam.situation import Action
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    TemplateObjectVariable,
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
