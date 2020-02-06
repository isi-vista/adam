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
from adam.ontology import THING
from adam.ontology.phase1_ontology import (
    GO,
    AGENT,
    bigger_than,
    GOAL,
    SELF_MOVING,
    GAILA_PHASE_1_ONTOLOGY,
    HOLLOW,
    HAS_SPACE_UNDER,
)
from adam.ontology.phase1_spatial_relations import (
    Region,
    PROXIMAL,
    INTERIOR,
    Direction,
    GRAVITATIONAL_DOWN,
    GRAVITATIONAL_UP,
)
from adam.situation import Action
from adam.situation.templates.phase1_templates import (
    TemplateObjectVariable,
    Phase1SituationTemplate,
    sampled,
)

BOOL_SET = immutableset([True, False])


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


def _go_behind_template(
    agent: TemplateObjectVariable,
    goal_object: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
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
                            distance=PROXIMAL,
                            direction=Direction(
                                positive=False,
                                relative_to_axis=FacingAddresseeAxis(goal_object),
                            ),
                        ),
                    ),
                ],
            )
        ],
    )


def _go_in_front_of_template(
    agent: TemplateObjectVariable,
    goal_object: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"go_in_front_of-{agent.handle}-behind-{goal_object.handle}",
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
                                positive=True,
                                relative_to_axis=FacingAddresseeAxis(goal_object),
                            ),
                        ),
                    ),
                ],
            )
        ],
    )


def _go_over_template(
    agent: TemplateObjectVariable,
    goal_object: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
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
                            goal_object, distance=PROXIMAL, direction=GRAVITATIONAL_DOWN
                        ),
                    ),
                ],
            )
        ],
    )


def _go_under_template(
    agent: TemplateObjectVariable,
    goal_object: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
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
                            goal_object, distance=PROXIMAL, direction=GRAVITATIONAL_UP
                        ),
                    ),
                ],
            )
        ],
    )


def _make_go_to(num_samples: int = 5, *, noise_objects: int = 0) -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[SELF_MOVING])
    goal_object = standard_object("goal_object")

    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )

    return phase1_instances(
        "Go To",
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
    )


def _make_go_in(num_samples: int = 5, *, noise_objects: int = 0) -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[SELF_MOVING])
    goal_object = standard_object("goal_object", required_properties=[HOLLOW])

    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )
    return phase1_instances(
        "Go In",
        flatten(
            [
                sampled(
                    _go_in_template(agent, goal_object, background),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                    max_to_sample=num_samples,
                )
            ]
        ),
    )


def _make_go_beside(
    num_samples: int = 5, *, noise_objects: int = 0
) -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[SELF_MOVING])
    goal_object = standard_object("goal_object")

    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )
    return phase1_instances(
        "Go Beside",
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
    )


def _make_go_behind(
    num_samples: int = 5, *, noise_objects: int = 0
) -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[SELF_MOVING])
    goal_object = standard_object("goal_object")

    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )
    return phase1_instances(
        "Go Behind",
        flatten(
            [
                sampled(
                    _go_behind_template(agent, goal_object, background),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                    max_to_sample=num_samples,
                )
            ]
        ),
    )


def _make_go_in_front_of(
    num_samples: int = 5, *, noise_objects: int = 0
) -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[SELF_MOVING])
    goal_object = standard_object("goal_object")

    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )
    return phase1_instances(
        "Go In Front Of",
        flatten(
            [
                sampled(
                    _go_in_front_of_template(agent, goal_object, background),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                    max_to_sample=num_samples,
                )
            ]
        ),
    )


def _make_go_over(num_samples: int = 5, *, noise_objects: int = 0) -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[SELF_MOVING])
    goal_object = standard_object("goal_object")

    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )
    return phase1_instances(
        "Go Over",
        flatten(
            [
                sampled(
                    _go_over_template(agent, goal_object, background),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                    max_to_sample=num_samples,
                )
            ]
        ),
    )


def _make_go_under(
    num_samples: int = 5, *, noise_objects: int = 0
) -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[SELF_MOVING])
    goal_object = standard_object("goal_object", required_properties=[HAS_SPACE_UNDER])

    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )
    return phase1_instances(
        "Go Under",
        flatten(
            [
                sampled(
                    _go_under_template(agent, goal_object, background),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                    max_to_sample=num_samples,
                )
            ]
        ),
    )


def make_verb_prepositions_curriculum(
    num_samples: int = 5, *, num_noise_objects: int = 0
):
    return [
        _make_go_to(num_samples, noise_objects=num_noise_objects),
        _make_go_in(num_samples, noise_objects=num_noise_objects),
        _make_go_beside(num_samples, noise_objects=num_noise_objects),
        _make_go_behind(num_samples, noise_objects=num_noise_objects),
        _make_go_in_front_of(num_samples, noise_objects=num_noise_objects),
        _make_go_over(num_samples, noise_objects=num_noise_objects),
        _make_go_under(num_samples, noise_objects=num_noise_objects),
    ]
