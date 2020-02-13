from typing import Iterable

from immutablecollections import immutableset
from more_itertools import flatten

from adam.axes import HorizontalAxisOfObject, FacingAddresseeAxis
from adam.curriculum.curriculum_utils import (
    standard_object,
    phase1_instances,
    PHASE1_CHOOSER,
    Phase1InstanceGroup,
    GROUND_OBJECT_TEMPLATE,
)
from adam.ontology import THING
from adam.ontology.during import DuringAction
from adam.ontology.phase1_ontology import (
    AGENT,
    bigger_than,
    GOAL,
    GAILA_PHASE_1_ONTOLOGY,
    HOLLOW,
    JUMP,
    ANIMATE,
    INANIMATE_OBJECT,
    CAN_HAVE_THINGS_RESTING_ON_THEM,
    JUMP_INITIAL_SUPPORTER_AUX,
    CAN_JUMP,
    strictly_above,
)
from adam.ontology.phase1_spatial_relations import (
    Region,
    Direction,
    INTERIOR,
    GRAVITATIONAL_UP,
    EXTERIOR_BUT_IN_CONTACT,
    DISTAL,
    PROXIMAL,
)
from adam.situation import Action
from adam.situation.templates.phase1_templates import (
    TemplateObjectVariable,
    Phase1SituationTemplate,
    sampled,
)

BOOL_SET = immutableset([True, False])


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
        constraining_relations=[bigger_than(goal_reference, agent)],
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
        constraining_relations=[bigger_than(goal_reference, agent)],
    )


def _jump_beside_template(
    # "Dad jumps beside a dog"
    agent: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    is_distal: bool,
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
                            distance=DISTAL if is_distal else PROXIMAL,
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
                                relative_to_axis=FacingAddresseeAxis(
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
                    at_some_point=[strictly_above(agent, object_in_path)]
                ),
                auxiliary_variable_bindings=[
                    (JUMP_INITIAL_SUPPORTER_AUX, GROUND_OBJECT_TEMPLATE)
                ],
            )
        ],
        constraining_relations=[bigger_than(agent, object_in_path)],
    )


def _make_jump_in(num_samples: int = 5, *, noise_objects: int = 0) -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[CAN_JUMP])
    goal_reference = standard_object(
        "goal_reference", THING, required_properties=[HOLLOW]
    )
    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )

    return phase1_instances(
        "Jump In",
        flatten(
            [
                sampled(
                    _jump_in_template(agent, goal_reference, background),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                    max_to_sample=num_samples,
                )
            ]
        ),
    )


def _make_jump_on(num_samples: int = 5, *, noise_objects: int = 0) -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[CAN_JUMP])
    goal_reference = standard_object(
        "goal_reference", THING, required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
    )
    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )

    return phase1_instances(
        "Jump On",
        flatten(
            [
                sampled(
                    _jump_on_template(agent, goal_reference, background),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                    max_to_sample=num_samples,
                )
            ]
        ),
    )


def _make_jump_beside(
    num_samples: int = 5, *, noise_objects: int = 0
) -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[CAN_JUMP])
    goal_reference = standard_object("goal_reference", THING)
    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )

    return phase1_instances(
        "Jump Beside",
        flatten(
            [
                sampled(
                    _jump_beside_template(
                        agent, goal_reference, background, is_distal, is_right
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                    max_to_sample=num_samples,
                )
                for is_distal in BOOL_SET
                for is_right in BOOL_SET
            ]
        ),
    )


def _make_jump_in_front_of_behind(
    num_samples: int = 5, *, noise_objects: int = 0
) -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[CAN_JUMP])
    goal_reference = standard_object("goal_reference", THING)
    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )

    return phase1_instances(
        "Jump In Front Of Behind",
        flatten(
            [
                sampled(
                    _jump_in_front_of_behind_template(
                        agent, goal_reference, background, is_distal, is_in_front
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                    max_to_sample=num_samples,
                )
                for is_distal in BOOL_SET
                for is_in_front in BOOL_SET
            ]
        ),
    )


def _make_jump_over(
    num_samples: int = 5, *, noise_objects: int = 0
) -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[CAN_JUMP])
    object_in_path = standard_object("goal_reference", THING)
    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )

    return phase1_instances(
        "Jump Over",
        flatten(
            [
                sampled(
                    _jump_over_template(agent, object_in_path, background),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                    max_to_sample=num_samples,
                )
            ]
        ),
    )


def make_verb_with_dynamic_prepositions_curriculum(
    num_samples: int = 5, *, num_noise_objects: int = 0
):
    return [
        _make_jump_on(num_samples, noise_objects=num_noise_objects),
        _make_jump_in(num_samples, noise_objects=num_noise_objects),
        _make_jump_beside(num_samples, noise_objects=num_noise_objects),
        _make_jump_in_front_of_behind(num_samples, noise_objects=num_noise_objects),
        _make_jump_over(num_samples, noise_objects=num_noise_objects),
    ]
