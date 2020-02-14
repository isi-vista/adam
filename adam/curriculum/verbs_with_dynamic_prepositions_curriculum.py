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
from adam.ontology.during import DuringAction
from adam.ontology.phase1_ontology import (
    AGENT,
    bigger_than,
    FLY,
    GOAL,
    GAILA_PHASE_1_ONTOLOGY,
    HOLLOW,
    ANIMATE,
    INANIMATE_OBJECT,
    CAN_HAVE_THINGS_RESTING_ON_THEM,
    SIT_THING_SAT_ON,
    CAN_BE_SAT_ON_BY_PEOPLE,
    near,
    CAN_FLY,
)
from adam.ontology.phase1_spatial_relations import (
    Region,
    Direction,
    DISTAL,
    INTERIOR,
    GRAVITATIONAL_UP,
    EXTERIOR_BUT_IN_CONTACT,
    VIA,
    SpatialPath,
    PROXIMAL,
)
from adam.situation import Action
from adam.situation.templates.phase1_templates import (
    TemplateObjectVariable,
    Phase1SituationTemplate,
    sampled,
)

BOOL_SET = immutableset([True, False])


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
        actions=[
            Action(
                FLY,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (GOAL, Region(goal_reference, distance=INTERIOR)),
                ],
            )
        ],
        constraining_relations=[bigger_than(goal_reference, agent)],
    )


def _fly_beside_template(
    # "A bird flies (along) beside a table"
    agent: TemplateObjectVariable,
    object_passed: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    is_distal: bool,
    is_right: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-flies-beside-{object_passed.handle}",
        salient_object_variables=[agent],
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
                                reference_object=Region(
                                    object_passed,
                                    distance=DISTAL if is_distal else PROXIMAL,
                                    direction=Direction(
                                        positive=is_right,
                                        relative_to_axis=HorizontalAxisOfObject(
                                            object_passed,
                                            index=0
                                        ),
                                    ),
                                ),
                                reference_axis=HorizontalAxisOfObject(
                                    object_passed,
                                    index=0
                                ),
                            ),
                        )
                    ]
                ),
            )
        ],
    )


def _fly_behind_template(
    # "A bird flies (along) behind a truck"
    agent: TemplateObjectVariable,
    object_passed: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    is_distal: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-flies-behind-{object_passed.handle}",
        salient_object_variables=[agent],
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
                                reference_object=Region(
                                    object_passed,
                                    distance=DISTAL if is_distal else PROXIMAL,
                                    direction=Direction(
                                        positive=False,
                                        relative_to_axis=FacingAddresseeAxis(
                                            object_passed
                                        ),
                                    ),
                                ),
                                reference_axis=FacingAddresseeAxis(object_passed),
                            ),
                        )
                    ]
                ),
            )
        ],
    )


def _make_fly_in(num_samples: int = 5, *, noise_objects: int = 0) -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[CAN_FLY])
    goal_reference = standard_object(
        "goal_reference", THING, required_properties=[HOLLOW]
    )
    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )

    return phase1_instances(
        "Fly In",
        flatten(
            [
                sampled(
                    _fly_in_template(agent, goal_reference, background),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                    max_to_sample=num_samples,
                )
            ]
        ),
    )


def _make_fly_beside(
    num_samples: int = 5, *, noise_objects: int = 0
) -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[CAN_FLY])
    object_passed = standard_object("object_passed", THING)
    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )

    return phase1_instances(
        "Fly Beside",
        flatten(
            [
                sampled(
                    _fly_beside_template(
                        agent, object_passed, background, is_distal, is_right
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


def _make_fly_behind(
    num_samples: int = 5, *, noise_objects: int = 0
) -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[CAN_FLY])
    object_passed = standard_object("object_passed", THING)
    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )

    return phase1_instances(
        "Fly Behind",
        flatten(
            [
                sampled(
                    _fly_behind_template(agent, object_passed, background, is_distal),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                    max_to_sample=num_samples,
                )
                for is_distal in BOOL_SET
            ]
        ),
    )


def make_verb_with_dynamic_prepositions_curriculum(
    num_samples: int = 5, *, num_noise_objects: int = 0
):
    return [
        _make_fly_in(num_samples, noise_objects=num_noise_objects),
        _make_fly_over(num_samples, noise_objects=num_noise_objects),
        _make_fly_under(num_samples, noise_objects=num_noise_objects),
        _make_fly_beside(num_samples, noise_objects=num_noise_objects),
        _make_fly_behind(num_samples, noise_objects=num_noise_objects),
        _make_fly_up_down(num_samples, noise_objects=num_noise_objects),
    ]
