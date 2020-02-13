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
    GOAL,
    GAILA_PHASE_1_ONTOLOGY,
    HOLLOW,
    MOVE,
    ANIMATE,
    INANIMATE_OBJECT,
    CAN_HAVE_THINGS_RESTING_ON_THEM,
    THEME,
    contacts,
    HAS_SPACE_UNDER,
    SELF_MOVING,
)
from adam.ontology.phase1_spatial_relations import (
    Region,
    INTERIOR,
    GRAVITATIONAL_DOWN,
    EXTERIOR_BUT_IN_CONTACT,
    DISTAL,
    PROXIMAL,
    Direction,
    GRAVITATIONAL_UP,
)
from adam.situation import Action
from adam.situation.templates.phase1_templates import (
    TemplateObjectVariable,
    Phase1SituationTemplate,
    sampled,
)

BOOL_SET = immutableset([True, False])


def _move_beside_template(
    # "A baby moves beside a car"
    agent: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    is_distal: bool,
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
            )
        ],
    )


def _move_in_front_of_behind_template(
    # "Mom moves in front of a house"
    agent: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
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
                                relative_to_axis=FacingAddresseeAxis(
                                    goal_reference, index=0
                                ),
                            ),
                        ),
                    ),
                ],
            )
        ],
    )


def _move_under_template(
    # "A dog moves under a chair"
    agent: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
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
        constraining_relations=[bigger_than(goal_reference, agent)],
    )


def _move_thing_in_template(
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
                during=DuringAction(continuously=[contacts(agent, theme)]),
            )
        ],
        constraining_relations=[bigger_than(goal_reference, theme)],
    )


def _move_thing_on_template(
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
                during=DuringAction(continuously=[contacts(agent, theme)]),
            )
        ],
        constraining_relations=[bigger_than(goal_reference, theme)],
    )


def _move_thing_under_template(
    # "A baby moves a car under a chair"
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
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
                during=DuringAction(continuously=[contacts(agent, theme)]),
            )
        ],
        constraining_relations=[bigger_than(goal_reference, theme)],
    )


def _move_thing_beside_template(
    # "A dog moves a cookie beside a baby"
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    is_distal: bool,
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
                during=DuringAction(continuously=[contacts(agent, theme)]),
            )
        ],
    )


def _move_thing_in_front_of_behind_template(
    # "Dad moves a chair behind a house"
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
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
                                relative_to_axis=FacingAddresseeAxis(
                                    goal_reference, index=0
                                ),
                            ),
                        ),
                    ),
                ],
                during=DuringAction(continuously=[contacts(agent, theme)]),
            )
        ],
    )


def _make_move_beside(
    num_samples: int = 5, *, noise_objects: int = 0
) -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[SELF_MOVING])
    goal_reference = standard_object("goal_reference", THING)
    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )

    return phase1_instances(
        "Move Beside",
        flatten(
            [
                sampled(
                    _move_beside_template(
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


def _make_move_in_front_of_behind(
    num_samples: int = 5, *, noise_objects: int = 0
) -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[SELF_MOVING])
    goal_reference = standard_object("goal_reference", THING)
    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )

    return phase1_instances(
        "Move In Front Of Behind",
        flatten(
            [
                sampled(
                    _move_in_front_of_behind_template(
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


def _make_move_under(
    num_samples: int = 5, *, noise_objects: int = 0
) -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[SELF_MOVING])
    goal_reference = standard_object(
        "goal_reference", THING, required_properties=[HAS_SPACE_UNDER]
    )
    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )

    return phase1_instances(
        "Move Under",
        flatten(
            [
                sampled(
                    _move_under_template(agent, goal_reference, background, is_distal),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                    max_to_sample=num_samples,
                )
                for is_distal in BOOL_SET
            ]
        ),
    )


def _make_move_thing_in(
    num_samples: int = 5, *, noise_objects: int = 0
) -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[ANIMATE])
    theme = standard_object("theme", THING, required_properties=[INANIMATE_OBJECT])
    goal_reference = standard_object(
        "goal_reference", THING, required_properties=[HOLLOW]
    )
    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )

    return phase1_instances(
        "Move Thing In",
        flatten(
            [
                sampled(
                    _move_thing_in_template(agent, theme, goal_reference, background),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                    max_to_sample=num_samples,
                )
            ]
        ),
    )


def _make_move_thing_on(
    num_samples: int = 5, *, noise_objects: int = 0
) -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[ANIMATE])
    theme = standard_object("theme", THING, required_properties=[INANIMATE_OBJECT])
    goal_reference = standard_object(
        "goal_reference", THING, required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
    )
    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )

    return phase1_instances(
        "Move Thing In",
        flatten(
            [
                sampled(
                    _move_thing_on_template(agent, theme, goal_reference, background),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                    max_to_sample=num_samples,
                )
            ]
        ),
    )


def _make_move_thing_under(
    num_samples: int = 5, *, noise_objects: int = 0
) -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[ANIMATE])
    theme = standard_object("theme", THING, required_properties=[INANIMATE_OBJECT])
    goal_reference = standard_object(
        "goal_reference", THING, required_properties=[HAS_SPACE_UNDER]
    )
    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )

    return phase1_instances(
        "Move Thing Under",
        flatten(
            [
                sampled(
                    _move_thing_under_template(
                        agent, theme, goal_reference, background, is_distal
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                    max_to_sample=num_samples,
                )
                for is_distal in BOOL_SET
            ]
        ),
    )


def _make_move_thing_beside(
    num_samples: int = 5, *, noise_objects: int = 0
) -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[ANIMATE])
    theme = standard_object("theme", THING, required_properties=[INANIMATE_OBJECT])
    goal_reference = standard_object("goal_reference", THING)
    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )

    return phase1_instances(
        "Move Thing Beside",
        flatten(
            [
                sampled(
                    _move_thing_beside_template(
                        agent, theme, goal_reference, background, is_distal, is_right
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


def _make_move_thing_in_front_of_behind(
    num_samples: int = 5, *, noise_objects: int = 0
) -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[ANIMATE])
    theme = standard_object("theme", THING, required_properties=[INANIMATE_OBJECT])
    goal_reference = standard_object("goal_reference", THING)
    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )

    return phase1_instances(
        "Move Thing In Front Of Behind",
        flatten(
            [
                sampled(
                    _move_thing_in_front_of_behind_template(
                        agent, theme, goal_reference, background, is_distal, is_in_front
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


def make_verb_with_dynamic_prepositions_curriculum(
    num_samples: int = 5, *, num_noise_objects: int = 0
):
    return [
        _make_move_beside(num_samples, noise_objects=num_noise_objects),
        _make_move_in_front_of_behind(num_samples, noise_objects=num_noise_objects),
        _make_move_under(num_samples, noise_objects=num_noise_objects),
        _make_move_thing_in(num_samples, noise_objects=num_noise_objects),
        _make_move_thing_on(num_samples, noise_objects=num_noise_objects),
        _make_move_thing_beside(num_samples, noise_objects=num_noise_objects),
        _make_move_thing_in_front_of_behind(num_samples, noise_objects=num_noise_objects),
        _make_move_thing_under(num_samples, noise_objects=num_noise_objects),
    ]
