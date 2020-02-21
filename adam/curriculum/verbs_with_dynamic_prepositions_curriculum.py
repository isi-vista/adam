from itertools import chain

from immutablecollections import immutableset, ImmutableSet
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
from adam.ontology.phase1_ontology import (
    bigger_than,
    FALL,
    GOAL,
    GAILA_PHASE_1_ONTOLOGY,
    HOLLOW,
    THEME,
    CAN_HAVE_THINGS_RESTING_ON_THEM,
)
from adam.ontology.phase1_spatial_relations import (
    Region,
    Direction,
    DISTAL,
    INTERIOR,
    GRAVITATIONAL_UP,
    EXTERIOR_BUT_IN_CONTACT,
    PROXIMAL,
)
from adam.situation import Action
from adam.situation.templates.phase1_templates import (
    TemplateObjectVariable,
    Phase1SituationTemplate,
    sampled,
)

BOOL_SET = immutableset([True, False])


# FALL templates


def _fall_on_template(
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: ImmutableSet[TemplateObjectVariable],
    syntax_hints: ImmutableSet[str],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{theme.handle}-falls-(down)-on-{goal_reference.handle}",
        salient_object_variables=[theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                FALL,
                argument_roles_to_fillers=[
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
            )
        ],
        syntax_hints=syntax_hints,
    )


def _fall_in_template(
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: ImmutableSet[TemplateObjectVariable],
    syntax_hints: ImmutableSet[str],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{theme.handle}-falls-(down)-in-{goal_reference.handle}",
        salient_object_variables=[theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                FALL,
                argument_roles_to_fillers=[
                    (THEME, theme),
                    (GOAL, Region(goal_reference, distance=INTERIOR)),
                ],
            )
        ],
        constraining_relations=[bigger_than(goal_reference, theme)],
        syntax_hints=syntax_hints,
    )


def _fall_beside_template(
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: ImmutableSet[TemplateObjectVariable],
    syntax_hints: ImmutableSet[str],
    is_right: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{theme.handle}-falls-(down)-beside-{goal_reference.handle}",
        salient_object_variables=[theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                FALL,
                argument_roles_to_fillers=[
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
        syntax_hints=syntax_hints,
    )


def _fall_in_front_of_behind_template(
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    background: ImmutableSet[TemplateObjectVariable],
    syntax_hints: ImmutableSet[str],
    is_distal: bool,
    is_in_front: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{theme.handle}-falls-(down)-in-front-of-{goal_reference.handle}",
        salient_object_variables=[theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                FALL,
                argument_roles_to_fillers=[
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
            )
        ],
        syntax_hints=syntax_hints,
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
                        _fall_on_template(theme, goal_on, background, syntax_hints),
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
                        _fall_in_template(theme, goal_in, background, syntax_hints),
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
                            theme, goal_reference, background, syntax_hints, is_right
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
                            syntax_hints,
                            is_distal,
                            is_in_front,
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
    return [_make_fall_with_prepositions(num_samples, noise_objects=num_noise_objects)]
