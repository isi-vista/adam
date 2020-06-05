from itertools import chain
from typing import Sequence

from immutablecollections import immutableset
from more_itertools import flatten

from adam.curriculum.curriculum_utils import (
    Phase1InstanceGroup,
    PHASE1_CHOOSER_FACTORY,
    phase1_instances,
    standard_object,
)
from adam.curriculum.phase1_curriculum import (
    throw_on_ground_template,
    throw_template,
    throw_up_down_template,
    throw_to_template,
    bare_move_template,
    transitive_move_template,
    make_jump_template,
    intransitive_roll,
    transitive_roll_with_surface,
    transitive_roll,
    bare_fly,
    fall_on_ground_template,
    falling_template,
)
from adam.language_specific.english.english_language_generator import (
    USE_ADVERBIAL_PATH_MODIFIER,
)
from adam.ontology import THING
from adam.ontology.phase1_ontology import (
    GAILA_PHASE_1_ONTOLOGY,
    ANIMATE,
    INANIMATE,
    BOX,
    FAST,
    SLOW,
    SELF_MOVING,
    CAN_JUMP,
    ROLLABLE,
    CAN_HAVE_THINGS_RESTING_ON_THEM,
    BIRD,
)
from adam.situation.templates.phase1_templates import sampled

BOOL_SET = immutableset([True, False])

# TODO: Renable big and small after future considerations
# TODO: See https://github.com/isi-vista/adam/issues/742
# def _big_x_template(
#   theme: TemplateObjectVariable, background: Iterable[TemplateObjectVariable]
# ) -> Phase1SituationTemplate:
#    learner = learner_template_factory()
#    computed_background = [learner]
#    computed_background.extend(background)
#    return Phase1SituationTemplate(
#        f"big-{theme.handle}",
#        salient_object_variables=[theme],
#        background_object_variables=computed_background,
#        asserted_always_relations=[bigger_than(theme, learner)],
#    )


# def _small_x_template(
#    theme: TemplateObjectVariable, background: Iterable[TemplateObjectVariable]
# ) -> Phase1SituationTemplate:
#    learner = learner_template_factory()
#    computed_background = [learner]
#    computed_background.extend(background)
#    return Phase1SituationTemplate(
#        f"small-{theme.handle}",
#        salient_object_variables=[theme],
#        background_object_variables=computed_background,
#        asserted_always_relations=[bigger_than(learner, theme)],
#    )


# def make_imprecise_size_descriptions(
#    num_samples: int = 5, *, num_noise_objects: int = 0
# ) -> Phase1InstanceGroup:
#    background = immutableset(
#        standard_object(f"noise_object_{x}") for x in range(num_noise_objects)
#    )

#    theme = standard_object("theme", THING)
#    big_small_template = [_big_x_template, _small_x_template]

#    return phase1_instances(
#        "Imprecise Size",
#        chain(
# Big, Small
#           flatten(
#                [
#                    sampled(
#                        template(theme, background),
#                        ontology=GAILA_PHASE_1_ONTOLOGY,
#                        chooser=PHASE1_CHOOSER_FACTORY(),
#                        max_to_sample=num_samples,
#                    )
#                    for template in big_small_template
#                ]
#            ),
# Tall, Short
# flatten(),
#        ),
#    )


def make_throw_imprecise_temporal_descriptions(
    num_samples: int = 5, *, num_noise_objects: int = 0  # pylint:disable=unused-argument
) -> Phase1InstanceGroup:
    thrower = standard_object("thrower_0", THING, required_properties=[ANIMATE])
    catcher = standard_object("catcher_0", THING, required_properties=[ANIMATE])
    object_thrown = standard_object("object_0", required_properties=[INANIMATE])
    implicit_goal_reference = standard_object("implicit_throw_goal_object", BOX)

    return phase1_instances(
        "throwing-with-temporal-descriptions",
        chain(
            # Throw on Ground
            flatten(
                sampled(
                    throw_on_ground_template(
                        thrower,
                        object_thrown,
                        spatial_properties=[FAST] if is_fast else [SLOW],
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples,
                )
                for is_fast in BOOL_SET
            ),
            # Throw
            flatten(
                sampled(
                    throw_template(
                        thrower,
                        object_thrown,
                        implicit_goal_reference,
                        spatial_properties=[FAST] if is_fast else [SLOW],
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples,
                )
                for is_fast in BOOL_SET
            ),
            # Throw up, down
            flatten(
                sampled(
                    throw_up_down_template(
                        thrower,
                        object_thrown,
                        implicit_goal_reference,
                        is_up=is_up,
                        spatial_properties=[FAST] if is_fast else [SLOW],
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples,
                )
                for is_fast in BOOL_SET
                for is_up in BOOL_SET
            ),
            # Throw To
            flatten(
                sampled(
                    throw_to_template(
                        thrower,
                        object_thrown,
                        catcher,
                        spatial_properties=[FAST] if is_fast else [SLOW],
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples,
                )
                for is_fast in BOOL_SET
            ),
        ),
    )


def make_move_imprecise_temporal_descriptions(
    num_samples: int = 5, *, num_noise_objects: int = 0  # pylint:disable=unused-argument
) -> Phase1InstanceGroup:
    self_mover_0 = standard_object(
        "self-mover_0", THING, required_properties=[SELF_MOVING]
    )

    other_mover_0 = standard_object("mover_0", THING, required_properties=[ANIMATE])
    movee_0 = standard_object("movee_0", THING, required_properties=[INANIMATE])
    move_goal_reference = standard_object(
        "move-goal-reference", THING, required_properties=[INANIMATE]
    )

    return phase1_instances(
        "move-with-temporal-descriptions",
        chain(
            # bare move (e.g. "a box moves") is about half of uses in child speed
            flatten(
                sampled(
                    bare_move_template(
                        self_mover_0,
                        move_goal_reference,
                        spatial_properties=[FAST] if is_fast else [SLOW],
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples,
                )
                for is_fast in BOOL_SET
            ),
            # Transitive Move
            flatten(
                sampled(
                    transitive_move_template(
                        other_mover_0,
                        movee_0,
                        move_goal_reference,
                        spatial_properties=[FAST] if is_fast else [SLOW],
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples,
                )
                for is_fast in BOOL_SET
            ),
        ),
    )


def make_jump_imprecise_temporal_descriptions(
    num_samples: int = 5, *, num_noise_objects: int = 0  # pylint:disable=unused-argument
) -> Phase1InstanceGroup:
    jumper = standard_object("jumper_0", THING, required_properties=[CAN_JUMP])

    return phase1_instances(
        "jumping",
        chain(
            flatten(
                [
                    sampled(
                        # "A person jumps"
                        make_jump_template(
                            jumper,
                            use_adverbial_path_modifier=use_adverbial_path_modifier,
                            spatial_properties=[FAST] if is_fast else [SLOW],
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples,
                    )
                    for use_adverbial_path_modifier in (True, False)
                    for is_fast in BOOL_SET
                ]
            )
        ),
    )


def make_roll_imprecise_temporal_descriptions(
    num_samples: int = 5, *, num_noise_objects: int = 0  # pylint:disable=unused-argument
) -> Phase1InstanceGroup:
    animate_0 = standard_object("object_0", THING, required_properties=[ANIMATE])
    rollable_0 = standard_object("object_1", required_properties=[ROLLABLE])
    rolling_surface = standard_object(
        "surface", THING, required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
    )

    return phase1_instances(
        "roll-imprecise-temporal-descriptions",
        chain(
            # rolls intransitively
            flatten(
                sampled(
                    intransitive_roll(
                        animate_0,
                        rolling_surface,
                        spatial_properties=[FAST] if is_fast else [SLOW],
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples,
                )
                for is_fast in BOOL_SET
            ),
            # rolls transitively
            flatten(
                sampled(
                    transitive_roll(
                        animate_0,
                        rollable_0,
                        rolling_surface,
                        spatial_properties=[FAST] if is_fast else [SLOW],
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples,
                )
                for is_fast in BOOL_SET
            ),
            # rolls on a surface
            flatten(
                sampled(
                    transitive_roll_with_surface(
                        animate_0,
                        rollable_0,
                        rolling_surface,
                        spatial_properties=[FAST] if is_fast else [SLOW],
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples,
                )
                for is_fast in BOOL_SET
            ),
        ),
    )


def make_fly_imprecise_temporal_descriptions(
    num_samples: int = 5, *, num_noise_objects: int = 0  # pylint:disable=unused-argument
) -> Phase1InstanceGroup:
    bird = standard_object("bird_0", BIRD)
    syntax_hints_options = ([], [USE_ADVERBIAL_PATH_MODIFIER])  # type: ignore

    return phase1_instances(
        "fly-imprecise-temporal-descripttions",
        chain(
            # Bare Fly
            flatten(
                sampled(
                    bare_fly(
                        bird,
                        up=is_up,
                        syntax_hints=syntax_hints,
                        spatial_properties=[FAST] if is_fast else [SLOW],
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples,
                )
                for is_up in BOOL_SET
                for syntax_hints in syntax_hints_options
                for is_fast in BOOL_SET
            )
        ),
    )


def make_fall_imprecise_temporal_descriptions(
    num_samples: int = 5, *, num_noise_objects: int = 0  # pylint:disable=unused-argument
) -> Phase1InstanceGroup:
    arbitary_object = standard_object("object_0", THING)
    syntax_hints_options = ([], [USE_ADVERBIAL_PATH_MODIFIER])  # type: ignore

    return phase1_instances(
        f"fall-imprecise-temporal-description",
        chain(
            # Any Object Falling
            flatten(
                sampled(
                    falling_template(
                        arbitary_object,
                        lands_on_ground=object_ends_up_on_ground,
                        syntax_hints=syntax_hints,
                        spatial_properties=[FAST] if is_fast else [SLOW],
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples,
                )
                for object_ends_up_on_ground in BOOL_SET
                for syntax_hints in syntax_hints_options
                for is_fast in BOOL_SET
            ),
            # Fall on Ground
            flatten(
                sampled(
                    fall_on_ground_template(
                        arbitary_object, spatial_properties=[FAST] if is_fast else [SLOW]
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples,
                )
                for is_fast in BOOL_SET
            ),
        ),
    )


def make_imprecise_temporal_descriptions(
    num_samples: int = 5, *, num_noise_objects: int = 0
) -> Sequence[Phase1InstanceGroup]:
    """
    One particular instantiation of the Imprecise Temporal Descriptions Curriculum
    """
    return [
        make_throw_imprecise_temporal_descriptions(
            num_samples, num_noise_objects=num_noise_objects
        ),
        make_move_imprecise_temporal_descriptions(
            num_samples, num_noise_objects=num_noise_objects
        ),
        make_jump_imprecise_temporal_descriptions(
            num_samples, num_noise_objects=num_noise_objects
        ),
        make_roll_imprecise_temporal_descriptions(
            num_samples, num_noise_objects=num_noise_objects
        ),
        make_fly_imprecise_temporal_descriptions(
            num_samples, num_noise_objects=num_noise_objects
        ),
        make_fall_imprecise_temporal_descriptions(
            num_samples, num_noise_objects=num_noise_objects
        ),
    ]
