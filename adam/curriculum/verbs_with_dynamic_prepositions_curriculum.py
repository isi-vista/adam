from typing import Iterable, Tuple

from immutablecollections import immutableset, ImmutableSet
from more_itertools import flatten

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
    AGENT,
    bigger_than,
    GOAL,
    GAILA_PHASE_1_ONTOLOGY,
    HOLLOW,
    SIT,
    ANIMATE,
    INANIMATE_OBJECT,
    CAN_HAVE_THINGS_RESTING_ON_THEM,
    SIT_THING_SAT_ON,
    CAN_BE_SAT_ON_BY_PEOPLE,
)
from adam.ontology.phase1_spatial_relations import (
    Region,
    INTERIOR,
    GRAVITATIONAL_UP,
    EXTERIOR_BUT_IN_CONTACT,
)
from adam.situation import Action
from adam.situation.templates.phase1_templates import (
    TemplateObjectVariable,
    Phase1SituationTemplate,
    sampled,
)

BOOL_SET = immutableset([True, False])


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
        constraining_relations=[bigger_than(surface, seat)],
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


def _make_sit_on(num_samples: int = 5, *, noise_objects: int = 0) -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[ANIMATE])
    seat = standard_object(
        "seat", INANIMATE_OBJECT, required_properties=[CAN_BE_SAT_ON_BY_PEOPLE]
    )
    surface = standard_object(
        "surface", THING, required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
    )
    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )
    syntax_hints_options = ([], [USE_ADVERBIAL_PATH_MODIFIER])  # type: ignore

    return phase1_instances(
        "Sit On",
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
    )


def _make_sit_in(num_samples: int = 5, *, noise_objects: int = 0) -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[ANIMATE])
    seat = standard_object("seat", INANIMATE_OBJECT, required_properties=[HOLLOW])
    surface = standard_object(
        "surface", THING, required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
    )
    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )
    syntax_hints_options = ([], [USE_ADVERBIAL_PATH_MODIFIER])  # type: ignore

    return phase1_instances(
        "Sit In",
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
    )


def make_verb_with_dynamic_prepositions_curriculum(
    num_samples: int = 5, *, num_noise_objects: int = 0
):
    return [
        _make_sit_on(num_samples, noise_objects=num_noise_objects),
        _make_sit_in(num_samples, noise_objects=num_noise_objects),
    ]
