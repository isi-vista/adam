from itertools import chain
from typing import Sequence, Iterable

from immutablecollections import immutableset
from more_itertools import flatten

from adam.curriculum.curriculum_utils import (
    Phase1InstanceGroup,
    learner_template_factory,
    PHASE1_CHOOSER_FACTORY,
    phase1_instances,
    standard_object,
)
from adam.ontology import THING
from adam.ontology.phase1_ontology import bigger_than, GAILA_PHASE_1_ONTOLOGY
from adam.situation.templates.phase1_templates import (
    TemplateObjectVariable,
    Phase1SituationTemplate,
    sampled,
)

BOOL_SET = immutableset([True, False])


def _big_x_template(
    theme: TemplateObjectVariable, background: Iterable[TemplateObjectVariable]
) -> Phase1SituationTemplate:
    learner = learner_template_factory()
    computed_background = [learner]
    computed_background.extend(background)
    return Phase1SituationTemplate(
        f"big-{theme.handle}",
        salient_object_variables=[theme],
        background_object_variables=computed_background,
        asserted_always_relations=[bigger_than(theme, learner)],
    )


def _small_x_template(
    theme: TemplateObjectVariable, background: Iterable[TemplateObjectVariable]
) -> Phase1SituationTemplate:
    learner = learner_template_factory()
    computed_background = [learner]
    computed_background.extend(background)
    return Phase1SituationTemplate(
        f"small-{theme.handle}",
        salient_object_variables=[theme],
        background_object_variables=computed_background,
        asserted_always_relations=[bigger_than(learner, theme)],
    )


def make_imprecise_size_descriptions(
    num_samples: int = 5, *, num_noise_objects: int = 0
) -> Phase1InstanceGroup:
    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(num_noise_objects)
    )

    theme = standard_object("theme", THING)
    big_small_template = [_big_x_template, _small_x_template]

    return phase1_instances(
        "Imprecise Size",
        chain(
            # Big, Small
            flatten(
                [
                    sampled(
                        template(theme, background),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples,
                    )
                    for template in big_small_template
                ]
            ),
            # Tall, Short
            # flatten(),
        ),
    )


def make_imprecise_temporal_descriptions(
    num_samples: int = 5, *, num_noise_objects: int = 0
) -> Sequence[Phase1InstanceGroup]:
    """
    One particular instantiation of the Imprecise Temporal Descriptions Curriculum
    """
    return []
