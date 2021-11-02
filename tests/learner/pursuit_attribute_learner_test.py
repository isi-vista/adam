from itertools import chain
import random

import pytest
from more_itertools import flatten

from adam.curriculum.curriculum_utils import (
    standard_object,
    phase1_instances,
    CHOOSER_FACTORY,
    TEST_CHOOSER_FACTORY,
)
from adam.curriculum.phase1_curriculum import _object_with_color_template
from adam.language.language_utils import phase1_language_generator
from adam.learner import LearningExample
from adam.learner.attributes import PursuitAttributeLearner
from adam.learner.integrated_learner import SymbolicIntegratedTemplateLearner
from adam.learner.language_mode import LanguageMode
from adam.ontology.phase1_ontology import (
    RED,
    WHITE,
    BLACK,
    GREEN,
    BLUE,
    BALL,
    BOOK,
    GAILA_PHASE_1_ONTOLOGY,
    CAT,
    BIRD,
)
from adam.situation.templates.phase1_templates import property_variable, sampled
from tests.learner import LANGUAGE_MODE_TO_TEMPLATE_LEARNER_OBJECT_RECOGNIZER


def pursuit_learner_factory(language_mode: LanguageMode):
    rng = random.Random()
    rng.seed(0)
    return SymbolicIntegratedTemplateLearner(
        object_learner=LANGUAGE_MODE_TO_TEMPLATE_LEARNER_OBJECT_RECOGNIZER[language_mode],
        attribute_learner=PursuitAttributeLearner(
            learning_factor=0.05,
            graph_match_confirmation_threshold=0.7,
            lexicon_entry_threshold=0.7,
            rng=rng,
            smoothing_parameter=0.002,
            ontology=GAILA_PHASE_1_ONTOLOGY,
            language_mode=language_mode,
            rank_gaze_higher=False,
        ),
    )


@pytest.mark.parametrize(
    "color_node,object_0_node,object_1_node",
    [
        (RED, BALL, CAT),
        (BLUE, BALL, CAT),
        (GREEN, BALL, CAT),
        (BLACK, BOOK, BIRD),
        (WHITE, BOOK, BIRD),
    ],
)
@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize("learner", [pursuit_learner_factory])
def test_pursuit_color_attribute(
    color_node, object_0_node, object_1_node, language_mode, learner
):
    color = property_variable(f"{color_node.handle}", color_node)
    object_0 = standard_object(
        f"{object_0_node.handle}", object_0_node, added_properties=[color]
    )
    object_1 = standard_object(
        f"{object_1_node.handle}", object_1_node, added_properties=[color]
    )

    color_object_template = _object_with_color_template(object_0, None)

    templates_with_n_samples = [
        (color_object_template, 2),
        (_object_with_color_template(object_1, None), 4),
    ]

    language_generator = phase1_language_generator(language_mode)

    color_train_curriculum = phase1_instances(
        f"{color.handle} Color Train",
        language_generator=language_generator,
        situations=chain(
            *[
                flatten(
                    [
                        sampled(
                            template,
                            chooser=CHOOSER_FACTORY(),
                            ontology=GAILA_PHASE_1_ONTOLOGY,
                            max_to_sample=n_samples,
                            block_multiple_of_the_same_type=True,
                        )
                        for template, n_samples in templates_with_n_samples
                    ]
                )
            ]
        ),
    )

    color_test_curriculum = phase1_instances(
        f"{color.handle} Color Test",
        situations=sampled(
            color_object_template,
            chooser=TEST_CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            max_to_sample=1,
            block_multiple_of_the_same_type=True,
        ),
        language_generator=language_generator,
    )

    processing_learner = learner(language_mode)

    for (
        _,
        linguistic_description,
        perceptual_representation,
    ) in color_train_curriculum.instances():
        processing_learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )

    for (
        _,
        test_lingustics_description,
        test_perceptual_representation,
    ) in color_test_curriculum.instances():
        descriptions_from_learner = processing_learner.describe(
            test_perceptual_representation
        )
        gold = test_lingustics_description.as_token_sequence()
        assert descriptions_from_learner
        assert gold in [desc.as_token_sequence() for desc in descriptions_from_learner]
