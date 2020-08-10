from itertools import chain

import pytest

from adam.curriculum.curriculum_utils import (
    phase1_instances,
    PHASE1_CHOOSER_FACTORY,
    PHASE1_TEST_CHOOSER_FACTORY,
)
from adam.language.language_utils import phase1_language_generator
from adam.curriculum.imprecise_descriptions_curriculum import (
    _big_x_template,
    _little_x_template,
    _short_x_template,
    _tall_x_template,
)
from adam.learner import LearningExample

# from adam.learner.verbs import SubsetVerbLearnerNew
from adam.learner.attributes import SubsetAttributeLearner, SubsetAttributeLearnerNew
from adam.learner.integrated_learner import IntegratedTemplateLearner
from adam.learner.language_mode import LanguageMode
from adam.learner.objects import ObjectRecognizerAsTemplateLearner
from adam.ontology.phase1_ontology import GAILA_PHASE_1_ONTOLOGY
from adam.situation.templates.phase1_templates import sampled
from tests.learner import object_recognizer_factory


def subset_attribute_leaner_factory(language_mode: LanguageMode):
    return SubsetAttributeLearner(
        object_recognizer=object_recognizer_factory(language_mode),
        ontology=GAILA_PHASE_1_ONTOLOGY,
        language_mode=language_mode,
    )


def integrated_learner_factory(language_mode: LanguageMode):
    return IntegratedTemplateLearner(
        object_learner=ObjectRecognizerAsTemplateLearner(
            object_recognizer=object_recognizer_factory(language_mode),
            language_mode=language_mode,
        ),
        attribute_learner=SubsetAttributeLearnerNew(
            ontology=GAILA_PHASE_1_ONTOLOGY, beam_size=5, language_mode=language_mode
        ),
        # action_learner=SubsetVerbLearnerNew(
        #    ontology=GAILA_PHASE_1_ONTOLOGY, beam_size=5, language_mode=language_mode
        # ),
    )


def run_imprecise_test(learner, situation_template, language_generator):
    train_curriculum = phase1_instances(
        "train",
        chain(
            *[
                sampled(
                    situation_template,
                    max_to_sample=10,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    # this is a hack since our current object recognizer will throw a runtime error if there are percieved objects not in the description
                    block_multiple_of_the_same_type=False,
                )
            ]
        ),
        language_generator=language_generator,
    )
    test_curriculum = phase1_instances(
        "test",
        chain(
            *[
                sampled(
                    situation_template,
                    max_to_sample=1,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_TEST_CHOOSER_FACTORY(),
                    block_multiple_of_the_same_type=False,
                )
            ]
        ),
        language_generator=language_generator,
    )

    for (
        _,
        linguistic_description,
        perceptual_representation,
    ) in train_curriculum.instances():
        # Get the object matches first - preposition learner can't learn without already recognized objects
        learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )
    for (
        _,
        test_lingustics_description,
        test_perceptual_representation,
    ) in test_curriculum.instances():
        descriptions_from_learner = learner.describe(test_perceptual_representation)
        gold = test_lingustics_description.as_token_sequence()
        assert descriptions_from_learner
        assert gold in [desc.as_token_sequence() for desc in descriptions_from_learner]


@pytest.mark.parametrize(
    "learner", [subset_attribute_leaner_factory, integrated_learner_factory]
)
@pytest.mark.parametrize("language", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
def test_tall(learner, language):
    run_imprecise_test(
        learner(language),
        _tall_x_template(background=[]),
        language_generator=phase1_language_generator(language),
    )


@pytest.mark.parametrize(
    "learner", [subset_attribute_leaner_factory, integrated_learner_factory]
)
@pytest.mark.parametrize("language", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
def test_short(learner, language):
    run_imprecise_test(
        learner(language),
        _short_x_template(background=[]),
        language_generator=phase1_language_generator(language),
    )


@pytest.mark.parametrize(
    "learner", [subset_attribute_leaner_factory, integrated_learner_factory]
)
@pytest.mark.parametrize("language", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
def test_big(learner, language):
    run_imprecise_test(
        learner(language),
        _big_x_template(background=[]),
        language_generator=phase1_language_generator(language),
    )


@pytest.mark.parametrize(
    "learner", [subset_attribute_leaner_factory, integrated_learner_factory]
)
@pytest.mark.parametrize("language", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
def test_small(learner, language):
    run_imprecise_test(
        learner(language),
        _little_x_template(background=[]),
        language_generator=phase1_language_generator(language),
    )
