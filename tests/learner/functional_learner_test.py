import pytest

from adam.curriculum.phase2_curriculum import _make_sit_on_chair_curriculum

# from adam.experiment.experiment_utils import _make_sit_on_curriculum
from adam.language.language_utils import phase2_language_generator
from adam.learner import LanguageMode, LearningExample
from adam.learner.functional_learner import FunctionalLearner
from adam.learner.integrated_learner import SymbolicIntegratedTemplateLearner
from adam.learner.verbs import SubsetVerbLearner
from adam.ontology.phase2_ontology import GAILA_PHASE_2_ONTOLOGY
from tests.learner import LANGUAGE_MODE_TO_TEMPLATE_LEARNER_OBJECT_RECOGNIZER


def integrated_learner_factory(language_mode: LanguageMode):
    return SymbolicIntegratedTemplateLearner(
        object_learner=LANGUAGE_MODE_TO_TEMPLATE_LEARNER_OBJECT_RECOGNIZER[language_mode],
        action_learner=SubsetVerbLearner(
            ontology=GAILA_PHASE_2_ONTOLOGY, beam_size=5, language_mode=language_mode
        ),
        functional_learner=FunctionalLearner(language_mode=language_mode),
    )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
def test_functional_learner(language_mode: LanguageMode):
    # TODO: currently the _make_sit_curriculum defaults to bed instead of chair so chair isn't predicted in the testing
    sit_train = _make_sit_on_chair_curriculum(
        5, 0, phase2_language_generator(language_mode)
    )
    sit_test = _make_sit_on_chair_curriculum(
        1, 0, phase2_language_generator(language_mode)
    )

    learner = integrated_learner_factory(language_mode)

    for (_, linguistic_description, perceptual_representation) in sit_train.instances():
        learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )

    for (_, linguistic_description, perceptual_representation) in sit_test.instances():
        descriptions_from_learner = learner.describe(perceptual_representation)
        gold = linguistic_description.as_token_sequence()
        assert descriptions_from_learner
        assert gold in [desc.as_token_sequence() for desc in descriptions_from_learner]
