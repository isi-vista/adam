from adam.curriculum.attribute_constraining_action_curriculum import (
    make_german_eat_test_curriculum,
    make_animal_eat_curriculum,
    make_human_eat_curriculum,
)
from tests.curriculum.phase1_curriculum_test import curriculum_test
from adam.language_specific.english.english_language_generator import (
    GAILA_PHASE_1_LANGUAGE_GENERATOR,
)
from adam.language_specific.chinese.chinese_language_generator import (
    GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR,
)
import pytest


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_human_eat_curriculum(language_generator):
    curriculum_test(make_human_eat_curriculum(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_animal_eat_curriculum(language_generator):
    curriculum_test(make_animal_eat_curriculum(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_eat_test_curriculum(language_generator):
    curriculum_test(make_german_eat_test_curriculum(None, None, language_generator))
