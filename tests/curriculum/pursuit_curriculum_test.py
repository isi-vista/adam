from adam.curriculum.pursuit_curriculum import make_simple_pursuit_curriculum
from tests.curriculum.phase1_curriculum_test import curriculum_test
import pytest
from adam.language_specific.chinese.chinese_language_generator import (
    GAILA_PHASE_2_CHINESE_LANGUAGE_GENERATOR,
)
from adam.language_specific.english.english_language_generator import (
    GAILA_PHASE_2_LANGUAGE_GENERATOR,
)


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_2_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_2_LANGUAGE_GENERATOR],
)
def test_simple_pursuit_curriculum(language_generator):
    curriculum_test(make_simple_pursuit_curriculum(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_2_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_2_LANGUAGE_GENERATOR],
)
def test_simple_pursuit_curriculum_with_noise(language_generator):
    curriculum_test(make_simple_pursuit_curriculum(None, 2, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_2_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_2_LANGUAGE_GENERATOR],
)
def test_simple_pursuit_curriculum_all_variables(language_generator):
    curriculum_test(
        make_simple_pursuit_curriculum(
            num_instances=15,
            num_objects_in_instance=4,
            num_noise_instances=2,
            language_generator=language_generator,
        )
    )
