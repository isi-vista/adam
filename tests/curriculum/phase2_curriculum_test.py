from adam.curriculum.curriculum_utils import Phase1InstanceGroup
from adam.curriculum.phase2_curriculum import (
    _make_drink_cups_curriculum,
    _make_put_in_curriculum,
    _make_sit_on_chair_curriculum,
    make_multiple_object_situation,
)
from adam.language_specific.chinese.chinese_language_generator import (
    GAILA_PHASE_2_CHINESE_LANGUAGE_GENERATOR,
)
from adam.language_specific.english.english_language_generator import (
    GAILA_PHASE_2_LANGUAGE_GENERATOR,
)
import pytest


def curriculum_test(curriculum: Phase1InstanceGroup) -> None:
    for _ in curriculum.instances():
        # we don't need to do anything
        # the curriculum may be dynamically generated
        # so we just want to test we can instantiate it
        pass


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_2_LANGUAGE_GENERATOR, GAILA_PHASE_2_CHINESE_LANGUAGE_GENERATOR],
)
def test_drink_cups_curriculum(language_generator):
    curriculum_test(_make_drink_cups_curriculum(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_2_LANGUAGE_GENERATOR, GAILA_PHASE_2_CHINESE_LANGUAGE_GENERATOR],
)
def test_put_in_curriculum(language_generator):
    curriculum_test(_make_put_in_curriculum(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_2_LANGUAGE_GENERATOR, GAILA_PHASE_2_CHINESE_LANGUAGE_GENERATOR],
)
def test_sit_on_chairs_curriculum(language_generator):
    curriculum_test(_make_sit_on_chair_curriculum(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_2_LANGUAGE_GENERATOR, GAILA_PHASE_2_CHINESE_LANGUAGE_GENERATOR],
)
def test_multiple_objects_curriculum(language_generator):
    curriculum_test(make_multiple_object_situation(None, None, language_generator))
