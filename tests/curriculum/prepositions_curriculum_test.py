from adam.language_specific.chinese.chinese_language_generator import (
    GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR,
)
from adam.language_specific.english.english_language_generator import (
    GAILA_PHASE_1_LANGUAGE_GENERATOR,
)
import pytest
from adam.curriculum.preposition_curriculum import (
    _make_on_training,
    _make_beside_training,
    _make_under_training,
    _make_over_training,
    _make_in_training,
    _make_in_front_training,
    _make_behind_training,
    _make_in_front_tests,
    _make_behind_tests,
    _make_in_tests,
    _make_over_tests,
    _make_under_tests,
    _make_beside_tests,
    _make_on_tests,
    _make_near_training,
    _make_near_tests,
    _make_far_training,
    _make_far_tests,
)
from tests.curriculum.phase1_curriculum_test import curriculum_test


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_on_training(language_generator):
    curriculum_test(_make_on_training(language_generator=language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_beside_training(language_generator):
    curriculum_test(_make_beside_training(language_generator=language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_under_training(language_generator):
    curriculum_test(_make_under_training(language_generator=language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_over_training(language_generator):
    curriculum_test(_make_over_training(language_generator=language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_in_training(language_generator):
    curriculum_test(_make_in_training(language_generator=language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_behind_training(language_generator):
    curriculum_test(_make_behind_training(language_generator=language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_in_front_training(language_generator):
    curriculum_test(_make_in_front_training(language_generator=language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_near_training(language_generator):
    curriculum_test(_make_near_training(language_generator=language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_far_training(language_generator):
    curriculum_test(_make_far_training(language_generator=language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_on_tests(language_generator):
    curriculum_test(_make_on_tests(language_generator=language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_beside_tests(language_generator):
    curriculum_test(_make_beside_tests(language_generator=language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_under_tests(language_generator):
    curriculum_test(_make_under_tests(language_generator=language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_over_tests(language_generator):
    curriculum_test(_make_over_tests(language_generator=language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_in_tests(language_generator):
    curriculum_test(_make_in_tests(language_generator=language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_behind_tests(language_generator):
    curriculum_test(_make_behind_tests(language_generator=language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_in_front_tests(language_generator):
    curriculum_test(_make_in_front_tests(language_generator=language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_near_tests(language_generator):
    curriculum_test(_make_near_tests(language_generator=language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_far_tests(language_generator):
    curriculum_test(_make_far_tests(language_generator=language_generator))
