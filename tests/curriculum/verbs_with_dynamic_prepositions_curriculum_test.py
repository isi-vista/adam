from adam.curriculum.verbs_with_dynamic_prepositions_curriculum import (
    _make_push_with_prepositions,
    _make_go_with_prepositions,
    _make_throw_with_prepositions,
    _make_roll_with_prepositions,
    _make_sit_with_prepositions,
    _make_take_with_prepositions,
    _make_fall_with_prepositions,
    _make_put_with_prepositions,
    _make_move_with_prepositions,
    _make_jump_with_prepositions,
    _make_fly_with_prepositions,
    _make_come_with_prepositions,
)
from tests.curriculum.phase1_curriculum_test import curriculum_test
from adam.language_specific.chinese.chinese_language_generator import (
    GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR,
)
from adam.language_specific.english.english_language_generator import (
    GAILA_PHASE_1_LANGUAGE_GENERATOR,
)
import pytest


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_make_push(language_generator):
    curriculum_test(_make_push_with_prepositions(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_make_go(language_generator):
    curriculum_test(_make_go_with_prepositions(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_make_throw(language_generator):
    curriculum_test(_make_throw_with_prepositions(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_make_roll(language_generator):
    curriculum_test(_make_roll_with_prepositions(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_make_sit(language_generator):
    curriculum_test(_make_sit_with_prepositions(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_make_take(language_generator):
    curriculum_test(_make_take_with_prepositions(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_make_fall(language_generator):
    curriculum_test(_make_fall_with_prepositions(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_make_put(language_generator):
    curriculum_test(_make_put_with_prepositions(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_make_move(language_generator):
    curriculum_test(_make_move_with_prepositions(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_make_jump(language_generator):
    curriculum_test(_make_jump_with_prepositions(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_make_fly(language_generator):
    curriculum_test(_make_fly_with_prepositions(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_make_come(language_generator):
    curriculum_test(_make_come_with_prepositions(None, None, language_generator))
