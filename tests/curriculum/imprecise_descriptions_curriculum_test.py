from adam.curriculum.imprecise_descriptions_curriculum import (
    make_move_imprecise_temporal_descriptions,
    make_throw_imprecise_temporal_descriptions,
    make_jump_imprecise_temporal_descriptions,
    make_roll_imprecise_temporal_descriptions,
    make_fly_imprecise_temporal_descriptions,
    make_fall_imprecise_temporal_descriptions,
    make_imprecise_size_descriptions,
    make_walk_run_subtle_verb_distinction,
    make_pass_toss_subtle_verb_distinction,
    make_push_shove_subtle_verb_distinctions,
    make_take_grab_subtle_verb_distinction,
    make_eat_big_small_curriculum,
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
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_make_eat_big_small_curriculum(language_generator):
    curriculum_test(make_eat_big_small_curriculum(language_generator=language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test__pass_toss_subtle_verb_distinction(language_generator):
    curriculum_test(
        make_pass_toss_subtle_verb_distinction(language_generator=language_generator)
    )


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test__push_shove_subtle_verb_distinction(language_generator):
    curriculum_test(
        make_push_shove_subtle_verb_distinctions(language_generator=language_generator)
    )


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test__take_grab_subtle_verb_distinction(language_generator):
    curriculum_test(
        make_take_grab_subtle_verb_distinction(language_generator=language_generator)
    )


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test__walk_run_subtle_verb_distinction(language_generator):
    curriculum_test(
        make_walk_run_subtle_verb_distinction(language_generator=language_generator)
    )


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test__throw_imprecise_temporal_descriptions_curriculum(language_generator):
    curriculum_test(
        make_throw_imprecise_temporal_descriptions(language_generator=language_generator)
    )


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_move_imprecise_temporal_descriptions_curriculum(language_generator):
    curriculum_test(
        make_move_imprecise_temporal_descriptions(language_generator=language_generator)
    )


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_jump_imprecise_temporal_descriptions_curriculum(language_generator):
    curriculum_test(
        make_jump_imprecise_temporal_descriptions(language_generator=language_generator)
    )


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_roll_imprecise_temporal_descriptions_curriculum(language_generator):
    curriculum_test(
        make_roll_imprecise_temporal_descriptions(language_generator=language_generator)
    )


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_fly_imprecise_temporal_descriptions_curriculum(language_generator):
    curriculum_test(
        make_fly_imprecise_temporal_descriptions(language_generator=language_generator)
    )


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_fall_imprecise_temporal_descriptions_curriculum(language_generator):
    curriculum_test(
        make_fall_imprecise_temporal_descriptions(language_generator=language_generator)
    )


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_imprecise_size_descriptions_curriculum(language_generator):
    curriculum_test(
        make_imprecise_size_descriptions(language_generator=language_generator)
    )
