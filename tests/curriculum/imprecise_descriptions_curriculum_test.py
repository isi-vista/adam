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
)
from tests.curriculum.phase1_curriculum_test import curriculum_test


def test__pass_toss_subtle_verb_distinction():
    curriculum_test(make_pass_toss_subtle_verb_distinction())


def test__push_shove_subtle_verb_distinction():
    curriculum_test(make_push_shove_subtle_verb_distinctions())


def test__take_grab_subtle_verb_distinction():
    curriculum_test(make_take_grab_subtle_verb_distinction())


def test__walk_run_subtle_verb_distinction():
    curriculum_test(make_walk_run_subtle_verb_distinction())


def test__throw_imprecise_temporal_descriptions_curriculum():
    curriculum_test(make_throw_imprecise_temporal_descriptions())


def test_move_imprecise_temporal_descriptions_curriculum():
    curriculum_test(make_move_imprecise_temporal_descriptions())


def test_jump_imprecise_temporal_descriptions_curriculum():
    curriculum_test(make_jump_imprecise_temporal_descriptions())


def test_roll_imprecise_temporal_descriptions_curriculum():
    curriculum_test(make_roll_imprecise_temporal_descriptions())


def test_fly_imprecise_temporal_descriptions_curriuclum():
    curriculum_test(make_fly_imprecise_temporal_descriptions())


def test_fall_imprecise_temporal_descriptions_curriculum():
    curriculum_test(make_fall_imprecise_temporal_descriptions())


def test_imprecise_size_descriptions_curriculum():
    curriculum_test(make_imprecise_size_descriptions())
