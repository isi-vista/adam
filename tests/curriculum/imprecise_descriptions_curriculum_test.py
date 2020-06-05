import pytest

from adam.curriculum.imprecise_descriptions_curriculum import (
    make_move_imprecise_temporal_descriptions,
    make_throw_imprecise_temporal_descriptions,
    make_jump_imprecise_temporal_descriptions,
    make_roll_imprecise_temporal_descriptions,
    make_fly_imprecise_temporal_descriptions,
    make_fall_imprecise_temporal_descriptions,
)
from tests.curriculum.phase1_curriculum_test import curriculum_test


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
