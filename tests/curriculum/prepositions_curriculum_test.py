from adam.curriculum.preposition_curriculum import (
    _make_on_training,
    _make_beside_training,
    _make_under_training,
    _make_over_training,
    _make_in_training,
    _make_in_front_training,
    _make_behind_training,
)
from tests.curriculum.phase1_curriculum_test import curriculum_test


def test_on_training():
    curriculum_test(_make_on_training())


def test_beside_training():
    curriculum_test(_make_beside_training())


def test_under_training():
    curriculum_test(_make_under_training())


def test_over_training():
    curriculum_test(_make_over_training())


def test_in_training():
    curriculum_test(_make_in_training())


def test_behind_training():
    curriculum_test(_make_behind_training())


def test_in_front_training():
    curriculum_test(_make_in_front_training())
