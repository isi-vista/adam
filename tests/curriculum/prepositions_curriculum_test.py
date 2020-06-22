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


def test_near_training():
    curriculum_test(_make_near_training())


def test_on_tests():
    curriculum_test(_make_on_tests())


def test_beside_tests():
    curriculum_test(_make_beside_tests())


def test_under_tests():
    curriculum_test(_make_under_tests())


def test_over_tests():
    curriculum_test(_make_over_tests())


def test_in_tests():
    curriculum_test(_make_in_tests())


def test_behind_tests():
    curriculum_test(_make_behind_tests())


def test_in_front_tests():
    curriculum_test(_make_in_front_tests())


def test_near_tests():
    curriculum_test(_make_near_tests())
