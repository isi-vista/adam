from adam.curriculum.verbs_with_prepositions_curriculum import (
    _make_go_to,
    _make_go_in,
    _make_go_beside,
    _make_go_in_front_of,
    _make_go_behind,
    _make_go_over,
    _make_go_under,
)
from tests.curriculum.phase1_curriculum_test import curriculum_test


def test_go_to():
    curriculum_test(_make_go_to())


def test_go_in():
    curriculum_test(_make_go_in())


def test_go_beside():
    curriculum_test(_make_go_beside())


def test_go_behind():
    curriculum_test(_make_go_behind())


def test_go_in_front_of():
    curriculum_test(_make_go_in_front_of())


def test_go_over():
    curriculum_test(_make_go_over())


def test_go_under():
    curriculum_test(_make_go_under())
