from adam.curriculum.verbs_with_dynamic_prepositions_curriculum import (
    _make_push_with_prepositions,
    _make_go_with_prepositions,
    _make_roll_with_prepositions,
    _make_sit_with_prepositions,
    _make_fall_with_prepositions,
)
from tests.curriculum.phase1_curriculum_test import curriculum_test


def test_make_push():
    curriculum_test(_make_push_with_prepositions())


def test_make_go():
    curriculum_test(_make_go_with_prepositions())


def test_make_roll():
    curriculum_test(_make_roll_with_prepositions())


def test_make_sit():
    curriculum_test(_make_sit_with_prepositions())


def test_make_fall():
    curriculum_test(_make_fall_with_prepositions())
