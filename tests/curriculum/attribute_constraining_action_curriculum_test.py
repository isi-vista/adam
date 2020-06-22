from adam.curriculum.attribute_constraining_action_curriculum import (
    make_german_eat_test_curriculum,
    make_animal_eat_curriculum,
    make_human_eat_curriculum,
)
from tests.curriculum.phase1_curriculum_test import curriculum_test


def test_human_eat_curriculum():
    curriculum_test(make_human_eat_curriculum())


def test_animal_eat_curriculum():
    curriculum_test(make_animal_eat_curriculum())


def test_eat_test_curriculum():
    curriculum_test(make_german_eat_test_curriculum())
