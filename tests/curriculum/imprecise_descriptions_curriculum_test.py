from adam.curriculum.imprecise_descriptions_curriculum import (
    make_imprecise_size_descriptions,
)
from tests.curriculum.phase1_curriculum_test import curriculum_test


def test_imprecise_size_descriptions_curriculum():
    curriculum_test(make_imprecise_size_descriptions())
