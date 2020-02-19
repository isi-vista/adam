from adam.curriculum.verbs_with_dynamic_prepositions_curriculum import (
    _make_push_with_prepositions,
)
from tests.curriculum.phase1_curriculum_test import curriculum_test


def test_make_push():
    curriculum_test(_make_push_with_prepositions())
