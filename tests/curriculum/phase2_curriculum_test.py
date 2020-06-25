from adam.curriculum.curriculum_utils import Phase1InstanceGroup
from adam.curriculum.phase2_curriculum import (
    _make_drink_cups_curriculum,
    _make_put_in_curriculum,
    _make_sit_on_chair_curriculum,
)


def curriculum_test(curriculum: Phase1InstanceGroup) -> None:
    for _ in curriculum.instances():
        # we don't need to do anything
        # the curriculum may be dynamically generated
        # so we just want to test we can instantiate it
        pass


def test_drink_cups_curriculum():
    curriculum_test(_make_drink_cups_curriculum())


def test_put_in_curriculum():
    curriculum_test(_make_put_in_curriculum())


def test_sit_on_chairs_curriculum():
    curriculum_test(_make_sit_on_chair_curriculum())
