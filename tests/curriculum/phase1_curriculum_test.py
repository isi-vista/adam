from typing import List

from adam.curriculum.phase1_curriculum import (
    GAILA_PHASE_1_CURRICULUM,
    _make_fly_curriculum,
    _Phase1InstanceGroup, _make_roll_curriculum)

def _test_curriculum(curriculum: List[_Phase1InstanceGroup]) -> None:
    for _ in curriculum.instances():
        # we don't need to do anything
        # the curriculum may be dynamically generated
        # so we just want to test we can instantiate it
        pass

def test_instantiate_curriculum():
    _test_curriculum(GAILA_PHASE_1_CURRICULUM)


def test_instantiate_fly_curriculum():
    _test_curriculum(_make_fly_curriculum())

def test_roll_curriculum():
    _test_curriculum(_make_roll_curriculum())
