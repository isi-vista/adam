from adam.curriculum.phase1_curriculum import (
    GAILA_PHASE_1_CURRICULUM,
    _make_fly_curriculum,
)


def test_instantiate_curriculum():
    for sub_curriculum in GAILA_PHASE_1_CURRICULUM:
        for _ in sub_curriculum.instances():
            # we don't need to do anything
            # the curriculum may be dynamically generated
            # so we just want to test we can instantiate it
            pass


def test_instantiate_fly_curriculum():
    for _ in _make_fly_curriculum().instances():
        pass
