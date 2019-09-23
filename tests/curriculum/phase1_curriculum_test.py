from adam.curriculum.phase1_curriculum import (
    GAILA_PHASE_1_CURRICULUM,
    _Phase1InstanceGroup,
    _make_fly_curriculum,
    _make_roll_curriculum,
    _make_jump_curriculum,
    _make_drink_curriculum,
)


def _test_curriculum(curriculum: _Phase1InstanceGroup) -> None:
    for _ in curriculum.instances():
        # we don't need to do anything
        # the curriculum may be dynamically generated
        # so we just want to test we can instantiate it
        pass


def test_instantiate_curriculum():
    for subcurriculum in GAILA_PHASE_1_CURRICULUM:
        _test_curriculum(subcurriculum)


def test_instantiate_fly_curriculum():
    _test_curriculum(_make_fly_curriculum())


def test_roll_curriculum():
    _test_curriculum(_make_roll_curriculum())


def test_jump_curriculum():
    _test_curriculum(_make_jump_curriculum())


def test_drink_curriculum():
    _test_curriculum(_make_drink_curriculum())
