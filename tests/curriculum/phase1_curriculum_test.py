from adam.curriculum.phase1_curriculum import (
    GAILA_PHASE_1_CURRICULUM,
    _Phase1InstanceGroup,
    _make_fly_curriculum,
    _make_roll_curriculum,
    _make_jump_curriculum,
    _make_drink_curriculum,
    _make_eat_curriculum,
    _make_transfer_of_possession_curriculum,
    _make_sit_curriculum,
    _make_put_curriculum,
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


def test_eat_curriculum():
    _test_curriculum(_make_eat_curriculum())


def test_transfer_of_possession():
    _test_curriculum(_make_transfer_of_possession_curriculum())


def test_sit():
    _test_curriculum(_make_sit_curriculum())


def test_put():
    _test_curriculum(_make_put_curriculum())
