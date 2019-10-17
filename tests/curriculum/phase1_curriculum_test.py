from adam.curriculum.phase1_curriculum import (
    _Phase1InstanceGroup,
    _make_drink_curriculum,
    _make_eat_curriculum,
    _make_fall_curriculum,
    _make_fly_curriculum,
    _make_go_curriculum,
    _make_jump_curriculum,
    _make_move_curriculum,
    _make_push_curriculum,
    _make_come_curriculum,
    _make_put_curriculum,
    _make_roll_curriculum,
    _make_sit_curriculum,
    _make_spin_curriculum,
    _make_take_curriculum,
    _make_speaker_addressee_curriculum,
    _make_throw_curriculum,
    _make_object_under_or_over_object_curriculum,
    _make_transfer_of_possession_curriculum,
    _make_put_on_speaker_addressee_body_part_curriculum,
    _make_object_beside_object_curriculum,
    _make_behind_in_front_curriculum,
    _make_objects_with_colors_curriculum,
    _make_each_object_by_itself_curriculum,
    _make_multiple_objects_curriculum,
    _make_object_on_ground_curriculum,
    _make_person_has_object_curriculum,
    _make_object_on_object_curriculum,
    _make_object_in_other_object_curriculum,
)


def _test_curriculum(curriculum: _Phase1InstanceGroup) -> None:
    for _ in curriculum.instances():
        # we don't need to do anything
        # the curriculum may be dynamically generated
        # so we just want to test we can instantiate it
        pass


def test_each_object_by_itself_curriculum():
    _test_curriculum(_make_each_object_by_itself_curriculum())


def test_objects_with_colors_curriculum():
    _test_curriculum(_make_objects_with_colors_curriculum())


def test_multiple_objects_curriculum():
    _test_curriculum(_make_multiple_objects_curriculum())


def test_object_on_ground_curriculum():
    _test_curriculum(_make_object_on_ground_curriculum())


def test_person_has_object_curriculum():
    _test_curriculum(_make_person_has_object_curriculum())


def test_fall_curriculum():
    _test_curriculum(_make_fall_curriculum())


def test_transfer_of_possession_curriculum():
    _test_curriculum(_make_transfer_of_possession_curriculum())


def test_object_on_object_curriculum():
    _test_curriculum(_make_object_on_object_curriculum())


def test_object_beside_object_curriculum():
    _test_curriculum(_make_object_beside_object_curriculum())


def test_object_under_or_over_object_curriculum():
    _test_curriculum(_make_object_under_or_over_object_curriculum())


def test_object_in_other_object_curriculum():
    _test_curriculum(_make_object_in_other_object_curriculum())


def test_instantiate_fly_curriculum():
    _test_curriculum(_make_fly_curriculum())


def test_roll_curriculum():
    _test_curriculum(_make_roll_curriculum())


def test_speaker_addressee():
    _test_curriculum(_make_speaker_addressee_curriculum())


def test_jump_curriculum():
    _test_curriculum(_make_jump_curriculum())


def test_put():
    _test_curriculum(_make_put_curriculum())


def test_put_on_speaker_addressee_body_part_curriculum():
    _test_curriculum(_make_put_on_speaker_addressee_body_part_curriculum())


def test_drink_curriculum():
    _test_curriculum(_make_drink_curriculum())


def test_eat_curriculum():
    _test_curriculum(_make_eat_curriculum())


def test_sit_curriculum():
    _test_curriculum(_make_sit_curriculum())


def test_take_curriculum():
    _test_curriculum(_make_take_curriculum())


def test_move_curriculum():
    _test_curriculum(_make_move_curriculum())


def test_spin_curriculum():
    _test_curriculum(_make_spin_curriculum())


def test_go_curriculum():
    _test_curriculum(_make_go_curriculum())


def test_push_curriculum():
    _test_curriculum(_make_push_curriculum())


def test_throw_curriculum():
    _test_curriculum(_make_throw_curriculum())


def test_come_curriculum():
    _test_curriculum(_make_come_curriculum())


def test_behind_in_front_curriculum():
    _test_curriculum(_make_behind_in_front_curriculum())
