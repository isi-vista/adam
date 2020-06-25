from adam.curriculum.phase1_curriculum import (
    Phase1InstanceGroup,
    _make_behind_in_front_curriculum,
    _make_come_curriculum,
    _make_drink_curriculum,
    _make_each_object_by_itself_curriculum,
    _make_eat_curriculum,
    _make_fall_curriculum,
    _make_fly_curriculum,
    _make_go_curriculum,
    _make_jump_curriculum,
    _make_move_curriculum,
    _make_object_beside_object_curriculum,
    _make_object_in_other_object_curriculum,
    _make_object_on_ground_curriculum,
    _make_object_on_object_curriculum,
    _make_object_under_or_over_object_curriculum,
    _make_objects_with_colors_curriculum,
    _make_person_has_object_curriculum,
    _make_push_curriculum,
    _make_put_curriculum,
    _make_put_on_speaker_addressee_body_part_curriculum,
    _make_roll_curriculum,
    _make_sit_curriculum,
    _make_speaker_addressee_curriculum,
    _make_spin_curriculum,
    _make_take_curriculum,
    _make_throw_curriculum,
    _make_transfer_of_possession_curriculum,
    _make_objects_with_colors_is_curriculum,
    _make_pass_curriculum,
    _make_part_whole_curriculum,
    _make_plural_objects_curriculum,
    _make_generic_statements_curriculum,
)


def curriculum_test(curriculum: Phase1InstanceGroup) -> None:
    for _ in curriculum.instances():
        # we don't need to do anything
        # the curriculum may be dynamically generated
        # so we just want to test we can instantiate it
        pass


def test_each_object_by_itself_curriculum():
    curriculum_test(_make_each_object_by_itself_curriculum())


def test_objects_with_colors_curriculum():
    curriculum_test(_make_objects_with_colors_curriculum())


def test_objects_with_colors_is_curriculum():
    cur = _make_objects_with_colors_is_curriculum().instances()
    for c in cur:
        assert c[1].as_token_sequence()[2] == "is"
    curriculum_test(_make_objects_with_colors_is_curriculum())


def test_instantiate_fly_curriculum():
    curriculum_test(_make_fly_curriculum())


def test_plural_objects_curriculum():
    curriculum_test(_make_plural_objects_curriculum())


def test_object_on_ground_curriculum():
    curriculum_test(_make_object_on_ground_curriculum())


def test_person_has_object_curriculum():
    curriculum_test(_make_person_has_object_curriculum())


def test_part_whole_curriculum():
    curriculum_test(_make_part_whole_curriculum())


def test_fall_curriculum():
    curriculum_test(_make_fall_curriculum())


def test_transfer_of_possession_curriculum():
    curriculum_test(_make_transfer_of_possession_curriculum())


def test_object_on_object_curriculum():
    curriculum_test(_make_object_on_object_curriculum())


def test_object_beside_object_curriculum():
    curriculum_test(_make_object_beside_object_curriculum())


def test_object_under_or_over_object_curriculum():
    curriculum_test(_make_object_under_or_over_object_curriculum())


def test_object_in_other_object_curriculum():
    curriculum_test(_make_object_in_other_object_curriculum())


def test_roll_curriculum():
    curriculum_test(_make_roll_curriculum())


def test_speaker_addressee():
    curriculum_test(_make_speaker_addressee_curriculum())


def test_jump_curriculum():
    curriculum_test(_make_jump_curriculum())


def test_put():
    curriculum_test(_make_put_curriculum())


def test_put_on_speaker_addressee_body_part_curriculum():
    curriculum_test(_make_put_on_speaker_addressee_body_part_curriculum())


def test_drink_curriculum():
    curriculum_test(_make_drink_curriculum())


def test_eat_curriculum():
    curriculum_test(_make_eat_curriculum())


def test_sit_curriculum():
    curriculum_test(_make_sit_curriculum())


def test_take_curriculum():
    curriculum_test(_make_take_curriculum())


def test_move_curriculum():
    curriculum_test(_make_move_curriculum())


def test_spin_curriculum():
    curriculum_test(_make_spin_curriculum())


def test_go_curriculum():
    curriculum_test(_make_go_curriculum())


def test_push_curriculum():
    curriculum_test(_make_push_curriculum())


def test_throw_curriculum():
    curriculum_test(_make_throw_curriculum())


def test_pass_curriculum():
    curriculum_test(_make_pass_curriculum())


def test_come_curriculum():
    curriculum_test(_make_come_curriculum())


def test_behind_in_front_curriculum():
    curriculum_test(_make_behind_in_front_curriculum())


def test_generics_curriculum():
    curriculum_test(_make_generic_statements_curriculum())
