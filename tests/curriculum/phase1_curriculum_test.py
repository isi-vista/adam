import pytest

from adam.curriculum.phase1_curriculum import (
    Phase1InstanceGroup,
    _make_behind_in_front_curriculum,
    _make_come_curriculum,
    _make_drink_curriculum,
    _make_each_object_by_itself_curriculum,
    _make_eat_curriculum,
    _make_fall_curriculum,
    _make_fly_curriculum,
    _make_generic_statements_curriculum,
    _make_go_curriculum,
    _make_jump_curriculum,
    _make_move_curriculum,
    _make_my_your_object_curriculum,
    _make_object_beside_object_curriculum,
    _make_object_in_other_object_curriculum,
    _make_object_on_ground_curriculum,
    _make_object_on_object_curriculum,
    _make_object_under_or_over_object_curriculum,
    _make_objects_with_colors_curriculum,
    _make_objects_with_colors_is_curriculum,
    _make_part_whole_curriculum,
    _make_pass_curriculum,
    _make_person_has_object_curriculum,
    _make_plural_objects_curriculum,
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
    _make_transitive_roll_curriculum,
)
from adam.language_specific.chinese.chinese_language_generator import (
    GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR,
)
from adam.language_specific.english.english_language_generator import (
    GAILA_PHASE_1_LANGUAGE_GENERATOR,
)


def curriculum_test(curriculum: Phase1InstanceGroup) -> None:
    for _ in curriculum.instances():
        # we don't need to do anything
        # the curriculum may be dynamically generated
        # so we just want to test we can instantiate it
        pass


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_each_object_by_itself_curriculum(language_generator):
    curriculum_test(
        _make_each_object_by_itself_curriculum(None, None, language_generator)
    )


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_my_your_object_curriculum(language_generator):
    curriculum_test(_make_my_your_object_curriculum(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_transitive_roll_curriculum(language_generator):
    curriculum_test(_make_transitive_roll_curriculum(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_objects_with_colors_curriculum(language_generator):
    curriculum_test(_make_objects_with_colors_curriculum(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_objects_with_colors_is_curriculum(language_generator):
    cur = _make_objects_with_colors_is_curriculum(
        None, None, language_generator
    ).instances()
    for c in cur:
        assert (
            language_generator == GAILA_PHASE_1_LANGUAGE_GENERATOR
            and c[1].as_token_sequence()[2] == "is"
        ) or (
            language_generator == GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR
            and c[1].as_token_sequence()[1] == "shr4"
        )
    curriculum_test(
        _make_objects_with_colors_is_curriculum(None, None, language_generator)
    )


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_instantiate_fly_curriculum(language_generator):
    curriculum_test(_make_fly_curriculum(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_plural_objects_curriculum(language_generator):
    curriculum_test(_make_plural_objects_curriculum(language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_object_on_ground_curriculum(language_generator):
    curriculum_test(_make_object_on_ground_curriculum(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_person_has_object_curriculum(language_generator):
    curriculum_test(_make_person_has_object_curriculum(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_part_whole_curriculum(language_generator):
    curriculum_test(_make_part_whole_curriculum(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_fall_curriculum(language_generator):
    curriculum_test(_make_fall_curriculum(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_transfer_of_possession_curriculum(language_generator):
    curriculum_test(
        _make_transfer_of_possession_curriculum(None, None, language_generator)
    )


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_object_on_object_curriculum(language_generator):
    curriculum_test(_make_object_on_object_curriculum(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_object_beside_object_curriculum(language_generator):
    curriculum_test(_make_object_beside_object_curriculum(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_object_under_or_over_object_curriculum(language_generator):
    curriculum_test(
        _make_object_under_or_over_object_curriculum(None, None, language_generator)
    )


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_object_in_other_object_curriculum(language_generator):
    curriculum_test(
        _make_object_in_other_object_curriculum(None, None, language_generator)
    )


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_roll_curriculum(language_generator):
    curriculum_test(_make_roll_curriculum(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_speaker_addressee(language_generator):
    curriculum_test(_make_speaker_addressee_curriculum(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_jump_curriculum(language_generator):
    curriculum_test(_make_jump_curriculum(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_put(language_generator):
    curriculum_test(_make_put_curriculum(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_put_on_speaker_addressee_body_part_curriculum(language_generator):
    curriculum_test(
        _make_put_on_speaker_addressee_body_part_curriculum(
            None, None, language_generator
        )
    )


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_drink_curriculum(language_generator):
    curriculum_test(_make_drink_curriculum(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_eat_curriculum(language_generator):
    curriculum_test(_make_eat_curriculum(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_sit_curriculum(language_generator):
    curriculum_test(_make_sit_curriculum(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_take_curriculum(language_generator):
    curriculum_test(_make_take_curriculum(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_move_curriculum(language_generator):
    curriculum_test(_make_move_curriculum(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_spin_curriculum(language_generator):
    curriculum_test(_make_spin_curriculum(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_go_curriculum(language_generator):
    curriculum_test(_make_go_curriculum(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_push_curriculum(language_generator):
    curriculum_test(_make_push_curriculum(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_throw_curriculum(language_generator):
    curriculum_test(_make_throw_curriculum(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_pass_curriculum(language_generator):
    curriculum_test(_make_pass_curriculum(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_come_curriculum(language_generator):
    curriculum_test(_make_come_curriculum(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_behind_in_front_curriculum(language_generator):
    curriculum_test(_make_behind_in_front_curriculum(None, None, language_generator))


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_generics_curriculum(language_generator):
    curriculum_test(_make_generic_statements_curriculum(None, None, language_generator))
