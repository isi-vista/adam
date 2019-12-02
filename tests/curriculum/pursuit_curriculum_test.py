from adam.curriculum.pursuit_curriculum import make_simple_pursuit_curriculum
from tests.curriculum.phase1_curriculum_test import curriculum_test


def test_simple_pursuit_curriculum():
    curriculum_test(make_simple_pursuit_curriculum())


def test_simple_pursuit_curriculum_with_noise():
    curriculum_test(make_simple_pursuit_curriculum(num_noise_instances=2))


def test_simple_pursuit_curriculum_all_variables():
    curriculum_test(
        make_simple_pursuit_curriculum(
            num_instances=15, num_objects_in_instance=4, num_noise_instances=2
        )
    )
