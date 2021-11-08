import pytest

from adam.curriculum.curriculum_utils import Phase3InstanceGroup
from adam.curriculum.phase3_curriculum import (
    phase_3_one_objects_curriculum,
    phase_3_m4_stretch_eval,
    phase_3_one_core_objects_curriculum,
    phase_3_one_stretch_objects_curriculum,
    phase_3_m4_core_eval,
)
from adam.language_specific.english.english_language_generator import (
    GAILA_PHASE_3_LANGUAGE_GENERATOR,
)


def curriculum_phase3_test(curriculum: Phase3InstanceGroup) -> None:
    for _ in curriculum.instances():
        # we don't need to do anything
        # the curriculum may be dynamically generated
        # so we just want to test we can instantiate it
        pass


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_3_LANGUAGE_GENERATOR],
)
@pytest.mark.parametrize(
    "curriculum",
    [
        phase_3_one_objects_curriculum,
        phase_3_m4_stretch_eval,
        phase_3_one_core_objects_curriculum,
        phase_3_one_stretch_objects_curriculum,
        phase_3_m4_core_eval,
    ],
)
def test_phase3_curriculum(language_generator, curriculum):
    curriculum_phase3_test(curriculum(1, 5, language_generator))
