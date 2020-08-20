from typing import List, Iterable

from immutablecollections import immutableset
from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

from adam.curriculum import ExplicitWithoutSituationInstanceGroup
from adam.curriculum.curriculum_utils import Phase1InstanceGroup
from adam.experiment.log_experiment import curriculum_from_params
from adam.experiment.curriculum_repository import write_experiment_curriculum
from adam.learner.language_mode import LanguageMode


CURRICULUM_REPOSITORY_PATH_PARAMETER = "curriculum_repository_path"
LANGUAGE_MODE_PARAMETER = "language_mode"


def evaluate_curriculum(
    lazy_curriculum: Iterable[Phase1InstanceGroup]
) -> List[Phase1InstanceGroup]:
    strict_curriculum: List[Phase1InstanceGroup] = []
    for instance_group in lazy_curriculum:
        # For now we convert each instance group to an `ExplicitWithoutSituationInstanceGroup`.
        #
        # Is there any reason we shouldn't have an `ExplicitInstanceGroup` where the situation is
        # optional?
        instances = tuple(
            (linguistic_description, perceptual_representation)
            for _, linguistic_description, perceptual_representation in instance_group.instances()
        )
        strict_curriculum.append(
            ExplicitWithoutSituationInstanceGroup(instance_group.name(), instances)
        )
    return strict_curriculum


def main(params: Parameters):
    curriculum_repository_path = params.creatable_directory(
        CURRICULUM_REPOSITORY_PATH_PARAMETER
    )
    language_mode = params.enum(
        LANGUAGE_MODE_PARAMETER, LanguageMode, default=LanguageMode.ENGLISH
    )

    train_curriculum, test_curriculum = curriculum_from_params(
        params, language_mode=language_mode
    )
    strict_curriculum = (
        evaluate_curriculum(train_curriculum),
        evaluate_curriculum(test_curriculum),
    )
    write_experiment_curriculum(
        curriculum_repository_path,
        params,
        language_mode,
        strict_curriculum,
        ignored_parameters=immutableset(
            [CURRICULUM_REPOSITORY_PATH_PARAMETER, LANGUAGE_MODE_PARAMETER]
        ),
    )


if __name__ == "__main__":
    parameters_only_entry_point(main)
