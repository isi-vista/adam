from typing import List, Iterable

from immutablecollections import immutableset
from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

from adam.curriculum import ExplicitWithSituationInstanceGroup
from adam.curriculum.curriculum_utils import Phase1InstanceGroup
from adam.experiment.log_experiment import curriculum_from_params
from adam.experiment.curriculum_repository import (
    write_experiment_curriculum,
    IGNORED_PARAMETERS,
)
from adam.learner.language_mode import LanguageMode


CURRICULUM_REPOSITORY_PATH_PARAMETER = "curriculum_repository_path"
LANGUAGE_MODE_PARAMETER = "language_mode"


def evaluate_curriculum(
    lazy_curriculum: Iterable[Phase1InstanceGroup]
) -> List[Phase1InstanceGroup]:
    strict_curriculum: List[Phase1InstanceGroup] = []
    for instance_group in lazy_curriculum:
        # We assume that the instance groups all specify Situations since otherwise you can't run
        # experiments on them.
        strict_curriculum.append(
            ExplicitWithSituationInstanceGroup(
                instance_group.name(), tuple(instance_group.instances())
            )
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
            IGNORED_PARAMETERS.union(
                {CURRICULUM_REPOSITORY_PATH_PARAMETER, LANGUAGE_MODE_PARAMETER}
            )
        ),
    )


if __name__ == "__main__":
    parameters_only_entry_point(main)
