"""
Code for working with *curriculum repositories*, which are structured directories containing
saved curricula to be used in experiments.

Curricula are saved and loaded to paths within the repository directory based on the parameters to
the curriculum or script. We use a fixed parameter ordering to insure that curriculum paths are
consistent across different runs and scripts.

Since the curriculum parameters are not currently "sandboxed" away from script parameters, we
include all parameters other than the ones that have been specifically ignored. The user can specify
additional ignored parameters as appropriate. Unrecognized parameters are an error.
"""
from typing import Tuple, AbstractSet
from pathlib import Path
import pickle

from attr import attrs, attrib
from attr.validators import instance_of, deep_iterable

from immutablecollections import immutableset, ImmutableSet
from immutablecollections.converter_utils import _to_tuple
from vistautils.parameters import Parameters

from adam.curriculum.curriculum_utils import Phase1InstanceGroup
from adam.learner.language_mode import LanguageMode, LANGUAGE_MODE_TO_NAME
from adam.pickle import AdamPickler, AdamUnpickler


_PARAMETER_ORDER: ImmutableSet[str] = immutableset(
    [
        "curriculum",
        "num_samples",
        "pursuit-curriculum-params.num_noise_objects",
        "pursuit-curriculum-params.num_instances",
        "pursuit-curriculum-params.num_noise_instances",
        "pursuit-curriculum-params.num_objects_in_instance",
        "pursuit-curriculum-params.add_gaze",
    ]
)


IGNORED_PARAMETERS: ImmutableSet[str] = immutableset(
    [
        "adam_root",
        "adam_experiment_root",
        "experiment",
        "experiment_name",
        "experiment_group_dir",
        "load_from_curriculum_repository",
        "learner",
        "hypothesis_log_dir",
        "include_image_links",
        "accuracy_to_txt",
        "num_pretty_descriptions",
        "sort_learner_descriptions_by_length",
        "save_state_every_n_steps",
        "spack_root",
        "conda_base_path",
        "conda_environment",
        "language_mode",
    ]
)


@attrs
class ExperimentCurriculum:
    """
    Represents a saved curriculum for some experiment.
    """

    train_curriculum: Tuple[Phase1InstanceGroup, ...] = attrib(
        validator=deep_iterable(member_validator=instance_of(Phase1InstanceGroup)),
        converter=_to_tuple,
    )
    test_curriculum: Tuple[Phase1InstanceGroup, ...] = attrib(
        validator=deep_iterable(member_validator=instance_of(Phase1InstanceGroup)),
        converter=_to_tuple,
    )


_EXPERIMENT_CURRICULUM_FILE_NAME = "curriculum.pkl"


def _build_curriculum_path(
    repository: Path,
    parameters: Parameters,
    language_mode: LanguageMode,
    *,
    ignored_parameters: AbstractSet[str] = IGNORED_PARAMETERS,
) -> Path:
    curriculum_file_path: Path = repository / LANGUAGE_MODE_TO_NAME[language_mode]
    unignored = immutableset(
        parameter
        for parameter, _ in parameters.namespaced_items()
        if parameter not in ignored_parameters
    )
    if not unignored.issubset(_PARAMETER_ORDER):
        unrecognized_parameters = unignored.difference(_PARAMETER_ORDER)
        raise RuntimeError(f"No defined order for parameters: {unrecognized_parameters}")

    if "curriculum" not in unignored:
        raise RuntimeError("Expected curriculum name, but none present in parameters.")

    for parameter in iter(_PARAMETER_ORDER):
        value = parameters.get_optional(parameter, object)
        curriculum_file_path = curriculum_file_path / f"{value}_{parameter}"

    return curriculum_file_path / _EXPERIMENT_CURRICULUM_FILE_NAME


def read_experiment_curriculum(
    repository: Path,
    parameters: Parameters,
    language_mode: LanguageMode,
    *,
    ignored_parameters: AbstractSet[str] = IGNORED_PARAMETERS,
) -> ExperimentCurriculum:
    curriculum_file_path = _build_curriculum_path(
        repository, parameters, language_mode, ignored_parameters=ignored_parameters
    )

    with curriculum_file_path.open("rb") as f:
        unpickler = AdamUnpickler(file=f)
        curriculum = unpickler.load()

    return curriculum


def write_experiment_curriculum(
    repository: Path,
    parameters: Parameters,
    language_mode: LanguageMode,
    curriculum: ExperimentCurriculum,
    *,
    ignored_parameters: AbstractSet[str] = IGNORED_PARAMETERS,
):
    curriculum_file_path = _build_curriculum_path(
        repository, parameters, language_mode, ignored_parameters=ignored_parameters
    )
    # Create the parent directory if it doesn't exist, otherwise we can't write to it
    curriculum_file_path.parent.mkdir(exist_ok=True, parents=True)

    with curriculum_file_path.open("wb") as f:
        pickler = AdamPickler(file=f, protocol=pickle.HIGHEST_PROTOCOL)
        pickler.dump(curriculum)
