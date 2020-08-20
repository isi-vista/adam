from typing import Iterable, Tuple, AbstractSet
from pathlib import Path
import pickle

from immutablecollections import immutableset, ImmutableSet
from vistautils.parameters import Parameters

from adam.curriculum.curriculum_utils import Phase1InstanceGroup
from adam.learner.language_mode import LanguageMode, LANGUAGE_MODE_TO_NAME
from adam.pickle import AdamPickler, AdamUnpickler


_PARAMETER_ORDER: ImmutableSet[str] = immutableset([
    "curriculum",
    "num_samples",

    # Pursuit parameters
    "num_noise_objects",
    "num_instances",
    "num_noise_instances",
    "num_objects_in_instance",
    "add_gaze",
])


IGNORED_PARAMETERS: ImmutableSet[str] = immutableset(
    [
        "adam_root",
        "adam_experiment_root",
        "experiment",
        "experiment_name",
        "experiment_group_dir",
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


Curriculum = Iterable[Phase1InstanceGroup]
ExperimentCurriculum = Tuple[Curriculum, Curriculum]


_EXPERIMENT_CURRICULUM_FILE_NAME = "curriculum.pkl"


def _build_curriculum_path(
    repository: Path,
    parameters: Parameters,
    language_mode: LanguageMode,
    *,
    ignored_parameters: AbstractSet[str] = IGNORED_PARAMETERS,
) -> Path:
    path: Path = repository / LANGUAGE_MODE_TO_NAME[language_mode]
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
        unqualified_name: str = parameter.split(".")[-1]
        value = parameters.get_optional(parameter, object)
        path = path / f"{value}_{unqualified_name}"

    return path / _EXPERIMENT_CURRICULUM_FILE_NAME


def read_experiment_curriculum(
    repository: Path,
    parameters: Parameters,
    language_mode: LanguageMode,
    *,
    ignored_parameters: AbstractSet[str] = IGNORED_PARAMETERS,
) -> ExperimentCurriculum:
    path = _build_curriculum_path(
        repository, parameters, language_mode, ignored_parameters=ignored_parameters
    )

    with path.open("rb") as f:
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
    path = _build_curriculum_path(
        repository, parameters, language_mode, ignored_parameters=ignored_parameters
    )
    # Create the parent directory if it doesn't exist, otherwise we can't write to it
    path.parent().mkdir(exist_ok=True, parents=True)

    with path.open("wb") as f:
        pickler = AdamPickler(file=f, protocol=pickle.HIGHEST_PROTOCOL)
        pickler.dump(curriculum)
