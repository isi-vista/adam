from typing import Iterable, Tuple
from pathlib import Path
import pickle
import logging

from immutablecollections import immutableset, ImmutableSet
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


Curriculum = Iterable[Phase1InstanceGroup]
ExperimentCurriculum = Tuple[Curriculum, Curriculum]


_EXPERIMENT_CURRICULUM_FILE_NAME = "curriculum.pkl"


def _build_curriculum_path(
    repository: Path, parameters: Parameters, language_mode: LanguageMode
) -> Path:
    path: Path = repository / LANGUAGE_MODE_TO_NAME[language_mode]

    all_parameters = immutableset(
        parameter for parameter, _ in parameters.namespaced_items()
    )
    ignored = all_parameters - _PARAMETER_ORDER
    logging.info(f"Ignoring parameters: {ignored}")

    parameters_present = all_parameters.intersection(_PARAMETER_ORDER)
    if "curriculum" not in parameters_present:
        raise RuntimeError("Expected curriculum name, but none present in parameters.")

    # Would it make sense to ignore None values? Could that ever be ambiguous?
    for parameter in iter(_PARAMETER_ORDER):
        value = parameters.get_optional(parameter, object)
        path = path / f"{value}_{parameter}"

    return path / _EXPERIMENT_CURRICULUM_FILE_NAME


def read_experiment_curriculum(
    repository: Path, parameters: Parameters, language_mode: LanguageMode
) -> ExperimentCurriculum:
    path = _build_curriculum_path(repository, parameters, language_mode)

    with path.open("rb") as f:
        unpickler = AdamUnpickler(file=f)
        curriculum = unpickler.load()

    return curriculum


def write_experiment_curriculum(
    repository: Path,
    parameters: Parameters,
    language_mode: LanguageMode,
    curriculum: ExperimentCurriculum,
):
    path = _build_curriculum_path(repository, parameters, language_mode)
    # Create the parent directory if it doesn't exist, otherwise we can't write to it
    path.parent.mkdir(exist_ok=True, parents=True)

    with path.open("wb") as f:
        pickler = AdamPickler(file=f, protocol=pickle.HIGHEST_PROTOCOL)
        pickler.dump(curriculum)
