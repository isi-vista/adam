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
        "train_curriculum.block_multiple_of_same_type",
        "train_curriculum.include_targets_in_noise",
        "train_curriculum.min_noise_relations",
        "train_curriculum.max_noise_relations",
        "train_curriculum.add_noise",
        "train_curriculum.shuffled",
        "train_curriculum.include_attributes",
        "train_curriculum.include_relations",
        "train_curriculum.random_seed",
        "train_curriculum.chooser_seed",
        "train_curriculum.min_noise_objects",
        "train_curriculum.max_noise_objects",
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
        "log_hypothesis_every_n_steps",
        "learner_logging_path",
        "log_learner_state",
        "resume_from_latest_logged_state",
        "post_observer.include_acc_observer",
        "post_observer.include_pr_observer",
        "post_observer.log_pr",
        "test_observer.accuracy_to_txt",
        "object_learner.learner_type",
        "object_learner.ontology",
        "object_learner.random_seed",
        "object_learner.learning_factor",
        "object_learner.graph_match_confirmation_threshold",
        "object_learner.lexicon_entry_threshold",
        "object_learner.smoothing_parameter",
        "attribute_learner.learner_type",
        "attribute_learner.ontology",
        "attribute_learner.random_seed",
        "attribute_learner.learning_factor",
        "attribute_learner.graph_match_confirmation_threshold",
        "attribute_learner.lexicon_entry_threshold",
        "attribute_learner.smoothing_parameter",
        "relation_learner.learner_type",
        "relation_learner.ontology",
        "relation_learner.random_seed",
        "relation_learner.learning_factor",
        "relation_learner.graph_match_confirmation_threshold",
        "relation_learner.lexicon_entry_threshold",
        "relation_learner.smoothing_parameter",
        "action_learner.learner_type",
        "action_learner.ontology",
        "action_learner.random_seed",
        "action_learner.learning_factor",
        "action_learner.graph_match_confirmation_threshold",
        "action_learner.lexicon_entry_threshold",
        "action_learner.smoothing_parameter",
        "include_functional_learner",
        "include_generics_learner",
    ]
)


@attrs
class ExperimentCurriculum:
    """
    Represents a saved curriculum for some experiment.
    """

    # Validators commented out because the `Phase1InstanceGroup` can't be used as it is a generic type
    train_curriculum: Tuple[Phase1InstanceGroup, ...] = attrib(
        # validator=deep_iterable(member_validator=instance_of(Phase1InstanceGroup)),
        converter=_to_tuple
    )
    test_curriculum: Tuple[Phase1InstanceGroup, ...] = attrib(
        # validator=deep_iterable(member_validator=instance_of(Phase1InstanceGroup)),
        converter=_to_tuple
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
