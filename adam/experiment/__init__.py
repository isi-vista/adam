"""
Allows managing experimental configurations in code.
"""
import logging
import pickle
import os
from itertools import chain

# for some reason, pylint doesn't recognize the types used in quoted type annotations
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Sequence,
    Tuple,
    Optional,
)  # pylint:disable=unused-import

from attr import attrib, attrs
from attr.validators import instance_of

# noinspection PyProtectedMember
from immutablecollections.converter_utils import _to_tuple
from vistautils.preconditions import check_arg

from adam.curriculum import InstanceGroup
from adam.experiment.observer import DescriptionObserver
from adam.language import LinguisticDescriptionT
from adam.situation import SituationT
from adam.learner import TopLevelLanguageLearner, LearningExample
from adam.perception import PerceptionT
from adam.random_utils import SequenceChooser


@attrs(frozen=True)
class Experiment(Generic[SituationT, LinguisticDescriptionT, PerceptionT]):
    r"""
    A particular experimental configuration.

    An experiment specifies what data is fed to the `LanguageLearner`, what `LanguageLearner` is
    used and how it is configured, how the trained `LanguageLearner` is tested, and what
    analysis is done on the results.

    At various stages in the experiment,
    `DescriptionObserver`\ s will be able to examine
    the descriptions of situations produced by the `LanguageLearner`.
    Based on their observations, they can provide reports at the end of the experiment.

    Note that `Experiment` objects should not be reused because components
    (especially the observers) may maintain state.
    """

    name: str = attrib(validator=instance_of(str))
    """A human-readable description of this experiment."""
    training_stages: "Sequence[InstanceGroup[SituationT, LinguisticDescriptionT, PerceptionT]]" = attrib(
        converter=_to_tuple, kw_only=True
    )
    """
    Every experiment has one or more *training_stages*;
    simple experiments will have only one,
    while those which reflect a curriculum
    which increases in complexity over time may have several.
    Each of these is an `InstanceGroup`
    providing triples of a `Situation`
    with the corresponding `LinguisticDescription` and `PerceptualRepresentation`.
    There are many ways an `InstanceGroup` could be specified,
    ranging from simply a collection of these triples (e.g. `ExplicitWithSituationInstanceGroup`)
    to a complex rule-governed process (e.g. `GeneratedFromSituationsInstanceGroup`).
    """
    learner_factory: Callable[
        [], TopLevelLanguageLearner[PerceptionT, LinguisticDescriptionT]
    ] = attrib(kw_only=True)
    """ A no-argument function which will return the `LanguageLearner` which should be trained."""
    sequence_chooser: SequenceChooser = attrib(
        validator=instance_of(SequenceChooser), kw_only=True
    )
    """
    Used for making all "random" decisions by all components to ensure reproducibility.
    """
    pre_example_training_observers: "Tuple[DescriptionObserver[SituationT, LinguisticDescriptionT, PerceptionT]]" = attrib(  # type: ignore
        converter=_to_tuple, default=tuple(), kw_only=True
    )
    r"""
    These `DescriptionObserver`\ s are provided
    with the description a `LanguageLearner` would give to a situation during training
    before it is shown the correct description.
    """
    post_example_training_observers: "Tuple[DescriptionObserver[SituationT, LinguisticDescriptionT, PerceptionT]]" = attrib(  # type: ignore
        converter=_to_tuple, default=tuple(), kw_only=True
    )
    """
    Same as *pre_example_training_observers*
    except they receive the learner's description
    after it has updated itself by seeing the correct description.
    """
    warm_up_test_instance_groups: "Sequence[InstanceGroup[Any, LinguisticDescriptionT, PerceptionT]]" = attrib(
        converter=_to_tuple, default=tuple(), kw_only=True
    )
    """
    May optionally be provided
    if at test time the `LanguageLearner` needs to be shown some observations before evaluation
    (for example, to introduce some new objects).
    """
    test_instance_groups: "Sequence[InstanceGroup[Any, LinguisticDescriptionT, PerceptionT]]" = attrib(
        converter=_to_tuple, default=tuple(), kw_only=True
    )
    r"""
    The situations and perceptual representations
    the trained `LanguageLearner` will be asked to describe for evaluation.
    These are specified by `InstanceGroup`\ s, just like the training data.
    """
    test_observers: "Sequence[DescriptionObserver[SituationT, LinguisticDescriptionT, PerceptionT]]" = attrib(
        converter=_to_tuple, default=tuple(), kw_only=True
    )
    r"""
    These `DescriptionObserver`\ s observe the descriptions
    given by the learner on the test instances.
    These are what should be used for computing evaluation metrics.
    """

    def __attrs_post_init__(self) -> None:
        check_arg(callable(self.learner_factory), "Learner factory must be callable")


def _learner_states_by_most_recent(learner_path: Path) -> Iterable[Tuple[int, Path]]:
    paths = []
    for logged_state_path in learner_path.glob("learner_state_at_*.pkl"):
        if not logged_state_path.is_file():
            logging.warning("Skipping non-file learner state %s.", str(logged_state_path))
            continue
        iteration_number_string = logged_state_path.name.replace("learner_state_at_", "").replace(".pkl", "")
        try:
            iteration_number = int(iteration_number_string)
        except ValueError:
            logging.warning("Skipping learner state file with bad iteration number %s.", iteration_number_string)
            continue
        paths.append((iteration_number, logged_state_path))
        logged_state_path.stat()
    return sorted(paths, reverse=True)


def execute_experiment(
    experiment: Experiment[SituationT, LinguisticDescriptionT, PerceptionT],
    *,
    log_path: Optional[Path] = None,
    log_hypotheses_every_n_examples: int = 250,
    learner_logging_path: Optional[Path] = None,
    log_learner_state: bool = True,
    load_learner_state: Optional[Path] = None,
    resume_from_latest_logged_state: bool = False,
    starting_point: int = 0,
    point_to_log: int = 0,
) -> None:
    """
    Runs an `Experiment`.
    """

    # the starting point must be greater than or equal to 0
    if starting_point < 0:
        logging.warning(f"Starting point {starting_point} is invalid, setting to 0")
        starting_point = 0

    if resume_from_latest_logged_state and load_learner_state is not None:
        raise RuntimeError(
            "Cannot both resume from latest logged state and load a specified state file."
        )

    if resume_from_latest_logged_state and learner_logging_path is None:
        raise RuntimeError(
            "Need state logging path to be able to resume from latest learner state."
        )

    if resume_from_latest_logged_state and starting_point != -1:
        raise RuntimeError(
            "Starting point should not be specified when"
        )

    # make the directories in which to log the learner
    if log_learner_state and learner_logging_path:
        learner_path = learner_logging_path / "learner_state"
        # if the directory in which we wish to log the learner doesn't exist, we must create it
        if not os.path.exists(learner_path):
            try:
                os.mkdir(learner_path)
            # if we don't have a directory where we can log our learner state, we simply don't log it
            except OSError:
                logging.warning("Cannot log learner state to %s", str(learner_path))
                log_learner_state = False
                logging.warning("Proceeding without logging learner state")

    logging.info("Beginning experiment %s", experiment.name)

    # if there is an existing learner to load, try to load it
    if load_learner_state:
        if starting_point == 0:
            logging.warning("Using existing learner, expected starting point > 0")
        logging.info("Loading existing learner from %s", str(load_learner_state))
        try:
            learner = pickle.load(open(load_learner_state, "rb"))
        # if the learner can't be loaded, just instantiate the default learner and notify the user
        except OSError:
            learner = experiment.learner_factory()
            logging.warning(
                "Unable to load learner at %s, using factory instead", load_learner_state
            )

    # If we should resume from the latest logged state, try to load each numbered state file,
    # starting with the file for the learner that got the farthest in the curriculum
    elif resume_from_latest_logged_state:
        logging.info("Attempting to result from latest learner state...")
        learner = None
        # This should already be set, but just in case it wasn't, for example if
        # log_learner_state = False, set it.
        learner_path = learner_logging_path / "learner_state"
        # Note that this is safe if the learner path doesn't exist -- this loop will simply never
        # execute.
        for iteration_number, learner_state_path in _learner_states_by_most_recent(learner_path):
            try:
                with learner_state_path.open("rb") as f:
                    learner = pickle.load(f)
            except OSError:
                logging.warning("Unable to open learner state at %s; skipping.", str(learner_state_path))
            except pickle.UnpicklingError:
                logging.warning("Couldn't unpickle learner state at %s; skipping.", str(learner_state_path))
        # Fall back on default learner
        if learner is None:
            logging.info("Could not load a saved learner; using factory instead.")
            learner = experiment.learner_factory()

    # if there's no existing learner, instantiate the default
    else:
        learner = experiment.learner_factory()
    logging.info("Instantiated learner %s", learner)

    num_observations = 0

    for training_stage in experiment.training_stages:
        if num_observations > starting_point:
            logging.info("Beginning training stage %s", training_stage.name())
        for (
            situation,
            linguistic_description,
            perceptual_representation,
        ) in training_stage.instances():
            num_observations += 1

            # don't learn from anything until we've reached the starting of the the learning
            if num_observations < starting_point:
                continue

            # log the start of the learning
            if num_observations == starting_point:
                logging.info("Beginning training stage %s", training_stage.name())

            # if we've reached the user-given point where we want to log the learner, log it here
            if (
                point_to_log > 0
                # we log after the nth input is given to the learner
                and num_observations - 1 == point_to_log
                and log_learner_state
            ):
                logging.info(f"Reached {point_to_log} instances, logging learner")
                # dump the learner to a pickle file
                pickle.dump(
                    learner,
                    open(
                        learner_path / f"learner_state_at_{str(point_to_log)}.pkl", "wb"
                    ),
                    pickle.HIGHEST_PROTOCOL,
                )

            # if we've reached the next num_observations where we should log hypotheses, log the hypotheses
            if log_path and num_observations % log_hypotheses_every_n_examples == 0:
                learner.log_hypotheses(log_path / str(num_observations))
                # if we are logging the learner state, we do it here
                if log_learner_state:
                    # dump the learner to a pickle file
                    pickle.dump(
                        learner,
                        open(
                            learner_path
                            / f"learner_state_at_{str(num_observations)}.pkl",
                            "wb",
                        ),
                        pickle.HIGHEST_PROTOCOL,
                    )

            if experiment.pre_example_training_observers:
                learner_descriptions_before_seeing_example = learner.describe(
                    perceptual_representation
                )
                if situation:
                    for pre_example_observer in experiment.pre_example_training_observers:
                        pre_example_observer.observe(
                            situation,
                            linguistic_description,
                            perceptual_representation,
                            learner_descriptions_before_seeing_example,
                        )
                        pre_example_observer.report()
                else:
                    raise ValueError(
                        "Observed training instances cannot lack a situation"
                    )

            learner.observe(
                LearningExample(perceptual_representation, linguistic_description),
                offset=starting_point,
            )

            if experiment.post_example_training_observers:
                learner_descriptions_after_seeing_example = learner.describe(
                    perceptual_representation
                )
                for post_example_observer in experiment.post_example_training_observers:
                    post_example_observer.observe(
                        situation,
                        linguistic_description,
                        perceptual_representation,
                        learner_descriptions_after_seeing_example,
                        offset=starting_point,
                    )
    logging.info("Training complete")

    for training_observer in chain(
        experiment.pre_example_training_observers,
        experiment.post_example_training_observers,
    ):
        training_observer.report()

    if log_path:
        learner.log_hypotheses(log_path / "final")
        # log the final learner if the user wishes for it to be logged
        if log_learner_state:
            pickle.dump(
                learner,
                open(learner_path / f"final_learner_state.pkl", "wb"),
                pickle.HIGHEST_PROTOCOL,
            )

    logging.info("Warming up for tests")
    for warm_up_instance_group in experiment.warm_up_test_instance_groups:
        for (
            situation,
            warm_up_test_instance_language,
            warm_up_test_instance_perception,
        ) in warm_up_instance_group.instances():
            learner.observe(
                LearningExample(
                    warm_up_test_instance_perception, warm_up_test_instance_language
                )
            )

    logging.info("Performing tests")
    num_test_observations = 0
    for test_instance_group in experiment.test_instance_groups:
        for (
            situation,
            test_instance_language,
            test_instance_perception,
        ) in test_instance_group.instances():
            logging.info(f"Test Description: {num_test_observations}")
            num_test_observations += 1
            descriptions_from_learner = learner.describe(test_instance_perception)
            for test_observer in experiment.test_observers:
                test_observer.observe(
                    situation,
                    test_instance_language,
                    test_instance_perception,
                    descriptions_from_learner,
                )

    for test_observer in experiment.test_observers:
        test_observer.report()

    logging.info("Experiment %s complete", experiment.name)
