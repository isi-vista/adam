import logging
import pickle
from pathlib import Path
from typing import Optional, cast

# TODO I can probably delete a lot of this stuff as legacy
from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

from adam.experiment import Experiment, execute_experiment
from adam.experiment.curriculum_repository import (
    read_experiment_curriculum,
)
from adam.experiment.log_experiment import (
    learner_factory_from_params,
    curriculum_from_params,
)
from adam.experiment.experiment_utils import observer_states_by_most_recent
from adam.experiment.observer import LearningProgressHtmlLogger, YAMLLogger
from adam.learner.language_mode import LanguageMode
from adam.learner.pursuit import HypothesisLogger
from adam.perception.perception_graph import GraphLogger
from adam.random_utils import RandomChooser

SYMBOLIC = "symbolic"
SIMULATED = "simulated"


def contrastive_learning_entry_point(params: Parameters) -> None:
    experiment_name = params.string("experiment")
    debug_log_dir = params.optional_creatable_directory("debug_log_directory")

    graph_logger: Optional[HypothesisLogger]
    if debug_log_dir:
        logging.info("Debug graphs will be written to %s", debug_log_dir)
        graph_logger = HypothesisLogger(debug_log_dir, enable_graph_rendering=True)
    else:
        graph_logger = None

    debug_perception_log_dir = params.optional_creatable_directory(
        "debug_perception_log_dir"
    )
    perception_graph_logger: Optional[GraphLogger]
    if debug_perception_log_dir:
        logging.info(
            "Debug perception graphs will be written to %s", debug_perception_log_dir
        )
        perception_graph_logger = GraphLogger(
            debug_perception_log_dir, enable_graph_rendering=True
        )
    else:
        perception_graph_logger = None

    language_mode = params.enum(
        "language_mode", LanguageMode, default=LanguageMode.ENGLISH
    )
    experiment_group_dir = params.optional_creatable_directory("experiment_group_dir")
    resume_from_last_logged_state = params.boolean(
        "resume_from_latest_logged_state", default=False
    )

    test_observer = []  # type: ignore
    pre_observer = []  # type: ignore
    post_observer = []  # type: ignore
    experiment_type = params.string(
        "experiment_type", valid_options=[SYMBOLIC, SIMULATED], default=SYMBOLIC
    )
    if experiment_type == SYMBOLIC:
        logging.info("Symbolic Experiment Section")
        logger = LearningProgressHtmlLogger.create_logger(params)

        curriculum_repository_path = params.optional_existing_directory(
            "load_from_curriculum_repository"
        )

        if curriculum_repository_path:
            curriculum = read_experiment_curriculum(
                curriculum_repository_path, params, language_mode
            )
            (training_instance_groups, test_instance_groups) = (
                curriculum.train_curriculum,
                curriculum.test_curriculum,
            )
        else:
            (training_instance_groups, test_instance_groups) = curriculum_from_params(
                params, language_mode
            )

        # Check if we have explicit observer states to load
        observers_state = params.optional_existing_file("observers_state_path")

        if resume_from_last_logged_state and observers_state:
            raise RuntimeError(
                "Can not resume from last logged state and provide explicit observer state paths"
            )

        if resume_from_last_logged_state:
            if not experiment_group_dir:
                raise RuntimeError(
                    "experiment_group_dir must be specified when resume_from_last_logged_state is true."
                )

            # Try to Load Observers
            for _, observers_state_path in observer_states_by_most_recent(
                cast(Path, experiment_group_dir) / "observer_state", "observers_state_at_"
            ):
                try:
                    with observers_state_path.open("rb") as f:
                        observers_holder = pickle.load(f)
                        pre_observer = observers_holder.pre_observers
                        post_observer = observers_holder.post_observers
                        test_observer = observers_holder.test_observers
                except OSError:
                    logging.warning(
                        "Unable to open observer state at %s; skipping.",
                        str(observers_state_path),
                    )
                except pickle.UnpicklingError:
                    logging.warning(
                        "Couldn't unpickle observer state at %s; skipping.",
                        str(observers_state_path),
                    )

            if not pre_observer and not post_observer and not test_observer:
                logging.warning("Reverting to default observers.")
                pre_observer = [
                    logger.pre_observer(  # type: ignore
                        params=params.namespace_or_empty("pre_observer"),
                        experiment_group_dir=experiment_group_dir,
                    )
                ]

                post_observer = [
                    logger.post_observer(  # type: ignore
                        params=params.namespace_or_empty("post_observer"),
                        experiment_group_dir=experiment_group_dir,
                    )
                ]

                test_observer = [
                    logger.test_observer(  # type: ignore
                        params=params.namespace_or_empty("post_observer"),
                        experiment_group_dir=experiment_group_dir,
                    )
                ]

        elif observers_state:
            try:
                with observers_state.open("rb") as f:
                    observers_holder = pickle.load(f)
                    pre_observer = observers_holder.pre_observers
                    post_observer = observers_holder.post_observers
                    test_observer = observers_holder.test_observers
            except OSError:
                logging.warning(
                    "Unable to open observer state at %s; skipping.", str(observers_state)
                )
            except pickle.UnpicklingError:
                logging.warning(
                    "Couldn't unpickle observer state at %s; skipping.",
                    str(observers_state),
                )
        else:
            pre_observer = [
                logger.pre_observer(  # type: ignore
                    params=params.namespace_or_empty("pre_observer"),
                    experiment_group_dir=experiment_group_dir,
                )
            ]

            post_observer = [
                logger.post_observer(  # type: ignore
                    params=params.namespace_or_empty("post_observer"),
                    experiment_group_dir=experiment_group_dir,
                )
            ]

            test_observer = [
                logger.test_observer(  # type: ignore
                    params=params.namespace_or_empty("test_observer"),
                    experiment_group_dir=experiment_group_dir,
                )
            ]
    else:
        logging.info("Simulated Experiment Section")
        (training_instance_groups, test_instance_groups) = curriculum_from_params(
            params, language_mode
        )
        yaml_observer_pre = YAMLLogger.from_params(  # type: ignore
            "pre_observer", params.namespace_or_empty("pre_observer")
        )
        if yaml_observer_pre:
            pre_observer.append(yaml_observer_pre)

        yaml_observer_post = YAMLLogger.from_params(  # type: ignore
            "post_observer", params.namespace_or_empty("post_observer")
        )
        if yaml_observer_post:
            post_observer.append(yaml_observer_post)

        yaml_observer_test = YAMLLogger.from_params(  # type: ignore
            "test_observer", params.namespace_or_empty("test_observer")
        )
        if yaml_observer_test:
            test_observer.append(yaml_observer_test)

    execute_experiment(
        Experiment(
            name=experiment_name,
            training_stages=training_instance_groups,
            learner_factory=learner_factory_from_params(
                params, graph_logger, language_mode
            ),
            pre_example_training_observers=pre_observer,
            post_example_training_observers=post_observer,
            test_instance_groups=test_instance_groups,
            test_observers=test_observer,
            sequence_chooser=RandomChooser.for_seed(
                params.integer("sequence_chooser_seed", default=0)
            ),
        ),
        log_path=params.optional_creatable_directory("hypothesis_log_dir"),
        log_hypotheses_every_n_examples=params.integer(
            "log_hypothesis_every_n_steps", default=250
        ),
        log_learner_state=params.boolean("log_learner_state", default=True),
        learner_logging_path=experiment_group_dir,
        starting_point=params.integer("starting_point", default=0),
        point_to_log=params.integer("point_to_log", default=0),
        load_learner_state=params.optional_existing_file("learner_state_path"),
        resume_from_latest_logged_state=resume_from_last_logged_state,
        debug_learner_pickling=params.boolean("debug_learner_pickling", default=False),
        perception_graph_logger=perception_graph_logger,
    )


if __name__ == "__main__":
    parameters_only_entry_point(contrastive_learning_entry_point)
