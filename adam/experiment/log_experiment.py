import logging
from itertools import repeat
from typing import Callable, Optional

from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

from adam.curriculum.m6_curriculum import (
    M6_PREPOSITION_CURRICULUM_OBJECTS,
    M6_PREPOSITION_SUBCURRICULUM_GENERATORS,
    instantiate_subcurricula,
    make_m6_curriculum,
)
from adam.curriculum.phase1_curriculum import _make_each_object_by_itself_curriculum
from adam.curriculum.pursuit_curriculum import make_simple_pursuit_curriculum
from adam.experiment import Experiment, execute_experiment
from adam.experiment.observer import LearningProgressHtmlLogger
from adam.learner import LanguageLearner
from adam.learner.object_recognizer import ObjectRecognizer
from adam.learner.preposition_subset import PrepositionSubsetLanguageLearner
from adam.learner.pursuit import PursuitLanguageLearner
from adam.perception.perception_graph import GraphLogger
from adam.random_utils import RandomChooser


def main(params: Parameters) -> None:
    output_dir = params.creatable_directory("output_directory")
    experiment_name = params.string("experiment")
    debug_log_dir = params.optional_creatable_directory("debug_log_directory")

    graph_logger: Optional[GraphLogger]
    if debug_log_dir:
        logging.info("Debug graphs will be written to %s", debug_log_dir)
        graph_logger = GraphLogger(debug_log_dir, enable_graph_rendering=True)
    else:
        graph_logger = None

    logger = LearningProgressHtmlLogger.create_logger(
        output_dir=output_dir,
        experiment_name=experiment_name,
        curriculum_name="m6_curriculum",
    )

    (training_instance_groups, test_instance_groups) = curriculum_from_params(params)

    execute_experiment(
        Experiment(
            name=experiment_name,
            training_stages=training_instance_groups,
            learner_factory=learner_factory_from_params(params, graph_logger),
            pre_example_training_observers=[logger.pre_observer()],
            post_example_training_observers=[logger.post_observer()],
            test_instance_groups=test_instance_groups,
            test_observers=[logger.test_observer()],
            sequence_chooser=RandomChooser.for_seed(0),
        )
    )


def learner_factory_from_params(
    params: Parameters, graph_logger: Optional[GraphLogger]
) -> Callable[[], LanguageLearner]:  # type: ignore
    learner_type = params.string("learner", ["pursuit", "preposition-subset"])
    if learner_type == "pursuit":
        return lambda: PursuitLanguageLearner.from_parameters(
            params.namespace("pursuit"), graph_logger=graph_logger
        )
    elif learner_type == "preposition-subset":
        return lambda: PrepositionSubsetLanguageLearner(
            graph_logger=GraphLogger(
                log_directory=params.creatable_directory("log_directory"),
                enable_graph_rendering=True,
                serialize_graphs=True,
            ),
            # Eval hack! This is specific to the M6 ontology
            object_recognizer=ObjectRecognizer.for_ontology_types(
                M6_PREPOSITION_CURRICULUM_OBJECTS
            ),
        )
    else:
        raise RuntimeError("can't happen")


def curriculum_from_params(params: Parameters):
    curriculum_name = params.string(
        "curriculum",
        ["m6-deniz", "each-object-by-itself", "pursuit-with-noise", "m6-preposition"],
    )
    if curriculum_name == "m6-deniz":
        return (make_m6_curriculum(), [])
    elif curriculum_name == "each-object-by-itself":
        return (
            # We show the learned each item 6 times,
            # because pursuit won't lexicalize anything it hasn't seen five times.
            list(repeat(_make_each_object_by_itself_curriculum(), 6)),
            [_make_each_object_by_itself_curriculum()],
        )
    elif curriculum_name == "pursuit-with-noise":
        pursuit_curriculum_params = params.namespace("pursuit-curriculum-params")
        num_instances = pursuit_curriculum_params.integer("num_instances")
        num_noise_instances = pursuit_curriculum_params.integer("num_noise_instances")
        num_objects_in_instance = pursuit_curriculum_params.integer(
            "num_objects_in_instance"
        )
        return (
            [
                make_simple_pursuit_curriculum(
                    num_instances=num_instances,
                    num_objects_in_instance=num_objects_in_instance,
                    num_noise_instances=num_noise_instances,
                )
            ],
            [_make_each_object_by_itself_curriculum()],
        )
    elif curriculum_name == "m6-preposition":
        return (instantiate_subcurricula(M6_PREPOSITION_SUBCURRICULUM_GENERATORS), [])
    else:
        raise RuntimeError("Can't happen")


if __name__ == "__main__":
    parameters_only_entry_point(main)
