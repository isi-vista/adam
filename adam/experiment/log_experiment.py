from typing import Callable

from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

from adam.curriculum.m6_curriculum import make_m6_curriculum
from adam.curriculum.phase1_curriculum import _make_each_object_by_itself_curriculum
from adam.curriculum.pursuit_curriculum import make_simple_pursuit_curriculum
from adam.experiment import execute_experiment, Experiment
from adam.experiment.observer import LearningProgressHtmlLogger
from adam.learner import LanguageLearner
from adam.learner.pursuit import PursuitLanguageLearner
from adam.random_utils import RandomChooser


def main(params: Parameters) -> None:
    output_dir = params.creatable_directory("output_directory")
    experiment_name = params.string("experiment")

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
            learner_factory=learner_factory_from_params(params),
            pre_example_training_observers=[logger.pre_observer()],
            post_example_training_observers=[logger.post_observer()],
            test_instance_groups=test_instance_groups,
            test_observers=[logger.test_observer()],
            sequence_chooser=RandomChooser.for_seed(0),
        )
    )


def learner_factory_from_params(
    params: Parameters
) -> Callable[[], LanguageLearner]:  # type: ignore
    learner_type = params.string("learner", ["pursuit"])
    if learner_type == "pursuit":
        return lambda: PursuitLanguageLearner.from_parameters(params.namespace("pursuit"))
    else:
        raise RuntimeError("can't happen")


def curriculum_from_params(params: Parameters):
    curriculum_name = params.string(
        "curriculum", ["m6-deniz", "each-object-by-itself", "pursuit-with-noise"]
    )
    if curriculum_name == "m6-deniz":
        return (make_m6_curriculum(), [])
    elif curriculum_name == "each-object-by-itself":
        return (
            [_make_each_object_by_itself_curriculum()],
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
            [],
        )
    else:
        raise RuntimeError("Can't happen")


if __name__ == "__main__":
    parameters_only_entry_point(main)
