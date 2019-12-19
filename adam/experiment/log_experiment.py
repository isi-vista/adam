from typing import Callable

from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

from adam.curriculum.m6_curriculum import make_m6_curriculum
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

    execute_experiment(
        Experiment(
            name=experiment_name,
            training_stages=curriculum_from_params(params),
            learner_factory=learner_factory_from_params(params),
            pre_example_training_observers=[logger.pre_observer()],
            post_example_training_observers=[logger.post_observer()],
            test_instance_groups=[],
            test_observers=[],
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
    curriculum_name = params.string("curriculum", ["m6-deniz"])
    if curriculum_name == "m6-deniz":
        return make_m6_curriculum()
    else:
        raise RuntimeError("Can't happen")


if __name__ == "__main__":
    parameters_only_entry_point(main)
