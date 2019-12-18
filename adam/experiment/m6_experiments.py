from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

from adam.curriculum.m6_curriculum import make_m6_curriculum
from adam.experiment import execute_experiment, Experiment
from adam.experiment.observer import HTMLLogger, HTMLLoggerObserver
from adam.learner.subset import SubsetLanguageLearner
from adam.random_utils import RandomChooser


def main(params: Parameters) -> None:
    output_dir = params.creatable_directory("output_directory")
    experiment_name = params.optional_string("experiment", default="m6_experiments")

    logger = HTMLLogger(
        output_dir=output_dir,
        experiment_name=experiment_name,
        curriculum_name="m6_curriculum",
    )

    m6_curriculum = make_m6_curriculum()
    execute_experiment(
        Experiment(
            name="static-prepositions",
            training_stages=m6_curriculum,
            learner_factory=SubsetLanguageLearner,
            pre_example_training_observers=[
                HTMLLoggerObserver(name="Pre-observer", html_logger=logger)
            ],
            post_example_training_observers=[
                HTMLLoggerObserver(name="Post-observer", html_logger=logger)
            ],
            test_instance_groups=[],
            test_observers=[],
            sequence_chooser=RandomChooser.for_seed(0),
        )
    )


if __name__ == "__main__":
    parameters_only_entry_point(main)
