from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

from adam.curriculum.preposition_curriculum import (
    make_prepositions_curriculum_training,
)
from adam.curriculum_to_html import STR_TO_CURRICULUM
from adam.experiment import execute_experiment, Experiment
from adam.experiment.observer import HTMLLogger, HTMLLoggerObserver
from adam.learner.preposition_subset import PrepositionSubsetLanguageLearner
from adam.random_utils import RandomChooser


def main(params: Parameters) -> None:
    output_dir = params.creatable_directory("output_directory")
    experiment_name = params.optional_string(
        "experiment", default="m6_experiments"
    )

    logger = HTMLLogger(output_dir=output_dir,
                        experiment_name=experiment_name,
                        curriculum_name='prepositions_curriculum')

    preobserver = HTMLLoggerObserver(name='preobserver', html_logger=logger)

    execute_experiment(
        Experiment(
            name="static-prepositions",
            training_stages=make_prepositions_curriculum_training(),
            learner_factory=PrepositionSubsetLanguageLearner,
            pre_example_training_observers=[preobserver],
            post_example_training_observers=[],
            test_instance_groups=[],
            test_observers=[],
            sequence_chooser=RandomChooser.for_seed(0),
        )
    )



if __name__ == "__main__":
    parameters_only_entry_point(main)
