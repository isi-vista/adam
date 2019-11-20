from typing import Mapping

from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

from adam.curriculum.preposition_curriculum import (
    make_prepositions_curriculum,
    make_prepositions_curriculum_training,
)
from adam.experiment import execute_experiment, Experiment
from adam.language import LinguisticDescription
from adam.learner import LanguageLearner, LearningExample
from adam.perception import PerceptualRepresentation, PerceptionT
from adam.random_utils import RandomChooser


def main(params: Parameters) -> None:
    execute_experiment(
        Experiment(
            name="static-prepositions",
            training_stages=make_prepositions_curriculum_training(),
            learner_factory=StaticPrepositionLearner,
            pre_example_training_observers=[],
            post_example_training_observers=[],
            test_instance_groups=[],
            test_observers=[],
            sequence_chooser=RandomChooser.for_seed(0),
        )
    )


class StaticPrepositionLearner(LanguageLearner):
    def observe(
        self, learning_example: LearningExample[PerceptionT, LinguisticDescription]
    ) -> None:
        pass

    def describe(
        self, perception: PerceptualRepresentation[PerceptionT]
    ) -> Mapping[LinguisticDescription, float]:
        return {}


if __name__ == "__main__":
    parameters_only_entry_point(main)
