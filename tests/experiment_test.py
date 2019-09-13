from adam.experiment import Experiment, execute_experiment
from adam.experiment.observer import TopChoiceExactMatchObserver
from adam.curriculum import GeneratedFromSituationsInstanceGroup
from adam.language.language_generator import SingleObjectLanguageGenerator
from adam.language_specific.english.english_phase_1_lexicon import (
    GAILA_PHASE_1_ENGLISH_LEXICON,
)
from adam.learner import MemorizingLanguageLearner
from adam.math_3d import Point
from adam.ontology.phase1_ontology import BALL
from adam.perception import DummyVisualPerceptionGenerator
from adam.random_utils import RandomChooser
from adam.situation import LocatedObjectSituation, SituationObject


def test_simple_experiment():
    language_generator = SingleObjectLanguageGenerator(GAILA_PHASE_1_ENGLISH_LEXICON)
    perception_generator = DummyVisualPerceptionGenerator()

    only_show_truck = GeneratedFromSituationsInstanceGroup(
        name="only-ball",
        situations=[
            LocatedObjectSituation([(SituationObject(BALL), Point(0.0, 0.0, 0.0))])
        ],
        language_generator=language_generator,
        perception_generator=perception_generator,
        chooser=RandomChooser.for_seed(0),
    )

    experiment = Experiment(
        name="simple",
        training_stages=[only_show_truck],
        learner_factory=MemorizingLanguageLearner,
        pre_example_training_observers=[TopChoiceExactMatchObserver("pre")],
        post_example_training_observers=[TopChoiceExactMatchObserver("post")],
        warm_up_test_instance_groups=[only_show_truck],
        test_instance_groups=[only_show_truck],
        test_observers=[TopChoiceExactMatchObserver("test")],
        sequence_chooser=RandomChooser.for_seed(0),
    )

    execute_experiment(experiment)
