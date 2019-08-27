from adam.experiment import (
    Experiment,
    GeneratedFromExplicitSituationsInstanceGroup,
    TopChoiceExactMatchObserver,
    execute_experiment,
)
from adam.language.language_generator import SingleObjectLanguageGenerator
from adam.learner import MemorizingLanguageLearner
from adam.math_3d import Point
from adam.perception import DummyVisualPerceptionGenerator
from adam.random_utils import RandomChooser
from adam.situation import LocatedObjectSituation, SituationObject
from .testing_lexicon import ENGLISH_TESTING_LEXICON
from .testing_ontology import TRUCK


def test_simple_experiment():
    language_generator = SingleObjectLanguageGenerator(ENGLISH_TESTING_LEXICON)
    perception_generator = DummyVisualPerceptionGenerator()

    only_show_truck = GeneratedFromExplicitSituationsInstanceGroup(
        name="only-truck",
        situations=[
            LocatedObjectSituation([(SituationObject(TRUCK), Point(0.0, 0.0, 0.0))])
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
