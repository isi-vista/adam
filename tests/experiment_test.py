from adam.curriculum import GeneratedFromSituationsInstanceGroup
from adam.experiment import Experiment, execute_experiment
from adam.experiment.observer import (
    TopChoiceExactMatchObserver,
    CandidateAccuracyObserver,
)
from adam.language.language_generator import SingleObjectLanguageGenerator
from adam.language_specific.english.english_phase_1_lexicon import (
    GAILA_PHASE_1_ENGLISH_LEXICON,
)
from adam.learner import MemorizingLanguageLearner
from adam.math_3d import Point
from adam.ontology.phase1_ontology import BALL, GAILA_PHASE_1_ONTOLOGY
from adam.perception import DummyVisualPerceptionGenerator
from adam.random_utils import RandomChooser
from adam.situation import LocatedObjectSituation, SituationObject


def test_simple_experiment():
    language_generator = SingleObjectLanguageGenerator(GAILA_PHASE_1_ENGLISH_LEXICON)
    perception_generator = DummyVisualPerceptionGenerator()

    only_show_truck = GeneratedFromSituationsInstanceGroup(
        name="only-ball",
        situations=[
            LocatedObjectSituation(
                [
                    (
                        SituationObject.instantiate_ontology_node(
                            BALL, ontology=GAILA_PHASE_1_ONTOLOGY
                        ),
                        Point(0.0, 0.0, 0.0),
                    )
                ]
            )
        ],
        language_generator=language_generator,
        perception_generator=perception_generator,
        chooser=RandomChooser.for_seed(0),
    )

    pre_acc = CandidateAccuracyObserver("pre")
    post_acc = CandidateAccuracyObserver("post")
    test_acc = CandidateAccuracyObserver("test")

    experiment = Experiment(
        name="simple",
        training_stages=[only_show_truck],
        learner_factory=MemorizingLanguageLearner,
        pre_example_training_observers=[TopChoiceExactMatchObserver("pre"), pre_acc],
        post_example_training_observers=[TopChoiceExactMatchObserver("post"), post_acc],
        warm_up_test_instance_groups=[only_show_truck],
        test_instance_groups=[only_show_truck],
        test_observers=[TopChoiceExactMatchObserver("test"), test_acc],
        sequence_chooser=RandomChooser.for_seed(0),
    )

    execute_experiment(experiment)

    assert pre_acc.accuracy() == 0.0
    assert post_acc.accuracy() == 1.0
    assert test_acc.accuracy() == 1.0
