from adam.curriculum.phase1_curriculum import _phase1_instances, _CHOOSER, _LEARNER_OBJECT
from adam.experiment import Experiment, execute_experiment
from adam.experiment.observer import TopChoiceExactMatchObserver
from adam.learner import SubsetLanguageLearner
from adam.ontology.phase1_ontology import BALL
from adam.ontology.phase1_ontology import GAILA_PHASE_1_ONTOLOGY
from adam.random_utils import RandomChooser
from adam.situation.templates.phase1_templates import Phase1SituationTemplate, object_variable, all_possible, \
    color_variable
import logging


def test_subset_learner_ball():

    COLOR = color_variable("color")
    COLORED_BALL_OBJECT = object_variable("ball-with-color", BALL, added_properties=[COLOR])
    ball_template = Phase1SituationTemplate(
        "colored-ball-object", salient_object_variables=[COLORED_BALL_OBJECT, _LEARNER_OBJECT]
    )

    ball_curriculum = _phase1_instances(
        "all ball situations",
        situations=all_possible(
            ball_template, chooser=_CHOOSER, ontology=GAILA_PHASE_1_ONTOLOGY
        )
    )
    test_ball_curriculum = _phase1_instances(
        "all ball situations",
        situations=all_possible(
            ball_template, chooser=_CHOOSER, ontology=GAILA_PHASE_1_ONTOLOGY
        )
    )

    experiment = Experiment(
        name="simple",
        training_stages=[ball_curriculum],
        learner_factory=SubsetLanguageLearner,
        pre_example_training_observers=[TopChoiceExactMatchObserver("pre")],
        post_example_training_observers=[TopChoiceExactMatchObserver("post")],
        # warm_up_test_instance_groups=[ball_curriculum],
        test_instance_groups=[test_ball_curriculum],
        test_observers=[TopChoiceExactMatchObserver("test")],
        sequence_chooser=RandomChooser.for_seed(0),
    )

    execute_experiment(experiment)


# For debugging purposes to view the log of experiment
logging.basicConfig(level=logging.DEBUG)
test_subset_learner_ball()
