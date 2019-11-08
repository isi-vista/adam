from adam.curriculum.phase1_curriculum import phase1_instances, PHASE1_CHOOSER
from adam.language_specific.english.english_language_generator import IGNORE_COLORS
from adam.learner import LearningExample
from adam.learner.subset import SubsetLanguageLearner
from adam.ontology.phase1_ontology import BALL, LEARNER
from adam.ontology.phase1_ontology import GAILA_PHASE_1_ONTOLOGY
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    object_variable,
    all_possible,
    color_variable,
)


def test_subset_learner_ball():
    learner = object_variable("learner_0", LEARNER)
    colored_ball_object = object_variable(
        "ball-with-color", BALL, added_properties=[color_variable("color")]
    )

    ball_template = Phase1SituationTemplate(
        "colored-ball-object",
        salient_object_variables=[colored_ball_object, learner],
        syntax_hints=[IGNORE_COLORS],
    )

    ball_curriculum = phase1_instances(
        "all ball situations",
        situations=all_possible(
            ball_template, chooser=PHASE1_CHOOSER, ontology=GAILA_PHASE_1_ONTOLOGY
        ),
    )
    test_ball_curriculum = phase1_instances(
        "ball test",
        situations=all_possible(
            ball_template, chooser=PHASE1_CHOOSER, ontology=GAILA_PHASE_1_ONTOLOGY
        ),
    )

    learner = SubsetLanguageLearner()
    for training_stage in [ball_curriculum]:
        for (
            _,
            linguistic_description,
            perceptual_representation,
        ) in training_stage.instances():
            learner.observe(
                LearningExample(perceptual_representation, linguistic_description)
            )

    for test_instance_group in [test_ball_curriculum]:
        for (
            _,
            test_instance_language,
            test_instance_perception,
        ) in test_instance_group.instances():
            descriptions_from_learner = learner.describe(test_instance_perception)
            gold = test_instance_language.as_token_sequence()
            assert [desc.as_token_sequence() for desc in descriptions_from_learner][
                0
            ] == gold
