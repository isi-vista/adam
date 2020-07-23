from random import Random, shuffle

from adam.curriculum.curriculum_utils import PHASE1_CHOOSER_FACTORY
from adam.curriculum.phase1_curriculum import _make_plural_objects_curriculum
from adam.language_specific.english.english_language_generator import (
    GAILA_PHASE_1_LANGUAGE_GENERATOR,
)
from adam.learner import LanguageMode, LearningExample
from adam.learner.integrated_learner import IntegratedTemplateLearner
from adam.learner.quantifers import ToleranceRuleQuantifierTemplateLearner
from adam.ontology.phase1_ontology import BALL, GAILA_PHASE_1_ONTOLOGY
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    GAILA_PHASE_1_PERCEPTION_GENERATOR,
)
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    all_possible,
    object_variable,
)
from learner import LANGUAGE_MODE_TO_TEMPLATE_LEARNER_OBJECT_RECOGNIZER


def test_english_quantifier_learning():
    plural_curriculum = _make_plural_objects_curriculum(GAILA_PHASE_1_LANGUAGE_GENERATOR)
    instances = list(plural_curriculum.instances())
    shuffle(instances, Random(0).random)

    learner = IntegratedTemplateLearner(
        object_learner=LANGUAGE_MODE_TO_TEMPLATE_LEARNER_OBJECT_RECOGNIZER[
            LanguageMode.ENGLISH
        ],
        number_learner=ToleranceRuleQuantifierTemplateLearner(
            language_mode=LanguageMode.ENGLISH, min_types_to_lexicalize=4
        ),
        language_mode=LanguageMode.ENGLISH,
    )

    for (_, language, perception) in instances:
        learner.observe(
            LearningExample(perception=perception, linguistic_description=language)
        )

    ball_1 = object_variable("ball_1", BALL)
    ball_2 = object_variable("ball_2", BALL)
    ball_3 = object_variable("ball_3", BALL)
    ball_4 = object_variable("ball_4", BALL)

    one_ball_situation = Phase1SituationTemplate(
        "one ball", salient_object_variables=[ball_1]
    )
    two_balls_situation = Phase1SituationTemplate(
        "two balls", salient_object_variables=[ball_1, ball_2]
    )
    three_balls_situation = Phase1SituationTemplate(
        "three balls", salient_object_variables=[ball_1, ball_2, ball_3]
    )
    four_balls_situation = Phase1SituationTemplate(
        "four balls", salient_object_variables=[ball_1, ball_2, ball_3, ball_4]
    )
    situation_to_references = {
        one_ball_situation: [("a", "ball")],
        two_balls_situation: [("two", "ball", "s"), ("ball", "s")],
        three_balls_situation: [("many", "ball", "s"), ("ball", "s")],
        four_balls_situation: [("many", "ball", "s"), ("ball", "s")],
    }

    for (test_situation_template, references) in situation_to_references.items():
        for test_situation in all_possible(
            test_situation_template,
            chooser=PHASE1_CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            block_multiple_objects_of_the_same_type=False,
        ):
            description = tuple(
                x.as_token_sequence()
                for x in learner.describe(
                    GAILA_PHASE_1_PERCEPTION_GENERATOR.generate_perception(
                        test_situation, chooser=PHASE1_CHOOSER_FACTORY()
                    )
                )
            )
            for reference in references:
                assert reference in description
