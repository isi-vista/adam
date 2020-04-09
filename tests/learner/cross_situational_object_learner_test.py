import logging
import random
from itertools import chain
from typing import Optional

from adam.curriculum.phase1_curriculum import PHASE1_CHOOSER_FACTORY, phase1_instances
from adam.curriculum.pursuit_curriculum import make_simple_pursuit_curriculum
from adam.language_specific.english.english_language_generator import IGNORE_COLORS
from adam.learner import LearningExample
from adam.learner.cross_situational import CrossSituationalLanguageLearner
from adam.ontology import OntologyNode
from adam.ontology.phase1_ontology import (
    BALL,
    BIRD,
    BOX,
    DOG,
    GAILA_PHASE_1_ONTOLOGY,
    LEARNER,
)
from adam.perception.perception_graph import DebugCallableType, DumpPartialMatchCallback
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    all_possible,
    color_variable,
    object_variable,
)
from adam.learner.cross_situational import CrossSituationalLanguageLearner


def test_cross_situational_learner():
    target_objects = [
        BALL,
        # PERSON,
        # CHAIR,
        # TABLE,
        # DOG,
        BIRD,
        BOX,
    ]

    target_train_templates = []
    target_test_templates = []
    for obj in target_objects:
        # Create train and test templates for the target objects
        train_obj_object = object_variable("obj-with-color", obj)
        obj_template = Phase1SituationTemplate(
            "colored-obj-object", salient_object_variables=[train_obj_object]
        )
        target_train_templates.extend(
            chain(
                *[
                    all_possible(
                        obj_template,
                        chooser=PHASE1_CHOOSER,
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                    )
                    for _ in range(50)
                ]
            )
        )

        test_obj_object = object_variable("obj-with-color", obj)
        test_template = Phase1SituationTemplate(
            "colored-obj-object",
            salient_object_variables=[test_obj_object],
            syntax_hints=[IGNORE_COLORS],
        )
        target_test_templates.extend(
            all_possible(
                test_template, chooser=PHASE1_CHOOSER, ontology=GAILA_PHASE_1_ONTOLOGY
            )
        )
    rng = random.Random()
    rng.seed(0)
    random.shuffle(target_train_templates, random=rng.random)

    # We can use this to generate the actual pursuit curriculum
    train_curriculum = make_simple_pursuit_curriculum(
        target_objects=target_objects,
        num_instances=30,
        num_objects_in_instance=3,
        num_noise_instances=0,
    )

    test_obj_curriculum = phase1_instances("obj test", situations=target_test_templates)

    # Graph matching threshold doesn't seem to matter that much, as often seems to be either a
    # complete or a very small match.
    # The lexicon threshold works better between 0.07-0.3, but we need to play around with it because we end up not
    # lexicalize items sufficiently because of diminishing lexicon probability through training
    learner = CrossSituationalLanguageLearner(
        graph_match_confirmation_threshold=0.85,
        lexicon_entry_threshold=0.7,
        smoothing_parameter=0.001,
        expected_number_of_meanings=10,
        ontology=GAILA_PHASE_1_ONTOLOGY,
    )  # type: ignore
    for training_stage in [train_curriculum]:
        for (
            _,
            linguistic_description,
            perceptual_representation,
        ) in training_stage.instances():
            learner.observe(
                LearningExample(perceptual_representation, linguistic_description)
            )

    for test_instance_group in [test_obj_curriculum]:
        for (
            _,
            test_instance_language,
            test_instance_perception,
        ) in test_instance_group.instances():
            logging.info("lang: %s", test_instance_language)
            descriptions_from_learner = learner.describe(test_instance_perception)
            gold = test_instance_language.as_token_sequence()
            print(gold)
            print([desc.as_token_sequence() for desc in descriptions_from_learner])
            assert [desc.as_token_sequence() for desc in descriptions_from_learner][
                0
            ] == gold
