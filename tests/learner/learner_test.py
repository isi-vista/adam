import logging
from itertools import chain
from typing import Optional

from adam.curriculum.phase1_curriculum import phase1_instances, PHASE1_CHOOSER
from adam.curriculum.pursuit_curriculum import make_simple_pursuit_curriculum
from adam.language_specific.english.english_language_generator import IGNORE_COLORS
from adam.learner import LearningExample
from adam.learner.pursuit import PursuitLanguageLearner
from adam.learner.subset import SubsetLanguageLearner
from adam.ontology import OntologyNode
from adam.ontology.phase1_ontology import BALL, LEARNER, DOG, CHAIR, PERSON, TABLE, BIRD, BOX
from adam.ontology.phase1_ontology import GAILA_PHASE_1_ONTOLOGY
from adam.perception.perception_graph import DumpPartialMatchCallback, DebugCallableType
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    object_variable,
    all_possible,
    color_variable,
)
import random as r
r.seed(0)


def run_subset_learner_for_object(
        obj: OntologyNode, debug_callback: Optional[DebugCallableType] = None
):
    learner_obj = object_variable("learner_0", LEARNER)
    colored_obj_object = object_variable(
        "obj-with-color", obj, added_properties=[color_variable("color")]
    )

    obj_template = Phase1SituationTemplate(
        "colored-obj-object",
        salient_object_variables=[colored_obj_object, learner_obj],
        syntax_hints=[IGNORE_COLORS],
    )

    obj_curriculum = phase1_instances(
        "all obj situations",
        situations=all_possible(
            obj_template, chooser=PHASE1_CHOOSER, ontology=GAILA_PHASE_1_ONTOLOGY
        ),
    )
    test_obj_curriculum = phase1_instances(
        "obj test",
        situations=all_possible(
            obj_template, chooser=PHASE1_CHOOSER, ontology=GAILA_PHASE_1_ONTOLOGY
        ),
    )

    learner = SubsetLanguageLearner(debug_callback)  # type: ignore
    for training_stage in [obj_curriculum]:
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
            descriptions_from_learner = learner.describe(test_instance_perception)
            gold = test_instance_language.as_token_sequence()
            assert [desc.as_token_sequence() for desc in descriptions_from_learner][
                       0
                   ] == gold


def test_subset_learner_ball():
    run_subset_learner_for_object(BALL)


def test_subset_learner_dog():
    debug_callback = DumpPartialMatchCallback(render_path="../renders/")
    # We pass this callback into the learner; it is executed if the learning takes too long, i.e after 60 seconds.
    run_subset_learner_for_object(DOG, debug_callback)


def test_pursuit_learner():
    target_objects = [BALL,
                      # PERSON,
                      # CHAIR,
                      # TABLE,
                      # DOG,
                      BIRD,
                      BOX
                      ]

    # We can use this to generate the actual pursuit curriculum
    # train_curriculum = make_simple_pursuit_curriculum(target_objects=target_objects,
    #                                                   num_instances=15,
    #                                                   num_objects_in_instance=3,
    #                                                   num_noise_instances=0
    #                                                   )

    target_train_templates = []
    target_test_templates = []
    for obj in target_objects:
        # Create train and test templates for the target objects
        train_obj_object = object_variable(
            "obj-with-color", obj
        )
        obj_template = Phase1SituationTemplate(
            "colored-obj-object",
            salient_object_variables=[train_obj_object]
        )
        target_train_templates.extend(chain(*[all_possible(
                obj_template, chooser=PHASE1_CHOOSER, ontology=GAILA_PHASE_1_ONTOLOGY)
                for _ in range(50)]),
        )

        test_obj_object = object_variable(
            "obj-with-color", obj
        )
        test_template = Phase1SituationTemplate(
            "colored-obj-object",
            salient_object_variables=[test_obj_object],
            syntax_hints=[IGNORE_COLORS],
        )
        target_test_templates.extend(all_possible(
            test_template, chooser=PHASE1_CHOOSER, ontology=GAILA_PHASE_1_ONTOLOGY
        ))

    r.shuffle(target_train_templates)
    train_curriculum = phase1_instances(
        "all obj situations",
        situations=target_train_templates,
    )
    test_obj_curriculum = phase1_instances(
        "obj test",
        situations=target_test_templates,
    )

    # All parameters should be in the range 0-1.
    # Learning factor works better when kept < 0.5
    # Graph matching threshold doesn't seem to matter that much, as often seems to be either a
    # complete or a very small match.
    # The lexicon threshold works better between 0.07-0.3, but we need to play around with it because we end up not
    # lexicalize items sufficiently because of diminishing lexicon probability through training
    learner = PursuitLanguageLearner(learning_factor=0.5,
                                     graph_match_confirmation_threshold=0.9,
                                     lexicon_entry_threshold=0.3
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
            descriptions_from_learner = learner.describe(test_instance_perception)
            gold = test_instance_language.as_token_sequence()
            assert [desc.as_token_sequence() for desc in descriptions_from_learner][
                       0
                   ] == gold
