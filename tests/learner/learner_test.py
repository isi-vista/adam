from typing import Optional


from adam.curriculum.phase1_curriculum import phase1_instances, PHASE1_CHOOSER
from adam.language_specific.english.english_language_generator import IGNORE_COLORS
from adam.learner import LearningExample

from adam.learner.subset import SubsetLanguageLearner
from adam.ontology import OntologyNode
from adam.ontology.phase1_ontology import BALL, LEARNER, DOG
from adam.ontology.phase1_ontology import GAILA_PHASE_1_ONTOLOGY
from adam.perception.perception_graph import DumpPartialMatchCallback, DebugCallableType
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    object_variable,
    all_possible,
    color_variable,
)


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

    learner = SubsetLanguageLearner(debug_callback=debug_callback)  # type: ignore
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
