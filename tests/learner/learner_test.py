from typing import Optional
import pytest

from adam.curriculum.phase1_curriculum import phase1_instances, PHASE1_CHOOSER
from adam.curriculum.preposition_curriculum import (
    _make_on_training,
    _make_on_tests,
    _make_beside_training,
    _make_beside_tests,
    _make_under_training,
    _make_under_tests,
    _make_over_training,
    _make_over_tests,
    _make_in_training,
    _make_in_tests,
    _make_behind_training,
    _make_behind_tests,
    _make_in_front_training,
    _make_in_front_tests,
)
from adam.language_specific.english.english_language_generator import IGNORE_COLORS
from adam.learner import LearningExample
from adam.learner.preposition_subset import PrepositionSubsetLanguageLearner
from adam.learner.subset import SubsetLanguageLearner
from adam.ontology.phase1_ontology import BALL, LEARNER
from adam.ontology import OntologyNode
from adam.ontology.phase1_ontology import BALL, LEARNER, DOG
from adam.ontology.phase1_ontology import GAILA_PHASE_1_ONTOLOGY
from adam.perception.perception_graph import DumpPartialMatchCallback, DebugCallableType, PerceptionGraph
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


def test_subset_preposition_on_learner():
    learner = PrepositionSubsetLanguageLearner()
    on_train_curriculum = _make_on_training(num_samples=1, noise_objects=False)
    on_test_curriculum = _make_on_tests(num_samples=1, noise_objects=False)

    for (
        _,
        linguistic_description,
        perceptual_representation,
    ) in on_train_curriculum.instances():
        learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )

    for (
        _,
        test_lingustics_description,
        test_perceptual_representation,
    ) in on_test_curriculum.instances():
        descriptions_from_learner = learner.describe(test_perceptual_representation)
        gold = test_lingustics_description.as_token_sequence()
        assert [desc.as_token_sequence() for desc in descriptions_from_learner][0] == gold


def test_subset_preposition_beside_learner():
    learner = PrepositionSubsetLanguageLearner()
    beside_train_curriculum = _make_beside_training(num_samples=1, noise_objects=False)
    beside_test_curriculum = _make_beside_tests(num_samples=1, noise_objects=False)

    for (
        _,
        linguistic_description,
        perceptual_representation,
    ) in beside_train_curriculum.instances():
        learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )

    for (
        _,
        test_linguistic_description,
        test_perceptual_representation,
    ) in beside_test_curriculum.instances():
        descriptions_from_learner = learner.describe(test_perceptual_representation)
        gold = test_linguistic_description.as_token_sequence()
        assert [desc.as_token_sequence() for desc in descriptions_from_learner][0] == gold


def test_subset_preposition_under_learner():
    learner = PrepositionSubsetLanguageLearner()
    under_train_curriculum = _make_under_training(num_samples=1, noise_objects=False)
    under_test_curriculum = _make_under_tests(num_samples=1, noise_objects=False)

    for (
        _,
        linguistic_description,
        perceptual_representation,
    ) in under_train_curriculum.instances():
        learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )

    for (
        _,
        test_linguistic_description,
        test_perceptual_representation,
    ) in under_test_curriculum.instances():
        descriptions_from_learner = learner.describe(test_perceptual_representation)
        gold = test_linguistic_description.as_token_sequence()
        assert [desc.as_token_sequence() for desc in descriptions_from_learner][0] == gold


def test_subset_preposition_over_learner():
    learner = PrepositionSubsetLanguageLearner()
    over_train_curriculum = _make_over_training(num_samples=1, noise_objects=False)
    over_test_curriculum = _make_over_tests(num_samples=1, noise_objects=False)

    for (
        _,
        linguistic_description,
        perceptual_representation,
    ) in over_train_curriculum.instances():
        learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )

    for (
        _,
        test_linguistic_description,
        test_perceptual_representation,
    ) in over_test_curriculum.instances():
        descriptions_from_learner = learner.describe(test_perceptual_representation)
        gold = test_linguistic_description.as_token_sequence()
        assert [desc.as_token_sequence() for desc in descriptions_from_learner][0] == gold


# See https://github.com/isi-vista/adam/issues/422
@pytest.mark.skip(msg="In Preposition Test Temporarily Disabled")
def test_subset_preposition_in_learner():
    learner = PrepositionSubsetLanguageLearner()
    in_train_curriculum = _make_in_training(num_samples=1, noise_objects=False)
    in_test_curriculum = _make_in_tests(num_samples=1, noise_objects=False)

    for (
        _,
        linguistic_description,
        perceptual_representation,
    ) in in_train_curriculum.instances():
        learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )

    for (
        _,
        test_linguistic_description,
        test_perceptual_representation,
    ) in in_test_curriculum.instances():
        descriptions_from_learner = learner.describe(test_perceptual_representation)
        gold = test_linguistic_description.as_token_sequence()
        assert [desc.as_token_sequence() for desc in descriptions_from_learner][0] == gold


def test_subset_preposition_behind_learner():
    learner = PrepositionSubsetLanguageLearner()
    behind_train_curriculum = _make_behind_training(num_samples=1, noise_objects=False)
    behind_test_curriculum = _make_behind_tests(num_samples=1, noise_objects=False)

    for (
        _,
        linguistic_description,
        perceptual_representation,
    ) in behind_train_curriculum.instances():
        learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )

    for (
        _,
        test_linguistic_description,
        test_perceptual_representation,
    ) in behind_test_curriculum.instances():
        descriptions_from_learner = learner.describe(test_perceptual_representation)
        gold = test_linguistic_description.as_token_sequence()
        assert [desc.as_token_sequence() for desc in descriptions_from_learner][0] == gold


def test_subset_preposition_in_front_learner():
    learner = PrepositionSubsetLanguageLearner()
    in_front_train_curriculum = _make_in_front_training(
        num_samples=1, noise_objects=False
    )
    in_front_test_curriculum = _make_in_front_tests(num_samples=1, noise_objects=False)

    for (
        _,
        linguistic_description,
        perceptual_representation,
    ) in in_front_train_curriculum.instances():
        learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )

    for (
        _,
        test_linguistic_description,
        test_perceptual_representation,
    ) in in_front_test_curriculum.instances():
        descriptions_from_learner = learner.describe(test_perceptual_representation)
        gold = test_linguistic_description.as_token_sequence()
        assert [desc.as_token_sequence() for desc in descriptions_from_learner][0] == gold
