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
from adam.learner import LearningExample
from adam.learner.preposition_subset import PrepositionSubsetLanguageLearner
import pytest


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
        assert descriptions_from_learner
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
