from typing import TypeVar
from pickle import HIGHEST_PROTOCOL
from io import BytesIO, SEEK_SET

from adam.curriculum.curriculum_utils import phase1_instances, PHASE1_CHOOSER_FACTORY
from adam.language.language_utils import phase1_language_generator
from adam.ontology.phase1_ontology import GAILA_PHASE_1_ONTOLOGY
from adam.pickle import AdamPickler, AdamUnpickler
from adam.learner import LearningExample
from adam.learner.integrated_learner import IntegratedTemplateLearner
from adam.learner.language_mode import LanguageMode
from adam.learner.object_recognizer import SHARED_WORLD_ITEMS
from adam.perception import GROUND_PERCEPTION, LEARNER_PERCEPTION
from adam.situation.templates.phase1_templates import sampled
from tests.learner import LANGUAGE_MODE_TO_TEMPLATE_LEARNER_OBJECT_RECOGNIZER
from tests.learner.subset_verb_learner_test import drink_test_template


T = TypeVar("T")  # pylint:disable=invalid-name


def _pickle_and_unpickle_object(input_object: T) -> T:
    stream = BytesIO()
    pickler = AdamPickler(file=stream, protocol=HIGHEST_PROTOCOL)
    pickler.dump(input_object)

    stream.seek(0, SEEK_SET)
    unpickler = AdamUnpickler(file=stream)
    return unpickler.load()


def test_pickle_preserves_shared_world_item_identity():
    for item in SHARED_WORLD_ITEMS:
        new_item = _pickle_and_unpickle_object(item)
        assert new_item is item


def test_pickle_preserves_ground_perception_identity():
    new_ground_perception = _pickle_and_unpickle_object(GROUND_PERCEPTION)
    assert new_ground_perception is GROUND_PERCEPTION


def test_pickle_preserves_learner_perception_identity():
    new_ground_perception = _pickle_and_unpickle_object(LEARNER_PERCEPTION)
    assert new_ground_perception is LEARNER_PERCEPTION


def test_object_recognition_with_drink_perception():
    """
    Regression test to confirm we can perform object recognition on a pickled and unpickled "drink"
    perception. If we do this using the normal pickling interface we get an error. This test checks
    that we don't run into such an error when we instead pickle and unpickle the perception using
    the AdamPickler and AdamUnpickler.

    See https://github.com/isi-vista/adam/issues/958.
    """
    language_mode = LanguageMode.ENGLISH
    template = drink_test_template()
    curriculum = phase1_instances(
        "train",
        sampled(
            template,
            max_to_sample=3,
            ontology=GAILA_PHASE_1_ONTOLOGY,
            chooser=PHASE1_CHOOSER_FACTORY(),
            block_multiple_of_the_same_type=True,
        ),
        language_generator=phase1_language_generator(language_mode),
    )

    object_recognizer = LANGUAGE_MODE_TO_TEMPLATE_LEARNER_OBJECT_RECOGNIZER[language_mode]
    learner = IntegratedTemplateLearner(object_learner=object_recognizer)

    for (_, linguistic_description, perceptual_representation) in curriculum.instances():
        new_perceptual_representation = _pickle_and_unpickle_object(
            perceptual_representation
        )
        learner.observe(
            LearningExample(new_perceptual_representation, linguistic_description)
        )
