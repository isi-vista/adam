import random
from pathlib import Path
from typing import Iterable

import pytest

from adam.axes import AxesInfo
from adam.curriculum.curriculum_utils import PHASE1_CHOOSER_FACTORY, phase1_instances
from adam.curriculum.phase1_curriculum import _make_plural_objects_curriculum, _make_generic_statements_curriculum, \
    _make_eat_curriculum, _make_drink_curriculum, _make_sit_curriculum, _make_jump_curriculum, _make_fly_curriculum
from adam.language.language_utils import phase1_language_generator
from adam.learner import LearningExample
from adam.learner.generics import PursuitGenericsLearner
from adam.learner.integrated_learner import IntegratedTemplateLearner
from adam.learner.language_mode import LanguageMode
from adam.learner.plurals import SubsetPluralLearnerNew
from adam.learner.verbs import SubsetVerbLearnerNew
from adam.ontology.ontology import Ontology
from adam.ontology.phase1_ontology import (
    GAILA_PHASE_1_ONTOLOGY,
    PHASE_1_CURRICULUM_OBJECTS,
    LIQUID,
    is_recognized_particular,
)
from adam.perception.perception_graph import DumpPartialMatchCallback
from adam.random_utils import RandomChooser
from adam.situation import SituationObject
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from tests.learner import LANGUAGE_MODE_TO_TEMPLATE_LEARNER_OBJECT_RECOGNIZER


def integrated_learner_factory(language_mode: LanguageMode):
    rng = random.Random()
    rng.seed(0)
    return IntegratedTemplateLearner(
        object_learner=LANGUAGE_MODE_TO_TEMPLATE_LEARNER_OBJECT_RECOGNIZER[language_mode],
        attribute_learner=SubsetPluralLearnerNew(
            ontology=GAILA_PHASE_1_ONTOLOGY, beam_size=5, language_mode=language_mode
        ),
        generics_learner=PursuitGenericsLearner(
            learning_factor=0.5,
            graph_match_confirmation_threshold=0.7,
            lexicon_entry_threshold=0.7,
            rng=rng,
            smoothing_parameter=0.002,
            ontology=GAILA_PHASE_1_ONTOLOGY,
            language_mode=language_mode,
        )
    )


def run_generics_test(learner, language_generator, language_mode):
    def build_object_multiples_situations(
        ontology: Ontology, *, samples_per_object: int = 10, chooser: RandomChooser
    ) -> Iterable[HighLevelSemanticsSituation]:
        for object_type in PHASE_1_CURRICULUM_OBJECTS:
            # Exclude slow objects for now
            if object_type.handle in ["bird", "dog", "truck"]:
                continue
            is_liquid = ontology.has_all_properties(object_type, [LIQUID])
            # don't want multiples of named people
            if not is_recognized_particular(ontology, object_type) and not is_liquid:
                for _ in range(samples_per_object):
                    num_objects = chooser.choice(range(2, 4))
                    yield HighLevelSemanticsSituation(
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        salient_objects=[
                            SituationObject.instantiate_ontology_node(
                                ontology_node=object_type,
                                debug_handle=object_type.handle + f"_{idx}",
                                ontology=GAILA_PHASE_1_ONTOLOGY,
                            )
                            for idx in range(num_objects)
                        ],
                        axis_info=AxesInfo(),
                    )

    # actions = list(_make_eat_curriculum(10, 0, language_generator).instances())
    # actions.extend(_make_drink_curriculum(10, 0, language_generator).instances())
    # # actions.extend(_make_sit_curriculum(20, 0, language_generator).instances())
    # actions.extend(_make_jump_curriculum(10, 0, language_generator).instances())
    # actions.extend(_make_fly_curriculum(10, 0, language_generator).instances())
    # Teach plurals
    plurals = list(phase1_instances(
        "plurals pretraining",
        build_object_multiples_situations(
            ontology=GAILA_PHASE_1_ONTOLOGY, chooser=PHASE1_CHOOSER_FACTORY()
        ),
        language_generator=language_generator,
    ).instances())
    for (
        _,
        linguistic_description,
        perceptual_representation,
    ) in plurals:
        # Get the object matches first - preposition learner can't learn without already recognized objects
        learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )

    generics_curriculum = _make_generic_statements_curriculum(
        num_samples=20, noise_objects=0, language_generator=language_generator
    )
    for (
        _,
        linguistic_description,
        perceptual_representation,
    ) in generics_curriculum.instances():
        # Get the object matches first - preposition learner can't learn without already recognized objects
        print(linguistic_description)
        learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )
    learner.log_hypotheses(Path('/renders/'))
    # test_curriculum = _make_plural_objects_curriculum(
    #     10, 0, language_generator=language_generator
    # )
    # for (
    #     _,
    #     test_lingustics_description,
    #     test_perceptual_representation,
    # ) in test_curriculum.instances():
    #     descriptions_from_learner = learner.describe(test_perceptual_representation)
    #     gold = test_lingustics_description.as_token_sequence()
    #     assert descriptions_from_learner
    #     # Skip "two" in Chinese for now - there are too many counting classifiers that make it hard to learn
    #     if language_mode == LanguageMode.CHINESE and "lyang3" in gold:
    #         continue
    #     assert gold in [desc.as_token_sequence() for desc in descriptions_from_learner]


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH
    # , LanguageMode.CHINESE
                                           ])
@pytest.mark.parametrize("learner", [integrated_learner_factory])
def test_generics(language_mode, learner):
    run_generics_test(
        learner(language_mode), phase1_language_generator(language_mode), language_mode
    )
