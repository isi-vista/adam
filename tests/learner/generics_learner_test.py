import random
from typing import Iterable

import pytest

from adam.axes import AxesInfo
from adam.curriculum.curriculum_utils import PHASE1_CHOOSER_FACTORY, phase1_instances
from adam.curriculum.phase1_curriculum import (
    _make_generic_statements_curriculum,
    _make_eat_curriculum,
    _make_colour_predicates_curriculum,
    _make_kind_predicates_curriculum,
    _make_objects_with_colors_curriculum,
)
from adam.language.language_utils import phase2_language_generator
from adam.learner import LearningExample
from adam.learner.attributes import SubsetAttributeLearner
from adam.learner.generics import SimpleGenericsLearner
from adam.learner.integrated_learner import IntegratedTemplateLearner
from adam.learner.language_mode import LanguageMode
from adam.learner.plurals import SubsetPluralLearner
from adam.learner.verbs import SubsetVerbLearner
from adam.ontology.ontology import Ontology
from adam.ontology.phase1_ontology import (
    GAILA_PHASE_1_ONTOLOGY,
    PHASE_1_CURRICULUM_OBJECTS,
    LIQUID,
    is_recognized_particular,
)
from adam.random_utils import RandomChooser
from adam.situation import SituationObject
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from tests.learner import LANGUAGE_MODE_TO_TEMPLATE_LEARNER_OBJECT_RECOGNIZER


def integrated_learner_factory(language_mode: LanguageMode):
    rng = random.Random()
    rng.seed(0)
    return IntegratedTemplateLearner(
        object_learner=LANGUAGE_MODE_TO_TEMPLATE_LEARNER_OBJECT_RECOGNIZER[language_mode],
        attribute_learner=SubsetAttributeLearner(
            ontology=GAILA_PHASE_1_ONTOLOGY, beam_size=5, language_mode=language_mode
        ),
        plural_learner=SubsetPluralLearner(
            ontology=GAILA_PHASE_1_ONTOLOGY, beam_size=5, language_mode=language_mode
        ),
        action_learner=SubsetVerbLearner(
            ontology=GAILA_PHASE_1_ONTOLOGY, beam_size=5, language_mode=language_mode
        ),
        generics_learner=SimpleGenericsLearner(),
    )


def run_generics_test(learner, language_mode):
    def build_object_multiples_situations(
        ontology: Ontology, *, samples_per_object: int = 3, chooser: RandomChooser
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

    language_generator = phase2_language_generator(language_mode)
    # Teach plurals
    plurals = phase1_instances(
        "plurals pretraining",
        build_object_multiples_situations(
            ontology=GAILA_PHASE_1_ONTOLOGY, chooser=PHASE1_CHOOSER_FACTORY()
        ),
        language_generator=language_generator,
    )

    curricula = [
        # Actions - verbs in generics
        _make_eat_curriculum(10, 0, language_generator),
        # Plurals
        plurals,
        # Color attributes
        _make_objects_with_colors_curriculum(None, None, language_generator),
        # Predicates
        _make_colour_predicates_curriculum(None, None, language_generator),
        _make_kind_predicates_curriculum(None, None, language_generator),
        # Generics
        _make_generic_statements_curriculum(
            num_samples=3, noise_objects=0, language_generator=language_generator
        ),
    ]

    for curriculum in curricula:
        for (
            _,
            linguistic_description,
            perceptual_representation,
        ) in curriculum.instances():
            # Get the object matches first - preposition learner can't learn without already recognized objects
            learner.observe(
                LearningExample(perceptual_representation, linguistic_description)
            )

    # learner.generics_learner.log_hypotheses(Path(f"./renders/{language_mode.name}"))


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
def test_generics(language_mode):
    learner = integrated_learner_factory(language_mode)
    run_generics_test(learner, language_mode)
