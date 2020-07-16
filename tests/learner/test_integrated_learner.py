from more_itertools import one
import pytest
from adam.learner.integrated_learner import IntegratedTemplateLearner
from adam.learner.language_mode import LanguageMode
from adam.ontology.phase1_ontology import DAD, GAILA_PHASE_1_ONTOLOGY
from adam.perception import PerceptualRepresentation
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator,
)
from adam.random_utils import RandomChooser
from adam.situation import SituationObject
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation

from tests.learner import LANGUAGE_MODE_TO_TEMPLATE_LEARNER_OBJECT_RECOGNIZER


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
def test_with_object_recognizer(language_mode):
    integrated_learner = IntegratedTemplateLearner(
        object_learner=LANGUAGE_MODE_TO_TEMPLATE_LEARNER_OBJECT_RECOGNIZER[language_mode],
        attribute_learner=None,
        relation_learner=None,
        action_learner=None,
    )

    dad_situation_object = SituationObject.instantiate_ontology_node(
        ontology_node=DAD, ontology=GAILA_PHASE_1_ONTOLOGY
    )
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[dad_situation_object]
    )
    perception_generator = HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator(
        GAILA_PHASE_1_ONTOLOGY
    )
    # We explicitly exclude ground in perception generation

    # this generates a static perception...
    perception = perception_generator.generate_perception(
        situation, chooser=RandomChooser.for_seed(0), include_ground=False
    )

    # so we need to construct a dynamic one by hand from two identical scenes
    dynamic_perception = PerceptualRepresentation(
        frames=[perception.frames[0], perception.frames[0]]
    )

    descriptions = integrated_learner.describe(dynamic_perception)

    assert len(descriptions) == 1
    assert (
        language_mode == LanguageMode.ENGLISH
        and one(descriptions.keys()).as_token_sequence() == ("Dad",)
    ) or (
        language_mode == LanguageMode.CHINESE
        and one(descriptions.keys()).as_token_sequence() == ("ba4 ba4",)
    )
