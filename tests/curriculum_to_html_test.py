from typing import Tuple

from immutablecollections import ImmutableSet

from adam import curriculum_to_html
from adam.experiment.instance_group import ExplicitWithSituationInstanceGroup
from adam.language_specific.english.english_language_generator import (
    SimpleRuleBasedEnglishLanguageGenerator,
)
from adam.language_specific.english.english_phase_1_lexicon import (
    GAILA_PHASE_1_ENGLISH_LEXICON,
)
from adam.ontology.phase1_ontology import GAILA_PHASE_1_ONTOLOGY, TABLE
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator,
)
from adam.random_utils import FixedIndexChooser
from adam.situation import SituationObject
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation

_SIMPLE_GENERATOR = SimpleRuleBasedEnglishLanguageGenerator(
    ontology_lexicon=GAILA_PHASE_1_ENGLISH_LEXICON
)

_PERCEPTION_GENERATOR = HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator(
    GAILA_PHASE_1_ONTOLOGY
)


def test_simple_curriculum_html():
    instances = []
    table = SituationObject(TABLE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        objects=ImmutableSet[table],
        # actions=ImmutableSet[],
        # relations=ImmutableSet[]
    )
    linguistics = _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    perception = _PERCEPTION_GENERATOR.generate_perception(
        situation=situation, chooser=FixedIndexChooser(0)
    )
    group = ExplicitWithSituationInstanceGroup(
        name="Test Group", instances=Tuple[Tuple[situation, linguistics, perception]]
    )
    instances.append(group)
    htmlExporter = curriculum_to_html.CurriculumToHtml()
    htmlExporter.generate(instances, "./", overwrite=True, title="Test Objects")
