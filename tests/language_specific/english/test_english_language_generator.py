from more_itertools import only

from adam.language_specific.english.english_phase_1_lexicon import (
    GAILA_PHASE_1_ENGLISH_LEXICON,
)
from adam.language_specific.english.english_syntax import (
    SIMPLE_ENGLISH_DEPENDENCY_TREE_LINEARIZER,
)
from adam.language_specific.english.english_language_generator import (
    SimpleRuleBasedEnglishLanguageGenerator,
)
from adam.ontology.phase1_ontology import GAILA_PHASE_1_ONTOLOGY, BALL, MOM
from adam.random_utils import FixedIndexChooser
from adam.situation import HighLevelSemanticsSituation, SituationObject


def test_common_noun():
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, objects=[SituationObject(BALL)]
    )
    generator = SimpleRuleBasedEnglishLanguageGenerator(
        ontology_lexicon=GAILA_PHASE_1_ENGLISH_LEXICON,
        dependency_tree_linearizer=SIMPLE_ENGLISH_DEPENDENCY_TREE_LINEARIZER,
    )
    assert only(
        generator.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("a", "ball")


def test_proper_noun():
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, objects=[SituationObject(MOM)]
    )
    generator = SimpleRuleBasedEnglishLanguageGenerator(
        ontology_lexicon=GAILA_PHASE_1_ENGLISH_LEXICON,
        dependency_tree_linearizer=SIMPLE_ENGLISH_DEPENDENCY_TREE_LINEARIZER,
    )
    assert only(
        generator.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("Mom",)
