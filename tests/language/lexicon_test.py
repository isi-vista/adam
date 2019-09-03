from adam.language.dependency.universal_dependencies import NOUN
from adam.language_specific.english.english_phase_1_lexicon import (
    GAILA_PHASE_1_ENGLISH_LEXICON,
)
from adam.ontology.phase1_ontology import BALL


def test_lexicon():
    ball_words = GAILA_PHASE_1_ENGLISH_LEXICON.words_for_node(BALL)
    assert len(ball_words) == 1
    assert ball_words[0].base_form == "ball"
    assert ball_words[0].part_of_speech == NOUN
