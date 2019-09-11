from adam.language.dependency.universal_dependencies import NOUN, VERB
from adam.language_specific.english.english_phase_1_lexicon import (
    GAILA_PHASE_1_ENGLISH_LEXICON,
    MASS_NOUN,
)
from adam.ontology.phase1_ontology import BALL, MILK, PUT


def test_lexicon():
    ball_words = GAILA_PHASE_1_ENGLISH_LEXICON.words_for_node(BALL)
    assert len(ball_words) == 1
    assert ball_words[0].base_form == "ball"
    assert ball_words[0].part_of_speech == NOUN


def test_properties():
    milk_words = GAILA_PHASE_1_ENGLISH_LEXICON.words_for_node(MILK)
    assert MASS_NOUN in milk_words[0].properties


def test_verb_form():
    put_words = GAILA_PHASE_1_ENGLISH_LEXICON.words_for_node(PUT)
    assert put_words[0].base_form == "put"
    assert put_words[0].part_of_speech == VERB
    assert put_words[0].verb_form_3SG_PRS == "puts"
