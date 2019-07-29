from adam.language.lexicon import NOMINAL
from .testing_lexicon import ENGLISH_TESTING_LEXICON
from .testing_ontology import TRUCK


def test_lexicon():
    truck_words = ENGLISH_TESTING_LEXICON.words_for_node(TRUCK)
    assert len(truck_words) == 1
    assert truck_words[0].base_form == "truck"
    assert set(truck_words[0].properties) == {NOMINAL}


def test_lexicon_entry():
    assert str(NOMINAL) == "+nominal"
