"""
A lexicon of English words for building tests.

It corresponds to the ontology in *_testing_ontology.py*
"""
from adam.language.lexicon import NOMINAL, LexiconEntry
from adam.language.ontology_dictionary import OntologyLexicon

from .testing_ontology import TRUCK, BALL, PERSON, DOG

ENGLISH_TESTING_LEXICON = OntologyLexicon(
    [
        (TRUCK, LexiconEntry("truck", [NOMINAL])),
        (DOG, LexiconEntry("dog", [NOMINAL])),
        (PERSON, LexiconEntry("person", [NOMINAL])),
        (BALL, LexiconEntry("ball", [NOMINAL])),
    ]
)
