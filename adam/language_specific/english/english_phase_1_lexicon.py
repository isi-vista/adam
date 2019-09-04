from adam.language.dependency.universal_dependencies import PROPER_NOUN, NOUN, VERB
from adam.language.lexicon import LexiconEntry
from adam.language.ontology_dictionary import OntologyLexicon
from adam.ontology.phase1_ontology import (
    MOM, BALL, TABLE, PUT, PUSH,
    BOOK, HOUSE, CAR, WATER, JUICE,
    CUP, BOX, CHAIR, HEAD, MILK,
    HAND, TRUCK, DOOR, HAT, COOKIE,
    DAD, BABY, DOG, BIRD
)

GAILA_PHASE_1_ENGLISH_LEXICON = OntologyLexicon(
    (
        (MOM, LexiconEntry("Mom", PROPER_NOUN)),
        (BALL, LexiconEntry("ball", NOUN)),
        (TABLE, LexiconEntry("table", NOUN)),
        (PUT, LexiconEntry("put", VERB)),
        (PUSH, LexiconEntry("push", VERB)),
        (BOOK, LexiconEntry("book", NOUN)),
        (HOUSE, LexiconEntry("house", NOUN)),
        (CAR, LexiconEntry("car", NOUN)),
        (WATER, LexiconEntry("water", NOUN)),
        (JUICE, LexiconEntry("juice", NOUN)),
        (CUP, LexiconEntry("cup", NOUN)),
        (BOX, LexiconEntry("box", NOUN)),
        (CHAIR, LexiconEntry("chair", NOUN)),
        (HEAD, LexiconEntry("head", NOUN)),
        (MILK, LexiconEntry("milk", NOUN)),
        (HAND, LexiconEntry("hand", NOUN)),
        (TRUCK, LexiconEntry("truck", NOUN)),
        (DOOR, LexiconEntry("door", NOUN)),
        (HAT, LexiconEntry("hat", NOUN)),
        (COOKIE, LexiconEntry("cookie", NOUN)),
        (DAD, LexiconEntry("Dad", PROPER_NOUN)),
        (BABY, LexiconEntry("bird", NOUN)),
        (DOG, LexiconEntry("dog", NOUN)),
        (BIRD, LexiconEntry("bird", NOUN))
    )
)
