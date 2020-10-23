from adam.language.dependency.universal_dependencies import NOUN
from adam.language.lexicon import LexiconEntry
from adam.language.ontology_dictionary import OntologyLexicon
from adam.language_specific.english.english_phase_2_lexicon import (
    GAILA_PHASE_2_ENGLISH_LEXICON,
)
from adam.ontology.integrated_learner_experiement_ontology import (
    INTEGRATED_EXPERIMENT_ONTOLOGY,
    ZUP,
    SPAD,
    DAYGIN,
    MAWG,
    TOMBUR,
    GLIM,
)

INTEGRATED_EXPERIMENT_ENGLISH_LEXICON = OntologyLexicon(
    ontology=INTEGRATED_EXPERIMENT_ONTOLOGY,
    ontology_node_to_word=[
        word
        for word in GAILA_PHASE_2_ENGLISH_LEXICON._ontology_node_to_word.items()  # pylint: disable=protected-access
    ]
    + [
        (ZUP, LexiconEntry("zup", NOUN, plural_form="zups")),
        (SPAD, LexiconEntry("spad", NOUN, plural_form="spads")),
        (DAYGIN, LexiconEntry("daygin", NOUN, plural_form="daygins")),
        (MAWG, LexiconEntry("mawg", NOUN, plural_form="mawg")),
        (TOMBUR, LexiconEntry("tombur", NOUN, plural_form="tomburs")),
        (GLIM, LexiconEntry("glim", NOUN, plural_form="glims")),
    ],
)
