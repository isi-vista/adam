from adam.language.dependency.universal_dependencies import NOUN
from adam.language.lexicon import LexiconEntry
from adam.language.ontology_dictionary import OntologyLexicon
from adam.language_specific.english.english_phase_1_lexicon import (
    GAILA_PHASE_1_ENGLISH_LEXICON,
)
from adam.ontology.phase2_ontology import (
    GAILA_PHASE_2_ONTOLOGY,
    CHAIR_5,
    CUP_2,
    CUP_3,
    CUP_4,
    CHAIR_2,
    CHAIR_3,
    CHAIR_4,
)

GAILA_PHASE_2_ENGLISH_LEXICON = OntologyLexicon(
    ontology=GAILA_PHASE_2_ONTOLOGY,
    ontology_node_to_word=[
        word
        for word in GAILA_PHASE_1_ENGLISH_LEXICON._ontology_node_to_word.items()  # pylint: disable=protected-access
    ]
    + [
        (CUP_2, LexiconEntry("cup", NOUN, plural_form="cups")),
        (CUP_3, LexiconEntry("cup", NOUN, plural_form="cups")),
        (CUP_4, LexiconEntry("cup", NOUN, plural_form="cups")),
        (CHAIR_2, LexiconEntry("chair", NOUN, plural_form="chairs")),
        (CHAIR_3, LexiconEntry("chair", NOUN, plural_form="chairs")),
        (CHAIR_4, LexiconEntry("chair", NOUN, plural_form="chairs")),
        (CHAIR_5, LexiconEntry("chair", NOUN, plural_form="chairs")),
    ],
)
