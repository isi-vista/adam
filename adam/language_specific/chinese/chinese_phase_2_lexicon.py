from adam.language.dependency.universal_dependencies import NOUN
from adam.language.lexicon import LexiconEntry
from adam.language.ontology_dictionary import OntologyLexicon
from adam.language_specific.chinese.chinese_phase_1_lexicon import (
    GAILA_PHASE_1_CHINESE_LEXICON,
)
from adam.ontology.phase2_ontology import (
    CHAIR_2,
    CHAIR_3,
    CHAIR_4,
    CHAIR_5,
    CUP_2,
    CUP_3,
    CUP_4,
    GAILA_PHASE_2_ONTOLOGY,
)

GAILA_PHASE_2_CHINESE_LEXICON = OntologyLexicon(
    ontology=GAILA_PHASE_2_ONTOLOGY,
    ontology_node_to_word=[
        word
        for word in GAILA_PHASE_1_CHINESE_LEXICON._ontology_node_to_word.items()  # pylint: disable=protected-access
    ]
    + [
        (CUP_2, LexiconEntry("bei1 dz", NOUN)),
        (CUP_3, LexiconEntry("bei1 dz", NOUN)),
        (CUP_4, LexiconEntry("bei1 dz", NOUN)),
        (CHAIR_2, LexiconEntry("yi3 dz", NOUN)),
        (CHAIR_3, LexiconEntry("yi3 dz", NOUN)),
        (CHAIR_4, LexiconEntry("yi3 dz", NOUN)),
        (CHAIR_5, LexiconEntry("yi3 dz", NOUN)),
    ],
)
