from adam.language.dependency.universal_dependencies import PROPER_NOUN, NOUN, VERB
from adam.language.lexicon import LexiconEntry
from adam.language.ontology_dictionary import OntologyLexicon
from adam.ontology.phase1_ontology import MOM, BALL, TABLE, PUT

GAILA_PHASE_1_ENGLISH_LEXICON = OntologyLexicon(
    (
        (MOM, LexiconEntry("Mom", PROPER_NOUN)),
        (BALL, LexiconEntry("ball", NOUN)),
        (TABLE, LexiconEntry("table", NOUN)),
        (PUT, LexiconEntry("put", VERB)),
    )
)
