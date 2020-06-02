#import universal dependencies
from adam.language.dependency.universal_dependencies import (
    ADJECTIVE,
    NOUN,
    PROPER_NOUN,
    VERB,
)
#import lexicon classes
from adam.language.lexicon import LexiconEntry, LexiconProperty
#import OntologyLexicon class
from adam.language.ontology_dictionary import OntologyLexicon

#TODO: update this when Chinese syntax file ready
from adam.language_specific.english.english_syntax import (
    FIRST_PERSON,
    SECOND_PERSON,
    NOMINATIVE,
    ACCUSATIVE,
)
#import ontology items
from adam.ontology.phase1_ontology import (
    BABY,
    BALL,
    BIRD,
    BLACK,
    BLUE,
    BOOK,
    BOX,
    CAR,
    CHAIR,
    COME,
    COOKIE,
    CUP,
    DAD,
    DOG,
    DOOR,
    DRINK,
    EAT,
    FALL,
    FLY,
    GAILA_PHASE_1_ONTOLOGY,
    GIVE,
    GO,
    GREEN,
    GROUND,
    HAND,
    HAS,
    HAT,
    HEAD,
    HOUSE,
    JUICE,
    JUMP,
    MILK,
    MOM,
    MOVE,
    PUSH,
    PUT,
    RED,
    ROLL,
    SIT,
    SPIN,
    TABLE,
    TAKE,
    THROW,
    TRANSPARENT,
    TRUCK,
    WATER,
    WHITE,
    LIGHT_BROWN,
    DARK_BROWN,
)

#TODO: do we need to specify mass nouns for Chinese since most scholars say nearly all Chinese nouns are mass?
MASS_NOUN = LexiconProperty("mass-noun")
#ditransitive property for verbs
ALLOWS_DITRANSITIVE = LexiconProperty("allows-ditransitive")


'''Chinese transcriptions use Yale romanization here since this allows us to maintain UTF-8 encoding 
(v.s. Pinyin, Simplified Chinese, etc.)'''

#TODO is it ok for there to be spaces in the strings or should they be replaced?
#first person nominative & accusative pronoun lexicon entry
I = LexiconEntry(
    "wo3",
    NOUN,
    plural_form="wo3 men",
    intrinsic_morphosyntactic_properties=[FIRST_PERSON],
)

#second person nominative & accusative pronoun lexicon entry
YOU = LexiconEntry(
    "ni3",
    NOUN,
    plural_form="ni3 men",
    intrinsic_morphosyntactic_properties=[SECOND_PERSON]
)

#define the main lexicon
#TODO: do plural forms need to be specified or can they be left?
#TODO: do we want to keep the mass noun / non-mass noun distinction even though Chinese doesn't make it morphologically?
GAILA_PHASE_1_CHINESE_LEXICON = OntologyLexicon(
    ontology=GAILA_PHASE_1_ONTOLOGY,
    ontology_node_to_word=(
        (GROUND, LexiconEntry("di4 myan4", NOUN)),
        (MOM, LexiconEntry("ma1", PROPER_NOUN)),
        (BALL, LexiconEntry("chyou2", NOUN)),
        (TABLE, LexiconEntry("jwo1", NOUN)),
        (PUT, LexiconEntry("fang4", NOUN)),
        (PUSH, LexiconEntry("twei1", NOUN)),
        (BOOK, LexiconEntry("shu1", NOUN)),
        (HOUSE, LexiconEntry("wu1", NOUN)),
        (CAR, LexiconEntry("chi4 che1", NOUN)),
        (WATER, LexiconEntry("shwei3", NOUN, [MASS_NOUN])),
        (JUICE, LexiconEntry("gwo3 jr1", NOUN, [MASS_NOUN])),
        (CUP, LexiconEntry("bei1 dz", NOUN)),
        (BOX, LexiconEntry("syang1", NOUN)),
        (CHAIR, LexiconEntry("yi3 dz", NOUN)),
        (HEAD, LexiconEntry("tou2", NOUN)),
        (MILK, LexiconEntry("nyou2 nai3", NOUN, [MASS_NOUN])),
        (HAND, LexiconEntry("shou3", NOUN))
    )
)

