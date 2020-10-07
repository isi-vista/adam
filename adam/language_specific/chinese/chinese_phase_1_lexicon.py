"""We use Yale romanization in this code to maintain UTF-8 encoding;
spaces indicate borders between Chinese characters.

Currently, mass nouns are specified despite this not being realised in
the same way in Chinese as in English.

We don't provide plural specifications for nouns since this isn't a
morphologically-realized feature in Chinese.

This vocabulary has been verified by a native speaker.
"""
from adam.language_specific import (
    FIRST_PERSON,
    SECOND_PERSON,
    MASS_NOUN,
    ALLOWS_DITRANSITIVE,
)
from adam.language.dependency.universal_dependencies import (
    ADJECTIVE,
    NOUN,
    PROPER_NOUN,
    VERB,
)
from adam.language.lexicon import LexiconEntry
from adam.language.ontology_dictionary import OntologyLexicon
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
    CAT,
    BEAR,
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
    WATERMELON,
    PASS,
    WALK,
    PAPER,
    BED,
    NONHUMAN_ANIMAL,
    PERSON,
)

ME = LexiconEntry("wo3", NOUN, intrinsic_morphosyntactic_properties=[FIRST_PERSON])
YOU = LexiconEntry("ni3", NOUN, intrinsic_morphosyntactic_properties=[SECOND_PERSON])
RUN = LexiconEntry("pau3", VERB)
GRAB = LexiconEntry("chyang3 na2 chi3", VERB)
SHOVE = LexiconEntry("yung4 li4 twei1", VERB)
# throw and toss are the same in Chinese
TOSS = LexiconEntry("reng1", VERB)
GAILA_PHASE_1_CHINESE_LEXICON = OntologyLexicon(
    ontology=GAILA_PHASE_1_ONTOLOGY,
    ontology_node_to_word=(
        (PERSON, LexiconEntry("ren2", NOUN)),
        (GROUND, LexiconEntry("di4 myan4", NOUN, counting_classifier="jang1")),
        (NONHUMAN_ANIMAL, LexiconEntry("dung4 wu4", NOUN, counting_classifier="jr1")),
        (BED, LexiconEntry("chwang2", NOUN, counting_classifier="jang1")),
        (PAPER, LexiconEntry("jr3", NOUN, counting_classifier="jang1")),
        (MOM, LexiconEntry("ma1 ma1", PROPER_NOUN)),
        (BALL, LexiconEntry("chyou2", NOUN)),
        (TABLE, LexiconEntry("jwo1 dz", NOUN, counting_classifier="jang1")),
        (PUT, LexiconEntry("fang4", VERB)),
        (PUSH, LexiconEntry("twei1", VERB)),
        (BOOK, LexiconEntry("shu1", NOUN, counting_classifier="ben3")),
        (HOUSE, LexiconEntry("wu1", NOUN, counting_classifier="jyan1")),
        (CAR, LexiconEntry("chi4 che1", NOUN, counting_classifier="lyang4")),
        (WATER, LexiconEntry("shwei3", NOUN, [MASS_NOUN], counting_classifier="bei1")),
        (JUICE, LexiconEntry("gwo3 jr1", NOUN, [MASS_NOUN], counting_classifier="bei1")),
        (CUP, LexiconEntry("bei1 dz", NOUN)),
        (BOX, LexiconEntry("syang1 dz", NOUN)),
        (CHAIR, LexiconEntry("yi3 dz", NOUN, counting_classifier="ba3")),
        (HEAD, LexiconEntry("tou2", NOUN)),
        (MILK, LexiconEntry("nyou2 nai3", NOUN, [MASS_NOUN], counting_classifier="bei1")),
        (HAND, LexiconEntry("shou3", NOUN, counting_classifier="jr1")),
        (TRUCK, LexiconEntry("ka3 che1", NOUN, counting_classifier="lyang4")),
        (DOOR, LexiconEntry("men2", NOUN, counting_classifier="shan4")),
        (HAT, LexiconEntry("mau4 dz", NOUN, counting_classifier="ding3")),
        (COOKIE, LexiconEntry("chyu1 chi2 bing3", NOUN, counting_classifier="kwai4")),
        (DAD, LexiconEntry("ba4 ba4", PROPER_NOUN)),
        (BABY, LexiconEntry("bau3 bau3", NOUN)),
        (DOG, LexiconEntry("gou3", NOUN, counting_classifier="jr1")),
        (CAT, LexiconEntry("mau1", NOUN, counting_classifier="jr1")),
        (BIRD, LexiconEntry("nyau3", NOUN, counting_classifier="jr1")),
        (BEAR, LexiconEntry("syung2", NOUN, counting_classifier="jr1")),
        (GO, LexiconEntry("chyu4", VERB)),
        (COME, LexiconEntry("lai2", VERB)),
        (TAKE, LexiconEntry("na2", VERB)),
        (EAT, LexiconEntry("chr1", VERB)),
        (GIVE, LexiconEntry("gei3", VERB, properties=[ALLOWS_DITRANSITIVE])),
        (SPIN, LexiconEntry("sywan2 jwan3", VERB)),
        (SIT, LexiconEntry("dzwo4", VERB)),
        (DRINK, LexiconEntry("he1", VERB)),
        (FALL, LexiconEntry("dye2 dau3", VERB)),
        # throw isn't ditransitive in Chinese
        (THROW, LexiconEntry("reng1", VERB)),
        (MOVE, LexiconEntry("yi2 dung4", VERB)),
        (JUMP, LexiconEntry("tyau4", VERB)),
        (HAS, LexiconEntry("you3", VERB)),
        (ROLL, LexiconEntry("gwun3", VERB)),
        (FLY, LexiconEntry("fei1", VERB)),
        (RED, LexiconEntry("hung2 se4", ADJECTIVE)),
        (BLUE, LexiconEntry("lan2 se4", ADJECTIVE)),
        (GREEN, LexiconEntry("lyu4 se4", ADJECTIVE)),
        (BLACK, LexiconEntry("hei1 se4", ADJECTIVE)),
        (WHITE, LexiconEntry("bai2 se4", ADJECTIVE)),
        (TRANSPARENT, LexiconEntry("tou4 ming2", ADJECTIVE)),
        (LIGHT_BROWN, LexiconEntry("chyan3 he2 se4", ADJECTIVE)),
        (DARK_BROWN, LexiconEntry("shen1 dzung1 se4", ADJECTIVE)),
        (WATERMELON, LexiconEntry("syi1 gwa1", NOUN)),
        (PASS, LexiconEntry("di4", VERB)),
        (WALK, LexiconEntry("bu4 sying2", VERB)),
    ),
)
