from adam.language.dependency.universal_dependencies import (
    ADJECTIVE,
    NOUN,
    PROPER_NOUN,
    VERB,
)
from adam.language.lexicon import LexiconEntry
from adam.language.ontology_dictionary import OntologyLexicon
from adam.language_specific import (
    FIRST_PERSON,
    SECOND_PERSON,
    NOMINATIVE,
    ACCUSATIVE,
    MASS_NOUN,
    ALLOWS_DITRANSITIVE,
)
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
    PASS,
    WATERMELON,
    WALK,
)

I = LexiconEntry(  # noqa: E741
    "I",
    NOUN,
    plural_form="we",
    intrinsic_morphosyntactic_properties=[FIRST_PERSON, NOMINATIVE],
)
ME = LexiconEntry(  # noqa: E741
    "me",
    NOUN,
    plural_form="us",
    intrinsic_morphosyntactic_properties=[FIRST_PERSON, ACCUSATIVE],
)

YOU = LexiconEntry(
    "you", NOUN, plural_form="y'all", intrinsic_morphosyntactic_properties=[SECOND_PERSON]
)

GRAB = LexiconEntry("grab", VERB, verb_form_sg3_prs="grabs")

RUN = LexiconEntry("run", VERB, verb_form_sg3_prs="runs")
TOSS = LexiconEntry(
    "toss", VERB, verb_form_sg3_prs="tosses", properties=[ALLOWS_DITRANSITIVE]
)

SHOVE = LexiconEntry("shove", VERB, verb_form_sg3_prs="shoves")


GAILA_PHASE_1_ENGLISH_LEXICON = OntologyLexicon(
    ontology=GAILA_PHASE_1_ONTOLOGY,
    ontology_node_to_word=(
        (GROUND, LexiconEntry("ground", NOUN, plural_form="grounds")),
        (MOM, LexiconEntry("Mom", PROPER_NOUN)),
        (BALL, LexiconEntry("ball", NOUN, plural_form="balls")),
        (WATERMELON, LexiconEntry("watermelon", NOUN, plural_form="watermelons")),
        (TABLE, LexiconEntry("table", NOUN, plural_form="tables")),
        (PUT, LexiconEntry("put", VERB, verb_form_sg3_prs="puts")),
        (PUSH, LexiconEntry("push", VERB, verb_form_sg3_prs="pushes")),
        (BOOK, LexiconEntry("book", NOUN, plural_form="books")),
        (HOUSE, LexiconEntry("house", NOUN, plural_form="houses")),
        (CAR, LexiconEntry("car", NOUN, plural_form="cars")),
        (WATER, LexiconEntry("water", NOUN, [MASS_NOUN])),
        (JUICE, LexiconEntry("juice", NOUN, [MASS_NOUN])),
        (CUP, LexiconEntry("cup", NOUN, plural_form="cups")),
        (BOX, LexiconEntry("box", NOUN, plural_form="boxes")),
        (CHAIR, LexiconEntry("chair", NOUN, plural_form="chairs")),
        (HEAD, LexiconEntry("head", NOUN, plural_form="heads")),
        (MILK, LexiconEntry("milk", NOUN, [MASS_NOUN])),
        (HAND, LexiconEntry("hand", NOUN, plural_form="hands")),
        (TRUCK, LexiconEntry("truck", NOUN, plural_form="trucks")),
        (DOOR, LexiconEntry("door", NOUN, plural_form="doors")),
        (HAT, LexiconEntry("hat", NOUN, plural_form="hats")),
        (COOKIE, LexiconEntry("cookie", NOUN, plural_form="cookies")),
        (DAD, LexiconEntry("Dad", PROPER_NOUN)),
        (BABY, LexiconEntry("baby", NOUN, plural_form="babies")),
        (DOG, LexiconEntry("dog", NOUN, plural_form="dogs")),
        (BIRD, LexiconEntry("bird", NOUN, plural_form="birds")),
        (GO, LexiconEntry("go", VERB, verb_form_sg3_prs="goes")),
        (COME, LexiconEntry("come", VERB, verb_form_sg3_prs="comes")),
        (TAKE, LexiconEntry("take", VERB, verb_form_sg3_prs="takes")),
        (EAT, LexiconEntry("eat", VERB, verb_form_sg3_prs="eats")),
        (
            GIVE,
            LexiconEntry(
                "give", VERB, verb_form_sg3_prs="gives", properties=[ALLOWS_DITRANSITIVE]
            ),
        ),
        (SPIN, LexiconEntry("spin", VERB, verb_form_sg3_prs="spins")),
        (SIT, LexiconEntry("sit", VERB, verb_form_sg3_prs="sits")),
        (DRINK, LexiconEntry("drink", VERB, verb_form_sg3_prs="drinks")),
        (FALL, LexiconEntry("fall", VERB, verb_form_sg3_prs="falls")),
        (
            THROW,
            LexiconEntry(
                "throw",
                VERB,
                verb_form_sg3_prs="throws",
                properties=[ALLOWS_DITRANSITIVE],
            ),
        ),
        (
            PASS,
            LexiconEntry(
                "pass", VERB, verb_form_sg3_prs="passes", properties=[ALLOWS_DITRANSITIVE]
            ),
        ),
        (MOVE, LexiconEntry("move", VERB, verb_form_sg3_prs="moves")),
        (WALK, LexiconEntry("walk", VERB, verb_form_sg3_prs="walks")),
        (JUMP, LexiconEntry("jump", VERB, verb_form_sg3_prs="jumps")),
        (HAS, LexiconEntry("have", VERB, verb_form_sg3_prs="has")),
        (ROLL, LexiconEntry("roll", VERB, verb_form_sg3_prs="rolls")),
        (FLY, LexiconEntry("fly", VERB, verb_form_sg3_prs="flies")),
        (RED, LexiconEntry("red", ADJECTIVE)),
        (BLUE, LexiconEntry("blue", ADJECTIVE)),
        (GREEN, LexiconEntry("green", ADJECTIVE)),
        (BLACK, LexiconEntry("black", ADJECTIVE)),
        (WHITE, LexiconEntry("white", ADJECTIVE)),
        (TRANSPARENT, LexiconEntry("transparent", ADJECTIVE)),
        (LIGHT_BROWN, LexiconEntry("light brown", ADJECTIVE)),
        (DARK_BROWN, LexiconEntry("dark brown", ADJECTIVE)),
    ),
)
