"""This file contains test cases for the Chinese Language Generator"""


from typing import Tuple
import pytest
from more_itertools import only
from adam.axes import (
    HorizontalAxisOfObject,
    FacingAddresseeAxis,
    AxesInfo,
    GRAVITATIONAL_AXIS_FUNCTION,
)
from adam.language_specific.chinese.chinese_language_generator import (
    PREFER_DITRANSITIVE,
    SimpleRuleBasedChineseLanguageGenerator,
    USE_ADVERBIAL_PATH_MODIFIER,
    IGNORE_HAS_AS_VERB,
    ATTRIBUTES_AS_X_IS_Y,
    IGNORE_GOAL,
    USE_VERTICAL_MODIFIERS,
    USE_ABOVE_BELOW,
    USE_NEAR,
)
from adam.language_specific.chinese.chinese_phase_1_lexicon import (
    GAILA_PHASE_1_CHINESE_LEXICON,
)
from adam.axes import GRAVITATIONAL_DOWN_TO_UP_AXIS
from adam.ontology import IN_REGION, IS_SPEAKER, IS_ADDRESSEE
from adam.ontology.during import DuringAction
from adam.ontology.phase1_ontology import (
    AGENT,
    SEMANTIC_ROLE,
    HARD_FORCE,
    SOFT_FORCE,
    COME,
    BABY,
    BALL,
    BIRD,
    BOX,
    CHAIR,
    COOKIE,
    CUP,
    PASS,
    DAD,
    DRINK,
    DRINK_CONTAINER_AUX,
    EAT,
    FALL,
    FLY,
    GAILA_PHASE_1_ONTOLOGY,
    GIVE,
    PUSH_SURFACE_AUX,
    PUSH_GOAL,
    GOAL,
    GREEN,
    GROUND,
    HAS,
    JUICE,
    MOM,
    PATIENT,
    PUSH,
    PUT,
    ROLL,
    SIT,
    TABLE,
    THEME,
    THROW,
    WATER,
    DOOR,
    HAT,
    on,
    strictly_above,
    strictly_over,
    strictly_under,
    strictly_below,
    JUMP,
    JUMP_INITIAL_SUPPORTER_AUX,
    DOG,
    HOLLOW,
    GO,
    BOOK,
    LEARNER,
    near,
    inside,
    TAKE,
    CAR,
    ROLL_SURFACE_AUXILIARY,
    TRUCK,
    HOUSE,
    RED,
    WATERMELON,
    MOVE,
    FAST,
    SLOW,
    WALK,
    WALK_SURFACE_AUXILIARY,
    bigger_than,
    far,
    SPATIAL_RELATION,
)
from adam.ontology.phase1_spatial_relations import (
    AWAY_FROM,
    FROM,
    TOWARD,
    DISTAL,
    EXTERIOR_BUT_IN_CONTACT,
    GRAVITATIONAL_DOWN,
    GRAVITATIONAL_UP,
    INTERIOR,
    Region,
    SpatialPath,
    Direction,
    PROXIMAL,
    VIA,
)
from adam.random_utils import FixedIndexChooser
from adam.relation import Relation, flatten_relations
from adam.situation import Action, SituationObject
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam_test_utils import situation_object
from tests.sample_situations import make_bird_flies_over_a_house
from tests.situation.situation_test import make_mom_put_ball_on_table

_SIMPLE_GENERATOR = SimpleRuleBasedChineseLanguageGenerator(
    ontology_lexicon=GAILA_PHASE_1_CHINESE_LEXICON
)

"""GLOBAL UTILITY FUNCTIONS"""

# function to generate tokens from a given situation
def generated_tokens(situation):
    return only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence()


# generates region-as-goal situations for go/come (qu/lai)
def region_as_goal_situation(
    goal: Region[SituationObject], goal_object: SituationObject
) -> HighLevelSemanticsSituation:
    agent = situation_object(DOG)
    learner = situation_object(LEARNER, properties=[IS_ADDRESSEE])

    return HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[agent, goal_object],
        other_objects=[learner],
        actions=[Action(GO, argument_roles_to_fillers=[(AGENT, agent), (GOAL, goal)])],
        axis_info=AxesInfo(
            addressee=learner,
            axes_facing=[
                (
                    learner,
                    # TODO: fix this hack
                    HorizontalAxisOfObject(obj, index=1).to_concrete_axis(  # type: ignore
                        None
                    ),
                )
                for obj in [agent, goal_object, learner]
                if obj.axes
            ],
        ),
        after_action_relations=[near(agent, goal_object)],
    )


# generates region-as-goal situations for go/come (qu/lai)
def region_as_goal_situation_come(
    goal: Region[SituationObject], goal_object: SituationObject
) -> HighLevelSemanticsSituation:
    agent = situation_object(DOG)
    learner = situation_object(LEARNER, properties=[IS_ADDRESSEE])

    return HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[agent, goal_object],
        other_objects=[learner],
        actions=[Action(COME, argument_roles_to_fillers=[(AGENT, agent), (GOAL, goal)])],
        axis_info=AxesInfo(
            addressee=learner,
            axes_facing=[
                (
                    learner,
                    # TODO: fix this hack
                    HorizontalAxisOfObject(obj, index=1).to_concrete_axis(  # type: ignore
                        None
                    ),
                )
                for obj in [agent, goal_object, learner]
                if obj.axes
            ],
        ),
        after_action_relations=[near(agent, goal_object)],
    )


""" BASIC NOUN PHRASE TESTS -- CHECKED BY NATIVE SPEAKER"""

# just a single common noun
def test_common_noun():
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[situation_object(BALL)]
    )
    assert generated_tokens(situation) == ("chyou2",)


# a single mass noun. The distinction isn't nearly as salient in Chinese
def test_mass_noun():
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[situation_object(WATER)]
    )
    assert generated_tokens(situation) == ("shwei3",)


# a single proper noun
def test_proper_noun():
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[situation_object(DAD)]
    )
    assert generated_tokens(situation) == ("ba4 ba4",)


# get the first person pronoun
def test_wo():
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[situation_object(DAD, properties=[IS_SPEAKER])],
    )
    assert generated_tokens(situation) == ("wo3",)


# get the second person pronoun
def test_ni():
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[situation_object(DAD, properties=[IS_ADDRESSEE])],
    )
    assert generated_tokens(situation) == ("ni3",)


"""NOUN PHRASES WITH ADJECTIVAL MODIFIERS -- CHECKED BY NATIVE SPEAKER"""

# basic adjective+noun
def test_green_ball():
    ball = situation_object(BALL, properties=[GREEN])
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[ball]
    )
    assert generated_tokens(situation) == ("lyu4 se4", "chyou2")


# possession for NP's: first person
def test_my_green_ball():
    ball = situation_object(BALL, properties=[GREEN])
    dad = situation_object(DAD, properties=[IS_SPEAKER])
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[ball, dad],
        always_relations=[Relation(HAS, dad, ball)],
        syntax_hints=[IGNORE_HAS_AS_VERB],
    )
    assert generated_tokens(situation) == ("wo3 de", "lyu4 se4", "chyou2")


# possession for NP's: second person
def test_your_green_ball():
    ball = situation_object(BALL, properties=[GREEN])
    dad = situation_object(DAD, properties=[IS_ADDRESSEE])
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[ball, dad],
        always_relations=[Relation(HAS, dad, ball)],
        syntax_hints=[IGNORE_HAS_AS_VERB],
    )
    assert generated_tokens(situation) == ("ni3 de", "lyu4 se4", "chyou2")


# possession for NP's: third person
def test_babade_green_ball():
    ball = situation_object(BALL, properties=[GREEN])
    dad = situation_object(DAD)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[ball, dad],
        always_relations=[Relation(HAS, dad, ball)],
        syntax_hints=[IGNORE_HAS_AS_VERB],
    )
    assert generated_tokens(situation) == ("ba4 ba4", "de", "lyu4 se4", "chyou2")


"""COUNTING NOUN PHRASE TESTS (WITH CLASSIFIERS) -- CHECKED BY NATIVE SPEAKER"""


# a single object -- note that we don't use yi clf noun yet but this could be implemented later on
def test_single_item():
    dog = situation_object(DOG)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[dog]
    )
    assert generated_tokens(situation) == ("gou3",)


# two objects, which can be counted distinctly
def test_two_items():
    dog1 = situation_object(DOG, debug_handle="dog1")
    dog2 = situation_object(DOG, debug_handle="dog2")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[dog1, dog2]
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("lyang3", "jr3", "gou3")


# many objects
def test_many_items():
    ball1 = situation_object(BALL, debug_handle="ball1")
    ball2 = situation_object(BALL, debug_handle="ball2")
    ball3 = situation_object(BALL, debug_handle="ball3")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[ball1, ball2, ball3]
    )
    assert generated_tokens(situation) == ("hen3 dwo1", "chyou2")


"""ADDITIONAL CLASSIFIER TESTS -- CHECKED BY NATIVE SPEAKER"""

# check the automobile classifier, lyang4
def test_car_truck_clf():
    car1 = situation_object(CAR, debug_handle="car1")
    car2 = situation_object(CAR, debug_handle="car2")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[car1, car2]
    )
    assert generated_tokens(situation) == ("lyang3", "lyang4", "chi4 che1")


# check the default classifier
def test_default_clf():
    box1 = situation_object(BOX, debug_handle="box1")
    box2 = situation_object(BOX, debug_handle="box2")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[box1, box2]
    )
    assert generated_tokens(situation) == ("lyang3", "ge4", "syang1 dz")


# test the classifier for flat surfaces
def test_table_classifier():
    table1 = situation_object(TABLE, debug_handle="table1")
    table2 = situation_object(TABLE, debug_handle="table2")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[table1, table2]
    )
    assert generated_tokens(situation) == ("lyang3", "jang1", "jwo1 dz")


# test the classifier for books and other bound objects
def test_book_classifier():
    book1 = situation_object(BOOK, debug_handle="book1")
    book2 = situation_object(BOOK, debug_handle="book2")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[book1, book2]
    )
    assert generated_tokens(situation) == ("lyang3", "ben3", "shu1")


# test the house/room classifier
def test_house_classifier():
    house1 = situation_object(HOUSE, debug_handle="house1")
    house2 = situation_object(HOUSE, debug_handle="house2")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[house1, house2]
    )
    assert generated_tokens(situation) == ("lyang3", "jyan1", "wu1")


# test the classifier for cups of liquid
def test_cups_of_liquid():
    water1 = situation_object(WATER, debug_handle="water1")
    water2 = situation_object(WATER, debug_handle="water2")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[water1, water2]
    )
    assert generated_tokens(situation) == ("lyang3", "bei1", "shwei3")


# test the classifier for chairs
def test_chair_classifiers():
    chair1 = situation_object(CHAIR, debug_handle="chair1")
    chair2 = situation_object(CHAIR, debug_handle="chair2")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[chair1, chair2]
    )
    assert generated_tokens(situation) == ("lyang3", "ba3", "yi3 dz")


# test the classifier for doors
def test_door_classifier():
    door1 = situation_object(DOOR, debug_handle="door1")
    door2 = situation_object(DOOR, debug_handle="door2")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[door1, door2]
    )
    assert generated_tokens(situation) == ("lyang3", "shan4", "men2")


# test the classifier for hats
def test_hat_classifier():
    hat1 = situation_object(HAT, debug_handle="hat1")
    hat2 = situation_object(HAT, debug_handle="hat2")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[hat1, hat2]
    )
    assert generated_tokens(situation) == ("lyang3", "ding3", "mau4 dz")


# test the classifier for cookies
def test_cookie_classifier():
    cookie1 = situation_object(COOKIE, debug_handle="cookie1")
    cookie2 = situation_object(COOKIE, debug_handle="cookie2")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[cookie1, cookie2]
    )
    assert generated_tokens(situation) == ("lyang3", "kwai4", "chyu1 chi2 bing3")


# tests the classifiers with a colour modifier
def test_colour_classifier():
    ball1 = situation_object(BALL, debug_handle="ball1", properties=[RED])
    ball2 = situation_object(BALL, debug_handle="ball2", properties=[RED])
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[ball1, ball2]
    )
    assert generated_tokens(situation) == ("lyang3", "ge4", "hung2 se4", "chyou2")


"""SALIENT VS. NOT SALIENT OBJECTS -- CHECKED BY NATIVE SPEAKER"""

# two trucks, only one is salient
def test_one_salient_object():
    truck1 = situation_object(TRUCK, debug_handle="truck1")
    truck2 = situation_object(TRUCK, debug_handle="truck2")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[truck1], other_objects=[truck2]
    )
    assert generated_tokens(situation) == ("ka3 che1",)


# many objects, only two are salient
def test_two_salient_objects():
    dog1 = situation_object(DOG, debug_handle="dog1")
    dog2 = situation_object(DOG, debug_handle="dog2")
    dog3 = situation_object(DOG, debug_handle="dog3")
    dog4 = situation_object(DOG, debug_handle="dog4")
    truck1 = situation_object(TRUCK, debug_handle="truck1")
    truck2 = situation_object(TRUCK, debug_handle="truck2")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dog1, dog2],
        other_objects=[dog3, truck1, truck2, dog4],
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("lyang3", "jr3", "gou3")


# many objects, only two are salient
def test_many_salient_objects():
    dog1 = situation_object(DOG, debug_handle="dog1")
    dog2 = situation_object(DOG, debug_handle="dog2")
    dog3 = situation_object(DOG, debug_handle="dog3")
    dog4 = situation_object(DOG, debug_handle="dog4")
    truck1 = situation_object(TRUCK, debug_handle="truck1")
    truck2 = situation_object(TRUCK, debug_handle="truck2")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dog1, dog2, dog4],
        other_objects=[dog3, truck1, truck2],
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("hen3 dwo1", "gou3")


"""NOUN PHRASES WITH LOCALISER NOMINAL MODIFIERS -- CHECKED BY NATIVE SPEAKER"""

# test two inanimate objects: the table on the ground
def test_noun_with_spatial_modifier():
    table = situation_object(TABLE)
    ground = situation_object(GROUND)

    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[table, ground],
        always_relations=[on(table, ground)],
    )
    assert generated_tokens(situation) == (
        "dzai4",
        "di4 myan4",
        "shang4",
        "de",
        "jwo1 dz",
    )


# tests mum being next to an object, a relation that is represented with a localiser phrase
def test_two_objects_with_mum_no_extra():
    bird1 = situation_object(BIRD, debug_handle="bird1")
    mum = situation_object(MOM, debug_handle="mum")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[bird1, mum],
        always_relations=[
            Relation(
                IN_REGION,
                mum,
                Region(
                    bird1,
                    distance=PROXIMAL,
                    direction=Direction(
                        positive=True,
                        relative_to_axis=HorizontalAxisOfObject(bird1, index=0),
                    ),
                ),
            )
        ],
    )
    assert generated_tokens(situation) == (
        "dzai4",
        "nyau3",
        "pang2 byan1",
        "de",
        "ma1 ma1",
    )


# tests mum being next to an object, a relation that is represented with a localiser phrase
def test_two_objects_with_mum():
    bird1 = situation_object(BIRD, debug_handle="bird1")
    bird2 = situation_object(BIRD, debug_handle="bird2")
    mum = situation_object(MOM, debug_handle="mum")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[bird1, mum],
        other_objects=[bird2],
        always_relations=[
            Relation(
                IN_REGION,
                mum,
                Region(
                    bird1,
                    distance=PROXIMAL,
                    direction=Direction(
                        positive=True,
                        relative_to_axis=HorizontalAxisOfObject(bird1, index=0),
                    ),
                ),
            )
        ],
    )
    assert generated_tokens(situation) == (
        "dzai4",
        "nyau3",
        "pang2 byan1",
        "de",
        "ma1 ma1",
    )


# tests mum being under a bird
def test_mum_under_object():
    bird1 = situation_object(BIRD, debug_handle="bird1")
    bird2 = situation_object(BIRD, debug_handle="bird2")
    mum = situation_object(MOM, debug_handle="mum")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[bird1, mum],
        other_objects=[bird2],
        always_relations=[
            Relation(
                IN_REGION,
                mum,
                Region(bird1, distance=DISTAL, direction=GRAVITATIONAL_DOWN),
            )
        ],
    )
    assert generated_tokens(situation) == (
        "dzai4",
        "nyau3",
        "sya4 myan4",
        "de",
        "ma1 ma1",
    )


# tests mum being above an object
def test_mum_above_object():
    bird1 = situation_object(BIRD, debug_handle="bird1")
    bird2 = situation_object(BIRD, debug_handle="bird2")
    mum = situation_object(MOM, debug_handle="mum")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[bird1, mum],
        other_objects=[bird2],
        always_relations=[
            Relation(
                IN_REGION, mum, Region(bird1, distance=DISTAL, direction=GRAVITATIONAL_UP)
            )
        ],
    )
    assert generated_tokens(situation) == (
        "dzai4",
        "nyau3",
        "shang4 myan4",
        "de",
        "ma1 ma1",
    )


# tests an object beside another object
def test_object_beside_object():
    # HACK FOR AXES - See https://github.com/isi-vista/adam/issues/316
    ball = situation_object(BALL)
    table = situation_object(TABLE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[ball, table],
        always_relations=[
            Relation(
                IN_REGION,
                ball,
                Region(
                    table,
                    distance=PROXIMAL,
                    direction=Direction(
                        positive=True,
                        relative_to_axis=HorizontalAxisOfObject(table, index=0),
                    ),
                ),
            )
        ],
    )
    assert generated_tokens(situation) == (
        "dzai4",
        "jwo1 dz",
        "pang2 byan1",
        "de",
        "chyou2",
    )


# in front and behind tests copied from English files
def test_object_behind_in_front_object():
    # HACK FOR AXES - See https://github.com/isi-vista/adam/issues/316
    box = situation_object(BOX)
    table = situation_object(TABLE)
    speaker = situation_object(MOM, properties=[IS_SPEAKER])
    addressee = situation_object(DAD, properties=[IS_ADDRESSEE])

    front_situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[box, table],
        other_objects=[speaker, addressee],
        always_relations=[
            Relation(
                IN_REGION,
                box,
                Region(
                    table,
                    distance=PROXIMAL,
                    direction=Direction(
                        positive=True, relative_to_axis=FacingAddresseeAxis(table)
                    ),
                ),
            )
        ],
        axis_info=AxesInfo(
            addressee=addressee,
            axes_facing=[
                (
                    addressee,
                    # TODO: fix this hack
                    HorizontalAxisOfObject(obj, index=1).to_concrete_axis(  # type: ignore
                        None
                    ),
                )
                for obj in [box, table, speaker, addressee]
                if obj.axes
            ],
        ),
    )
    assert generated_tokens(front_situation) == (
        "dzai4",
        "jwo1 dz",
        "chyan2 myan4",
        "de",
        "syang1 dz",
    )

    behind_situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[box, table],
        other_objects=[speaker, addressee],
        always_relations=[
            Relation(
                IN_REGION,
                box,
                Region(
                    table,
                    distance=PROXIMAL,
                    direction=Direction(
                        positive=False, relative_to_axis=FacingAddresseeAxis(table)
                    ),
                ),
            )
        ],
        axis_info=AxesInfo(
            addressee=addressee,
            axes_facing=[
                (
                    addressee,
                    # TODO: fix this hack
                    HorizontalAxisOfObject(obj, index=1).to_concrete_axis(  # type: ignore
                        None
                    ),
                )
                for obj in [box, table, speaker, addressee]
                if obj.axes
            ],
        ),
    )
    assert generated_tokens(behind_situation) == (
        "dzai4",
        "jwo1 dz",
        "hou4 myan4",
        "de",
        "syang1 dz",
    )


"""BASIC VP TESTING: SV, SVO, and SVIO -- CHECKED BY NATIVE SPEAKER"""


# basic intransitive verb testing
def test_falling():
    ball = situation_object(BALL)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[ball],
        actions=[Action(FALL, argument_roles_to_fillers=[(THEME, ball)])],
    )
    assert generated_tokens(situation) == ("chyou2", "dye2 dau3")


# test the simple subject-verb phrase "mum eats"
def test_simple_subject_verb():
    mum = situation_object(MOM, debug_handle="mum")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mum],
        actions=[Action(action_type=EAT, argument_roles_to_fillers=[(AGENT, mum)])],
    )
    assert generated_tokens(situation) == ("ma1 ma1", "chr1")


# test more complex SVO phrase "mum eats a cookie"
def test_simple_SVO():
    mum = situation_object(MOM, debug_handle="mum")
    cookie = situation_object(COOKIE, debug_handle="cookie")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mum, cookie],
        actions=[
            Action(
                action_type=EAT, argument_roles_to_fillers=[(AGENT, mum), (THEME, cookie)]
            )
        ],
    )
    assert generated_tokens(situation) == ("ma1 ma1", "chr1", "chyu1 chi2 bing3")


# test SVIO transfer of possession
def test_simple_SVIO_transfer():
    mum = situation_object(MOM, debug_handle="mum_subject")
    baby = situation_object(BABY, debug_handle="babyIO")
    cookie = situation_object(COOKIE, debug_handle="cookieDO")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mum, baby, cookie],
        actions=[
            Action(
                action_type=GIVE,
                argument_roles_to_fillers=[(AGENT, mum), (GOAL, baby), (THEME, cookie)],
            )
        ],
        syntax_hints=[PREFER_DITRANSITIVE],
    )
    assert generated_tokens(situation) == (
        "ma1 ma1",
        "gei3",
        "bau3 bau3",
        "chyu1 chi2 bing3",
    )


"""VP's WITH PERSONAL PRONOUNS -- CHECKED BY NATIVE SPEAKER"""

# test the simple subject-verb phrase "mum eats" with first person
def test_simple_subject_verb_me():
    mum = situation_object(MOM, debug_handle="mum", properties=[IS_SPEAKER])
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mum],
        actions=[Action(action_type=EAT, argument_roles_to_fillers=[(AGENT, mum)])],
    )
    assert generated_tokens(situation) == ("wo3", "chr1")


# test the simple subject-verb phrase "mum eats" with second person
def test_simple_subject_verb_you():
    mum = situation_object(MOM, debug_handle="mum", properties=[IS_ADDRESSEE])
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mum],
        actions=[Action(action_type=EAT, argument_roles_to_fillers=[(AGENT, mum)])],
    )
    assert generated_tokens(situation) == ("ni3", "chr1")


# test SVIO transfer of possession with personal pronouns
def test_simple_SVIO_transfer_with_personal_pronouns():
    mum = situation_object(MOM, debug_handle="mum_subject", properties=[IS_SPEAKER])
    baby = situation_object(BABY, debug_handle="babyIO", properties=[IS_ADDRESSEE])
    cookie = situation_object(COOKIE, debug_handle="cookieDO")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mum, baby, cookie],
        actions=[
            Action(
                action_type=GIVE,
                argument_roles_to_fillers=[(AGENT, mum), (GOAL, baby), (THEME, cookie)],
            )
        ],
        syntax_hints=[PREFER_DITRANSITIVE],
    )
    assert generated_tokens(situation) == ("wo3", "gei3", "ni3", "chyu1 chi2 bing3")


# test SVIO transfer of possession with personal pronouns
def test_simple_SVIO_transfer_with_personal_pronouns_and_ba():
    mum = situation_object(MOM, debug_handle="mum_subject", properties=[IS_SPEAKER])
    baby = situation_object(BABY, debug_handle="babyIO", properties=[IS_ADDRESSEE])
    cookie = situation_object(COOKIE, debug_handle="cookieDO")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mum, baby, cookie],
        actions=[
            Action(
                action_type=GIVE,
                argument_roles_to_fillers=[(AGENT, mum), (GOAL, baby), (THEME, cookie)],
            )
        ],
    )
    assert generated_tokens(situation) == (
        "wo3",
        "ba3",
        "chyu1 chi2 bing3",
        "gei3",
        "ni3",
    )


# test SVO with action/movement verb
def test_simple_SVO_movement():
    dad = situation_object(DAD)
    chair = situation_object(CHAIR)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dad, chair],
        actions=[
            Action(
                action_type=PUSH, argument_roles_to_fillers=[(AGENT, dad), (THEME, chair)]
            )
        ],
    )
    assert generated_tokens(situation) == ("ba4 ba4", "twei1", "yi3 dz")


# SVIO with pronouns
def test_you_give_me_a_cookie():
    you = situation_object(DAD, properties=[IS_ADDRESSEE])
    baby = situation_object(BABY, properties=[IS_SPEAKER])
    cookie = situation_object(COOKIE)
    situation_ditransitive = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[you, baby, cookie],
        actions=[
            Action(
                GIVE,
                argument_roles_to_fillers=[(AGENT, you), (GOAL, baby), (THEME, cookie)],
            )
        ],
        syntax_hints=[PREFER_DITRANSITIVE],
    )
    assert generated_tokens(situation_ditransitive) == (
        "ni3",
        "gei3",
        "wo3",
        "chyu1 chi2 bing3",
    )


"""VP's WITH LOCALIZERS AND VARIOUS SPEAKERS -- CHECKED BY NATIVE SPEAKER"""
# use zai by default and dao with after-action relations based on https://github.com/isi-vista/adam/issues/796
# a list of verbs that currently don't accept goals is at https://github.com/isi-vista/adam/issues/582

# this situation doesn't have any after-action relations so it uses zai, which is valid
def test_mom_put_a_ball_on_a_table_zai():
    mum = situation_object(MOM)
    ball = situation_object(BALL)
    table = situation_object(TABLE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mum, ball, table],
        actions=[
            Action(
                action_type=PUT,
                argument_roles_to_fillers=[
                    (AGENT, mum),
                    (THEME, ball),
                    (
                        GOAL,
                        Region(
                            reference_object=table,
                            distance=EXTERIOR_BUT_IN_CONTACT,
                            direction=GRAVITATIONAL_UP,
                        ),
                    ),
                ],
            )
        ],
    )
    assert generated_tokens(situation) == (
        "ma1 ma1",
        "ba3",
        "chyou2",
        "fang4",
        "dzai4",
        "jwo1 dz",
        "shang4",
    )


# this situation specifies the after_action_relations and so dao should be used
# since there was an explicit change in location
def test_mom_put_a_ball_on_a_table_dao():
    mum = situation_object(MOM)
    ball = situation_object(BALL)
    table = situation_object(TABLE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mum, ball, table],
        actions=[
            Action(
                action_type=PUT,
                argument_roles_to_fillers=[
                    (AGENT, mum),
                    (THEME, ball),
                    (
                        GOAL,
                        Region(
                            reference_object=table,
                            distance=EXTERIOR_BUT_IN_CONTACT,
                            direction=GRAVITATIONAL_UP,
                        ),
                    ),
                ],
            )
        ],
        after_action_relations=[on(ball, table)],
    )
    assert generated_tokens(situation) == (
        "ma1 ma1",
        "ba3",
        "chyou2",
        "fang4",
        "dau4",
        "jwo1 dz",
        "shang4",
    )


# this situation doesn't have any after-action relations so it uses zai, which is valid
# the mum is the speaker here, so we expect the first person 'wo'
def test_I_put_a_ball_on_a_table_zai():
    mum = situation_object(MOM, properties=[IS_SPEAKER])
    ball = situation_object(BALL)
    table = situation_object(TABLE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mum, ball, table],
        actions=[
            Action(
                action_type=PUT,
                argument_roles_to_fillers=[
                    (AGENT, mum),
                    (THEME, ball),
                    (
                        GOAL,
                        Region(
                            reference_object=table,
                            distance=EXTERIOR_BUT_IN_CONTACT,
                            direction=GRAVITATIONAL_UP,
                        ),
                    ),
                ],
            )
        ],
    )
    assert generated_tokens(situation) == (
        "wo3",
        "ba3",
        "chyou2",
        "fang4",
        "dzai4",
        "jwo1 dz",
        "shang4",
    )


# the speaker is putting the ball on the table here, using dao since we have after-action relations
def test_i_put_a_ball_on_a_table_dao():
    mum = situation_object(MOM, properties=[IS_SPEAKER])
    ball = situation_object(BALL)
    table = situation_object(TABLE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mum, ball, table],
        actions=[
            Action(
                action_type=PUT,
                argument_roles_to_fillers=[
                    (AGENT, mum),
                    (THEME, ball),
                    (
                        GOAL,
                        Region(
                            reference_object=table,
                            distance=EXTERIOR_BUT_IN_CONTACT,
                            direction=GRAVITATIONAL_UP,
                        ),
                    ),
                ],
            )
        ],
        after_action_relations=[on(ball, table)],
    )
    assert generated_tokens(situation) == (
        "wo3",
        "ba3",
        "chyou2",
        "fang4",
        "dau4",
        "jwo1 dz",
        "shang4",
    )


# this situation doesn't have any after-action relations so it uses zai, which is valid
# the mum is the speaker here, so we expect the first person 'wo'
def test_you_put_a_ball_on_a_table_zai():
    mum = situation_object(MOM, properties=[IS_ADDRESSEE])
    ball = situation_object(BALL)
    table = situation_object(TABLE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mum, ball, table],
        actions=[
            Action(
                action_type=PUT,
                argument_roles_to_fillers=[
                    (AGENT, mum),
                    (THEME, ball),
                    (
                        GOAL,
                        Region(
                            reference_object=table,
                            distance=EXTERIOR_BUT_IN_CONTACT,
                            direction=GRAVITATIONAL_UP,
                        ),
                    ),
                ],
            )
        ],
    )
    assert generated_tokens(situation) == (
        "ni3",
        "ba3",
        "chyou2",
        "fang4",
        "dzai4",
        "jwo1 dz",
        "shang4",
    )


# the speaker is putting the ball on the table here, using dao since we have after-action relations
def test_you_put_a_ball_on_a_table_dao():
    mum = situation_object(MOM, properties=[IS_ADDRESSEE])
    ball = situation_object(BALL)
    table = situation_object(TABLE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mum, ball, table],
        actions=[
            Action(
                action_type=PUT,
                argument_roles_to_fillers=[
                    (AGENT, mum),
                    (THEME, ball),
                    (
                        GOAL,
                        Region(
                            reference_object=table,
                            distance=EXTERIOR_BUT_IN_CONTACT,
                            direction=GRAVITATIONAL_UP,
                        ),
                    ),
                ],
            )
        ],
        after_action_relations=[on(ball, table)],
    )
    assert generated_tokens(situation) == (
        "ni3",
        "ba3",
        "chyou2",
        "fang4",
        "dau4",
        "jwo1 dz",
        "shang4",
    )


# an additional test for localizers using a different localizer
def test_dad_put_a_cookie_in_a_box_zai():
    dad = situation_object(DAD)
    cookie = situation_object(COOKIE)
    box = situation_object(BOX)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dad, cookie, box],
        actions=[
            Action(
                action_type=PUT,
                argument_roles_to_fillers=[
                    (AGENT, dad),
                    (THEME, cookie),
                    (GOAL, Region(reference_object=box, distance=INTERIOR)),
                ],
            )
        ],
    )
    assert generated_tokens(situation) == (
        "ba4 ba4",
        "ba3",
        "chyu1 chi2 bing3",
        "fang4",
        "dzai4",
        "syang1 dz",
        "li3",
    )


# an additional test for localizers using a different localizer. This one specifies
# a change in location, so we use dao instead of zai to indicate this change
def test_dad_put_a_cookie_in_a_box_dao():
    dad = situation_object(DAD)
    cookie = situation_object(COOKIE)
    box = situation_object(BOX)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dad, cookie, box],
        actions=[
            Action(
                action_type=PUT,
                argument_roles_to_fillers=[
                    (AGENT, dad),
                    (THEME, cookie),
                    (GOAL, Region(reference_object=box, distance=INTERIOR)),
                ],
            )
        ],
        after_action_relations=[on(cookie, box)],
    )
    assert generated_tokens(situation) == (
        "ba4 ba4",
        "ba3",
        "chyu1 chi2 bing3",
        "fang4",
        "dau4",
        "syang1 dz",
        "li3",
    )


# test with another localiser and first person
def test_i_put_cookie_in_box_zai():
    dad = situation_object(DAD, properties=[IS_SPEAKER])
    cookie = situation_object(COOKIE)
    box = situation_object(BOX)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dad, cookie, box],
        actions=[
            Action(
                action_type=PUT,
                argument_roles_to_fillers=[
                    (AGENT, dad),
                    (THEME, cookie),
                    (GOAL, Region(reference_object=box, distance=INTERIOR)),
                ],
            )
        ],
    )
    assert generated_tokens(situation) == (
        "wo3",
        "ba3",
        "chyu1 chi2 bing3",
        "fang4",
        "dzai4",
        "syang1 dz",
        "li3",
    )


# test with another localiser and second person
def test_you_put_cookie_in_box_zai():
    dad = situation_object(DAD, properties=[IS_ADDRESSEE])
    cookie = situation_object(COOKIE)
    box = situation_object(BOX)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dad, cookie, box],
        actions=[
            Action(
                action_type=PUT,
                argument_roles_to_fillers=[
                    (AGENT, dad),
                    (THEME, cookie),
                    (GOAL, Region(reference_object=box, distance=INTERIOR)),
                ],
            )
        ],
    )
    assert generated_tokens(situation) == (
        "ni3",
        "ba3",
        "chyu1 chi2 bing3",
        "fang4",
        "dzai4",
        "syang1 dz",
        "li3",
    )


# another meaning to test with localisers
def test_take_to_car():
    baby = situation_object(BABY)
    ball = situation_object(BALL)
    car = situation_object(CAR)

    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[baby, ball, car],
        actions=[
            Action(
                action_type=TAKE, argument_roles_to_fillers=[(AGENT, baby), (THEME, ball)]
            )
        ],
        after_action_relations=[near(ball, car)],
    )

    assert generated_tokens(situation) == (
        "bau3 bau3",
        "ba3",
        "chyou2",
        "na2",
        "dau4",
        "chi4 che1",
        "shang4",
    )


"""TESTS POSSESSIVE: WO DE AND NI DE -- CHECKED BY NATIVE SPEAKER"""

# tests the use of the first person possessive, 'wo de'
def test_i_put_cookie_in_my_box():
    dad = situation_object(DAD, properties=[IS_SPEAKER])
    cookie = situation_object(COOKIE)
    box = situation_object(BOX)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dad, cookie, box],
        always_relations=[Relation(HAS, dad, box)],
        actions=[
            Action(
                action_type=PUT,
                argument_roles_to_fillers=[
                    (AGENT, dad),
                    (THEME, cookie),
                    (GOAL, Region(reference_object=box, distance=INTERIOR)),
                ],
            )
        ],
    )
    assert generated_tokens(situation) == (
        "wo3",
        "ba3",
        "chyu1 chi2 bing3",
        "fang4",
        "dzai4",
        "wo3 de",
        "syang1 dz",
        "li3",
    )


# tests the use of the first person possessive, 'wo de'
def test_i_put_cookie_in_my_box_dao():
    dad = situation_object(DAD, properties=[IS_SPEAKER])
    cookie = situation_object(COOKIE)
    box = situation_object(BOX)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dad, cookie, box],
        always_relations=[Relation(HAS, dad, box)],
        actions=[
            Action(
                action_type=PUT,
                argument_roles_to_fillers=[
                    (AGENT, dad),
                    (THEME, cookie),
                    (GOAL, Region(reference_object=box, distance=INTERIOR)),
                ],
            )
        ],
        after_action_relations=[on(cookie, box)],
    )
    assert generated_tokens(situation) == (
        "wo3",
        "ba3",
        "chyu1 chi2 bing3",
        "fang4",
        "dau4",
        "wo3 de",
        "syang1 dz",
        "li3",
    )


# this is where another person in the interaction has a box, but we don't note this in our langauge
def test_third_person_cookie_in_his_box():
    dad = situation_object(DAD)
    cookie = situation_object(COOKIE)
    box = situation_object(BOX)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dad, cookie, box],
        always_relations=[Relation(HAS, dad, box)],
        actions=[
            Action(
                action_type=PUT,
                argument_roles_to_fillers=[
                    (AGENT, dad),
                    (THEME, cookie),
                    (GOAL, Region(reference_object=box, distance=INTERIOR)),
                ],
            )
        ],
    )
    assert generated_tokens(situation) == (
        "ba4 ba4",
        "ba3",
        "chyu1 chi2 bing3",
        "fang4",
        "dzai4",
        "syang1 dz",
        "li3",
    )


# tests the use of the second person possessive, 'ni de'
def test_you_put_cookie_in_your_box():
    dad = situation_object(DAD, properties=[IS_ADDRESSEE])
    cookie = situation_object(COOKIE)
    box = situation_object(BOX)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dad, cookie, box],
        always_relations=[Relation(HAS, dad, box)],
        actions=[
            Action(
                action_type=PUT,
                argument_roles_to_fillers=[
                    (AGENT, dad),
                    (THEME, cookie),
                    (GOAL, Region(reference_object=box, distance=INTERIOR)),
                ],
            )
        ],
    )
    assert generated_tokens(situation) == (
        "ni3",
        "ba3",
        "chyu1 chi2 bing3",
        "fang4",
        "dzai4",
        "ni3 de",
        "syang1 dz",
        "li3",
    )


# tests use of first person possessive 'wo de' when the speaker isn't the agent
def test_speaker_owner_of_box():
    dad = situation_object(DAD)
    cookie = situation_object(COOKIE)
    box = situation_object(BOX)
    mum = situation_object(MOM, properties=[IS_SPEAKER])
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dad, cookie, box, mum],
        always_relations=[Relation(HAS, mum, box)],
        actions=[
            Action(
                action_type=PUT,
                argument_roles_to_fillers=[
                    (AGENT, dad),
                    (THEME, cookie),
                    (GOAL, Region(reference_object=box, distance=INTERIOR)),
                ],
            )
        ],
    )
    assert generated_tokens(situation) == (
        "ba4 ba4",
        "ba3",
        "chyu1 chi2 bing3",
        "fang4",
        "dzai4",
        "wo3 de",
        "syang1 dz",
        "li3",
    )


# tests use of first person possessive 'wo de' when the speaker isn't the agent
def test_addressee_owner_of_box():
    dad = situation_object(DAD)
    cookie = situation_object(COOKIE)
    box = situation_object(BOX)
    mum = situation_object(MOM, properties=[IS_ADDRESSEE])
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dad, cookie, box, mum],
        always_relations=[Relation(HAS, mum, box)],
        actions=[
            Action(
                action_type=PUT,
                argument_roles_to_fillers=[
                    (AGENT, dad),
                    (THEME, cookie),
                    (GOAL, Region(reference_object=box, distance=INTERIOR)),
                ],
            )
        ],
    )
    assert generated_tokens(situation) == (
        "ba4 ba4",
        "ba3",
        "chyu1 chi2 bing3",
        "fang4",
        "dzai4",
        "ni3 de",
        "syang1 dz",
        "li3",
    )


# test the third person possessive, expressed by a separate speaker
def test_speaker_not_owner_of_box():
    dad = situation_object(DAD)
    cookie = situation_object(COOKIE)
    box = situation_object(BOX)
    mum = situation_object(MOM, properties=[IS_SPEAKER])
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dad, cookie, box, mum],
        always_relations=[Relation(HAS, dad, box)],
        actions=[
            Action(
                action_type=PUT,
                argument_roles_to_fillers=[
                    (AGENT, mum),
                    (THEME, cookie),
                    (GOAL, Region(reference_object=box, distance=INTERIOR)),
                ],
            )
        ],
    )
    assert generated_tokens(situation) == (
        "wo3",
        "ba3",
        "chyu1 chi2 bing3",
        "fang4",
        "dzai4",
        "ba4 ba4",
        "de",
        "syang1 dz",
        "li3",
    )


"""TEST HAVE POSSESSION -- CHECKED BY NATIVE SPEAKER"""

# this tests possession where the speaker has something that they possess
def test_i_have_my_ball():
    baby = situation_object(BABY, properties=[IS_SPEAKER])
    ball = situation_object(BALL)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[baby, ball],
        always_relations=[Relation(HAS, baby, ball)],
        actions=[],
    )
    assert generated_tokens(situation) == ("wo3", "you3", "wo3 de", "chyou2")


# this tests possession when a non-speaker has something that they possess
def test_dad_has_cookie():
    dad = situation_object(DAD)
    cookie = situation_object(COOKIE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dad, cookie],
        always_relations=[Relation(HAS, dad, cookie)],
        # actions=[],
    )
    assert generated_tokens(situation) == ("ba4 ba4", "you3", "chyu1 chi2 bing3")


# this tests possession where the addressee has something that they possess
def test_you_have_your_ball():
    baby = situation_object(BABY, properties=[IS_ADDRESSEE])
    ball = situation_object(BALL)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[baby, ball],
        always_relations=[Relation(HAS, baby, ball)],
        actions=[],
    )
    assert generated_tokens(situation) == ("ni3", "you3", "ni3 de", "chyou2")


"""PATH MODIFIERS -- CHECKED BY NATIVE SPEAKER"""

# this tests flight over a location using at_some_point
def test_path_modifier():
    bird = situation_object(BIRD)
    house = situation_object(HOUSE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[bird, house],
        actions=[
            Action(
                FLY,
                argument_roles_to_fillers=[(AGENT, bird)],
                during=DuringAction(
                    at_some_point=[
                        Relation(
                            IN_REGION,
                            bird,
                            Region(
                                reference_object=house,
                                distance=DISTAL,
                                direction=Direction(
                                    positive=True,
                                    relative_to_axis=GRAVITATIONAL_AXIS_FUNCTION,
                                ),
                            ),
                        )
                    ]
                ),
            )
        ],
    )
    assert generated_tokens(situation) == ("nyau3", "fei1", "gwo4", "wu1", "shang4 myan4")


# a slightly different test for over
def test_jumps_over():
    dad = situation_object(DAD)
    chair = situation_object(CHAIR)
    ground = situation_object(GROUND)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dad, chair],
        actions=[
            Action(
                JUMP,
                argument_roles_to_fillers=[(AGENT, dad)],
                during=DuringAction(at_some_point=[strictly_above(dad, chair)]),
                auxiliary_variable_bindings=[(JUMP_INITIAL_SUPPORTER_AUX, ground)],
            )
        ],
    )
    assert generated_tokens(situation) == (
        "ba4 ba4",
        "tyau4",
        "gwo4",
        "yi3 dz",
        "shang4 myan4",
    )


# this tests under, using dao. After_action relations are translated before during_action relations
def test_path_modifier_under():
    bird = situation_object(BIRD)
    table = situation_object(TABLE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[bird, table],
        actions=[
            Action(
                FLY,
                argument_roles_to_fillers=[(AGENT, bird)],
                during=DuringAction(
                    at_some_point=[
                        Relation(
                            IN_REGION,
                            bird,
                            Region(
                                reference_object=table,
                                distance=DISTAL,
                                direction=GRAVITATIONAL_DOWN,
                            ),
                        )
                    ]
                ),
            )
        ],
        after_action_relations=[
            Relation(
                IN_REGION,
                bird,
                Region(
                    reference_object=table, distance=DISTAL, direction=GRAVITATIONAL_DOWN
                ),
            )
        ],
    )
    assert generated_tokens(situation) == (
        "nyau3",
        "fei1",
        "dau4",
        "jwo1 dz",
        "sya4 myan4",
    )


# this is a different case for Chinese since there's no change in location so the PP is preverbal
def test_path_modifier_on():
    mom = situation_object(MOM)
    ball = situation_object(BALL)
    table = situation_object(TABLE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mom, ball, table],
        actions=[Action(ROLL, argument_roles_to_fillers=[(AGENT, mom), (THEME, ball)])],
        always_relations=[on(ball, table)],
    )
    assert generated_tokens(situation) == (
        "ma1 ma1",
        "dzai4",
        "jwo1 dz",
        "shang4",
        "gwun3",
        "chyou2",
    )


# test besides for path of flight
def test_bird_flies_path_beside():
    bird = situation_object(BIRD)
    car = situation_object(CAR)
    car_region = Region(
        car,
        distance=PROXIMAL,
        direction=Direction(
            positive=True, relative_to_axis=HorizontalAxisOfObject(car, index=0)
        ),
    )

    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[bird, car],
        actions=[
            Action(
                FLY,
                argument_roles_to_fillers=[(AGENT, bird)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            bird,
                            SpatialPath(
                                VIA,
                                reference_source_object=car_region,
                                reference_destination_object=car_region,
                                reference_axis=HorizontalAxisOfObject(car, index=0),
                            ),
                        )
                    ],
                    at_some_point=[Relation(IN_REGION, bird, car_region)],
                ),
            )
        ],
    )
    assert generated_tokens(situation) == (
        "nyau3",
        "fei1",
        "gwo4",
        "chi4 che1",
        "pang2 byan1",
    )


# test path modifiers with intransitive verbs
def test_ball_fell_on_ground():
    ball = situation_object(BALL)
    ground = situation_object(GROUND)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[ball, ground],
        actions=[Action(FALL, argument_roles_to_fillers=[(THEME, ball)])],
        after_action_relations=[on(ball, ground)],
    )
    assert generated_tokens(situation) == (
        "chyou2",
        "dye2 dau3",
        "dau4",
        "di4 myan4",
        "shang4",
    )


# another intransitive verb
def test_mom_sits_on_a_table():
    mom = situation_object(MOM)
    table = situation_object(TABLE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mom, table],
        actions=[
            Action(
                SIT,
                argument_roles_to_fillers=[
                    (AGENT, mom),
                    (
                        GOAL,
                        Region(
                            table,
                            direction=GRAVITATIONAL_UP,
                            distance=EXTERIOR_BUT_IN_CONTACT,
                        ),
                    ),
                ],
            )
        ],
    )
    assert generated_tokens(situation) == (
        "ma1 ma1",
        "dzwo4",
        "dzai4",
        "jwo1 dz",
        "shang4",
    )


"""ADV MODIFICATION -- CHECKED BY NATIVE SPEAKER"""

# fall down testing
def test_falling_down():
    ball = situation_object(BALL)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[ball],
        actions=[Action(FALL, argument_roles_to_fillers=[(THEME, ball)])],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER],
    )
    assert generated_tokens(situation) == ("chyou2", "dye2 dau3", "sya4 lai2")


# sit down testing
def test_sitting_down():
    mum = situation_object(MOM)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mum],
        actions=[Action(SIT, argument_roles_to_fillers=[(AGENT, mum)])],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER],
    )
    assert generated_tokens(situation) == ("ma1 ma1", "dzwo4", "sya4 lai2")


# test sitting up -- falling up doesn't make sense but we want to be sure we can use both despite the defaults in the
# language generator
def test_sitting_up():
    mum = situation_object(MOM)
    ground = situation_object(GROUND)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mum],
        actions=[
            Action(
                SIT,
                argument_roles_to_fillers=[(AGENT, mum)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            mum,
                            SpatialPath(
                                operator=AWAY_FROM,
                                reference_source_object=ground,
                                reference_destination_object=Region(
                                    ground, distance=DISTAL
                                ),
                            ),
                        )
                    ]
                ),
            )
        ],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER],
    )
    assert generated_tokens(situation) == ("ma1 ma1", "dzwo4", "chi3 lai2")


# direction of flight = up
def test_bird_flies_up():
    bird = situation_object(BIRD)
    ground = situation_object(GROUND)

    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[bird],
        actions=[
            Action(
                FLY,
                argument_roles_to_fillers=[(AGENT, bird)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            bird,
                            SpatialPath(
                                operator=AWAY_FROM,
                                reference_source_object=ground,
                                reference_destination_object=Region(
                                    ground, distance=DISTAL
                                ),
                            ),
                        )
                    ]
                ),
            )
        ],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER],
    )
    assert generated_tokens(situation) == ("nyau3", "fei1", "chi3 lai2")


# direction of flight = up
def test_bird_flies_down():
    bird = situation_object(BIRD)
    ground = situation_object(GROUND)

    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[bird],
        actions=[
            Action(
                FLY,
                argument_roles_to_fillers=[(AGENT, bird)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            bird,
                            SpatialPath(
                                operator=TOWARD,
                                reference_source_object=Region(ground, distance=DISTAL),
                                reference_destination_object=ground,
                            ),
                        )
                    ]
                ),
            )
        ],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER],
    )
    assert generated_tokens(situation) == ("nyau3", "fei1", "sya4 lai2")


"""GO/COME BEHAVE DIFFERENTLY IN CHINESE WITH ADVMODS -- CHECKED BY NATIVE SPEAKER"""

# testing going up
def test_going_up():
    bird = situation_object(BIRD)
    ground = situation_object(GROUND)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[bird],
        actions=[
            Action(
                GO,
                argument_roles_to_fillers=[(AGENT, bird)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            bird,
                            SpatialPath(
                                operator=AWAY_FROM,
                                reference_source_object=ground,
                                reference_destination_object=Region(
                                    ground, distance=DISTAL
                                ),
                            ),
                        )
                    ]
                ),
            )
        ],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER],
    )
    assert generated_tokens(situation) == ("nyau3", "shang4", "chyu4")


# testing going down
def test_going_down():
    bird = situation_object(BIRD)
    ground = situation_object(GROUND)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[bird],
        actions=[
            Action(
                GO,
                argument_roles_to_fillers=[(AGENT, bird)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            bird,
                            SpatialPath(
                                operator=TOWARD,
                                reference_source_object=Region(ground, distance=DISTAL),
                                reference_destination_object=ground,
                            ),
                        )
                    ]
                ),
            )
        ],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER],
    )
    assert generated_tokens(situation) == ("nyau3", "sya4", "chyu4")


# testing coming up
def test_coming_up():
    bird = situation_object(BIRD)
    ground = situation_object(GROUND)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[bird],
        actions=[
            Action(
                COME,
                argument_roles_to_fillers=[(AGENT, bird)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            bird,
                            SpatialPath(
                                operator=AWAY_FROM,
                                reference_source_object=ground,
                                reference_destination_object=Region(
                                    ground, distance=DISTAL
                                ),
                            ),
                        )
                    ]
                ),
            )
        ],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER],
    )
    assert generated_tokens(situation) == ("nyau3", "shang4", "lai2")


# testing going down
def test_coming_down():
    bird = situation_object(BIRD)
    ground = situation_object(GROUND)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[bird],
        actions=[
            Action(
                COME,
                argument_roles_to_fillers=[(AGENT, bird)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            bird,
                            SpatialPath(
                                operator=TOWARD,
                                reference_source_object=Region(ground, distance=DISTAL),
                                reference_destination_object=ground,
                            ),
                        )
                    ]
                ),
            )
        ],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER],
    )
    assert generated_tokens(situation) == ("nyau3", "sya4", "lai2")


# direction of jumping
def test_jump_up():
    dad = situation_object(DAD)
    ground = situation_object(GROUND)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dad],
        actions=[
            Action(
                JUMP,
                argument_roles_to_fillers=[(AGENT, dad)],
                auxiliary_variable_bindings=[(JUMP_INITIAL_SUPPORTER_AUX, ground)],
            )
        ],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER],
    )
    assert generated_tokens(situation) == ("ba4 ba4", "tyau4", "chi3 lai2")


# direction of jumping
def test_jump_down():
    dad = situation_object(DAD)
    ground = situation_object(GROUND)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dad],
        actions=[
            Action(
                JUMP,
                argument_roles_to_fillers=[(AGENT, dad)],
                auxiliary_variable_bindings=[(JUMP_INITIAL_SUPPORTER_AUX, ground)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            dad,
                            SpatialPath(
                                operator=TOWARD,
                                reference_source_object=Region(ground, distance=DISTAL),
                                reference_destination_object=ground,
                            ),
                        )
                    ]
                ),
            )
        ],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER],
    )
    assert generated_tokens(situation) == ("ba4 ba4", "tyau4", "sya4 lai2")


"""GO WITH GOAL -- CHECKED BY NATIVE SPEAKER"""


# this tests dao for going to a region, in which case we use the bare NP instead of a localiser phrase
def test_to_regions_as_goal():
    goal_object = situation_object(BOX, properties=[HOLLOW])
    assert generated_tokens(
        region_as_goal_situation(Region(goal_object, distance=PROXIMAL), goal_object)
    ) == ("gou3", "chyu4", "syang1 dz")


# this tests being inside a region
def test_in_region_as_goal():
    goal_object = situation_object(BOX, properties=[HOLLOW])
    assert generated_tokens(
        region_as_goal_situation(Region(goal_object, distance=INTERIOR), goal_object)
    ) == ("gou3", "chyu4", "dau4", "syang1 dz", "li3")


# this tests being next to a region
def test_beside_region_as_goal():
    goal_object = situation_object(BOX, properties=[HOLLOW])
    # Beside
    assert generated_tokens(
        region_as_goal_situation(
            Region(
                goal_object,
                distance=PROXIMAL,
                direction=Direction(
                    positive=True,
                    relative_to_axis=HorizontalAxisOfObject(goal_object, index=0),
                ),
            ),
            goal_object,
        )
    ) == ("gou3", "chyu4", "dau4", "syang1 dz", "pang2 byan1")


# this tests going behind a region
def test_behind_region_as_goal():
    goal_object = situation_object(BOX, properties=[HOLLOW])
    # Behind
    assert generated_tokens(
        region_as_goal_situation(
            Region(
                goal_object,
                distance=PROXIMAL,
                direction=Direction(
                    positive=False, relative_to_axis=FacingAddresseeAxis(goal_object)
                ),
            ),
            goal_object,
        )
    ) == ("gou3", "chyu4", "dau4", "syang1 dz", "hou4 myan4")


# this tests going in front of a region
def test_in_front_of_region_as_goal():
    # In front of
    goal_object = situation_object(BOX, properties=[HOLLOW])
    assert generated_tokens(
        region_as_goal_situation(
            Region(
                goal_object,
                distance=PROXIMAL,
                direction=Direction(
                    positive=True, relative_to_axis=FacingAddresseeAxis(goal_object)
                ),
            ),
            goal_object,
        )
    ) == ("gou3", "chyu4", "dau4", "syang1 dz", "chyan2 myan4")


# this tests going over a region
def test_over_region_as_goal():
    goal_object = situation_object(TABLE)
    # Over
    assert generated_tokens(
        region_as_goal_situation(
            Region(goal_object, distance=PROXIMAL, direction=GRAVITATIONAL_UP),
            goal_object,
        )
    ) == ("gou3", "chyu4", "dau4", "jwo1 dz", "shang4 myan4")


# this tests going under a region
def test_under_region_as_goal():
    goal_object = situation_object(TABLE)
    # Over
    assert generated_tokens(
        region_as_goal_situation(
            Region(goal_object, distance=PROXIMAL, direction=GRAVITATIONAL_DOWN),
            goal_object,
        )
    ) == ("gou3", "chyu4", "dau4", "jwo1 dz", "sya4 myan4")


"""COME WITH GOAL-- CHECKED BY NATIVE SPEAKER"""


# this tests dao for going to a region
def test_to_regions_as_goal_come():
    goal_object = situation_object(BOX, properties=[HOLLOW])
    assert generated_tokens(
        region_as_goal_situation_come(Region(goal_object, distance=PROXIMAL), goal_object)
    ) == ("gou3", "lai2", "syang1 dz")


# this tests being inside a region
def test_in_region_as_goal_come():
    goal_object = situation_object(BOX, properties=[HOLLOW])
    assert generated_tokens(
        region_as_goal_situation_come(Region(goal_object, distance=INTERIOR), goal_object)
    ) == ("gou3", "lai2", "dau4", "syang1 dz", "li3")


# this tests being next to a region
def test_beside_region_as_goal_come():
    goal_object = situation_object(BOX, properties=[HOLLOW])
    # Beside
    assert generated_tokens(
        region_as_goal_situation_come(
            Region(
                goal_object,
                distance=PROXIMAL,
                direction=Direction(
                    positive=True,
                    relative_to_axis=HorizontalAxisOfObject(goal_object, index=0),
                ),
            ),
            goal_object,
        )
    ) == ("gou3", "lai2", "dau4", "syang1 dz", "pang2 byan1")


# this tests going behind a region
def test_behind_region_as_goal_come():
    goal_object = situation_object(BOX, properties=[HOLLOW])
    # Behind
    assert generated_tokens(
        region_as_goal_situation_come(
            Region(
                goal_object,
                distance=PROXIMAL,
                direction=Direction(
                    positive=False, relative_to_axis=FacingAddresseeAxis(goal_object)
                ),
            ),
            goal_object,
        )
    ) == ("gou3", "lai2", "dau4", "syang1 dz", "hou4 myan4")


# this tests going in front of a region
def test_in_front_of_region_as_goal_come():
    # In front of
    goal_object = situation_object(BOX, properties=[HOLLOW])
    assert generated_tokens(
        region_as_goal_situation_come(
            Region(
                goal_object,
                distance=PROXIMAL,
                direction=Direction(
                    positive=True, relative_to_axis=FacingAddresseeAxis(goal_object)
                ),
            ),
            goal_object,
        )
    ) == ("gou3", "lai2", "dau4", "syang1 dz", "chyan2 myan4")


# this tests going over a region
def test_over_region_as_goal_come():
    goal_object = situation_object(TABLE)
    # Over
    assert generated_tokens(
        region_as_goal_situation_come(
            Region(goal_object, distance=PROXIMAL, direction=GRAVITATIONAL_UP),
            goal_object,
        )
    ) == ("gou3", "lai2", "dau4", "jwo1 dz", "shang4 myan4")


# this tests going under a region
def test_under_region_as_goal_come():
    goal_object = situation_object(TABLE)
    # Over
    assert generated_tokens(
        region_as_goal_situation_come(
            Region(goal_object, distance=PROXIMAL, direction=GRAVITATIONAL_DOWN),
            goal_object,
        )
    ) == ("gou3", "lai2", "dau4", "jwo1 dz", "sya4 myan4")


"""MISC TESTS REPLICATED FROM ENGLISH TESTING FILE -- CHECKED BY NATIVE SPEAKER"""


def test_roll():
    agent = situation_object(BABY)
    theme = situation_object(COOKIE)
    surface = situation_object(BOX)

    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[agent, theme, surface],
        actions=[
            Action(
                ROLL,
                argument_roles_to_fillers=[(AGENT, agent), (THEME, theme)],
                auxiliary_variable_bindings=[(ROLL_SURFACE_AUXILIARY, surface)],
            )
        ],
        always_relations=[on(theme, surface)],
    )
    assert generated_tokens(situation) == (
        "bau3 bau3",
        "dzai4",
        "syang1 dz",
        "shang4",
        "gwun3",
        "chyu1 chi2 bing3",
    )


"""TESTS FOR X_IS_Y"""

# simple x_is_y
def test_the_ball_is_green():
    ball = situation_object(BALL, properties=[GREEN])
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[ball],
        syntax_hints=[ATTRIBUTES_AS_X_IS_Y],
    )
    assert generated_tokens(situation) == ("chyou2", "shr4", "lyu4 se4")


# x_is_y with first person possession
def test_my_ball_is_green():
    ball = situation_object(BALL, properties=[GREEN])
    dad = situation_object(DAD, properties=[IS_SPEAKER])
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[ball, dad],
        always_relations=[Relation(HAS, dad, ball)],
        syntax_hints=[IGNORE_HAS_AS_VERB, ATTRIBUTES_AS_X_IS_Y],
    )
    assert generated_tokens(situation) == ("wo3 de", "chyou2", "shr4", "lyu4 se4")


# x_is_y with second person possession
def test_your_ball_is_red():
    ball = situation_object(BALL, properties=[RED])
    dad = situation_object(DAD, properties=[IS_ADDRESSEE])
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[ball, dad],
        always_relations=[Relation(HAS, dad, ball)],
        syntax_hints=[IGNORE_HAS_AS_VERB, ATTRIBUTES_AS_X_IS_Y],
    )
    assert generated_tokens(situation) == ("ni3 de", "chyou2", "shr4", "hung2 se4")


# x_is_y for classifier sentence
def test_two_balls_are_green():
    ball1 = situation_object(BALL, properties=[GREEN], debug_handle="ball1")
    ball2 = situation_object(BALL, properties=[GREEN], debug_handle="ball2")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[ball1, ball2],
        syntax_hints=[ATTRIBUTES_AS_X_IS_Y],
    )
    assert generated_tokens(situation) == ("lyang3", "ge4", "chyou2", "shr4", "lyu4 se4")


# x_is_y for a sentence with many (> 2) objects
def test_many_balls_are_red():
    ball1 = situation_object(BALL, properties=[RED], debug_handle="ball1")
    ball2 = situation_object(BALL, properties=[RED], debug_handle="ball2")
    ball3 = situation_object(BALL, properties=[RED], debug_handle="ball3")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[ball1, ball2, ball3],
        syntax_hints=[ATTRIBUTES_AS_X_IS_Y],
    )
    assert generated_tokens(situation) == ("hen3 dwo1", "chyou2", "shr4", "hung2 se4")


# test the new vocab item added
def test_my_watermelon():
    watermelon = situation_object(WATERMELON)
    baby = situation_object(BABY, properties=[IS_SPEAKER])
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[watermelon, baby],
        always_relations=[Relation(HAS, baby, watermelon)],
        syntax_hints=[IGNORE_HAS_AS_VERB],
    )
    assert generated_tokens(situation) == ("wo3 de", "syi1 gwa1")


# testing of towards/away from
def test_dad_moves_towards_cookie():
    dad = situation_object(DAD)
    cookie = situation_object(COOKIE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dad, cookie],
        actions=[
            Action(
                MOVE,
                argument_roles_to_fillers=[(AGENT, dad)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            dad,
                            SpatialPath(
                                operator=TOWARD,
                                reference_source_object=Region(cookie, distance=DISTAL),
                                reference_destination_object=cookie,
                                reference_axis=HorizontalAxisOfObject(dad, 1),
                            ),
                        )
                    ]
                ),
            )
        ],
    )
    assert generated_tokens(situation) == (
        "ba4 ba4",
        "chau2",
        "chyu1 chi2 bing3",
        "yi2 dung4",
    )


def test_dad_moves_away_from_cookie():
    dad = situation_object(DAD)
    cookie = situation_object(COOKIE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dad, cookie],
        actions=[
            Action(
                MOVE,
                argument_roles_to_fillers=[(AGENT, dad)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            dad,
                            SpatialPath(
                                operator=AWAY_FROM,
                                reference_source_object=cookie,
                                reference_destination_object=Region(
                                    cookie, distance=DISTAL
                                ),
                                reference_axis=HorizontalAxisOfObject(dad, 1),
                            ),
                        )
                    ]
                ),
            )
        ],
    )
    assert generated_tokens(situation) == (
        "ba4 ba4",
        "li2",
        "chyu1 chi2 bing3",
        "yi2 dung4",
    )


def test_jump_fast():
    mum = situation_object(MOM)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mum],
        actions=[
            Action(
                JUMP,
                argument_roles_to_fillers=[(AGENT, mum)],
                auxiliary_variable_bindings=[
                    (JUMP_INITIAL_SUPPORTER_AUX, situation_object(GROUND))
                ],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            mum,
                            SpatialPath(
                                None,
                                reference_source_object=situation_object(GROUND),
                                reference_destination_object=situation_object(GROUND),
                                properties=[FAST],
                            ),
                        )
                    ]
                ),
            )
        ],
    )
    assert generated_tokens(situation) == ("ma1 ma1", "kwai4 su4", "tyau4")


def test_I_walk_fast():
    mum = situation_object(MOM, properties=[IS_SPEAKER])
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mum],
        actions=[
            Action(
                WALK,
                argument_roles_to_fillers=[(AGENT, mum)],
                auxiliary_variable_bindings=[
                    (WALK_SURFACE_AUXILIARY, situation_object(GROUND))
                ],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            mum,
                            SpatialPath(
                                None,
                                reference_source_object=situation_object(GROUND),
                                reference_destination_object=situation_object(GROUND),
                                properties=[FAST],
                            ),
                        )
                    ]
                ),
            )
        ],
    )
    assert generated_tokens(situation) == ("wo3", "kwai4 su4", "bu4 sying2")


def test_I_walk_slowly():
    mum = situation_object(MOM, properties=[IS_SPEAKER])
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mum],
        actions=[
            Action(
                WALK,
                argument_roles_to_fillers=[(AGENT, mum)],
                auxiliary_variable_bindings=[
                    (WALK_SURFACE_AUXILIARY, situation_object(GROUND))
                ],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            mum,
                            SpatialPath(
                                None,
                                reference_source_object=situation_object(GROUND),
                                reference_destination_object=situation_object(GROUND),
                                properties=[SLOW],
                            ),
                        )
                    ]
                ),
            )
        ],
    )
    assert generated_tokens(situation) == ("wo3", "man4 man", "bu4 sying2")


def test_I_walk_out_of_house():
    dad = situation_object(DAD, properties=[IS_SPEAKER])
    house = situation_object(HOUSE)
    inside_relation = inside(dad, house)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dad, house],
        actions=[
            Action(
                WALK,
                argument_roles_to_fillers=[
                    (AGENT, dad),
                    (GOAL, Region(house, distance=DISTAL)),
                ],
            )
        ],
        before_action_relations=flatten_relations(inside_relation),
        after_action_relations=flatten_relations(
            [relation.negated_copy() for relation in inside_relation]
        ),
        syntax_hints=[IGNORE_GOAL],
    )
    assert generated_tokens(situation) == ("wo3", "bu4 sying2", "chu1", "wu1")


def test_big_truck():
    learner = situation_object(LEARNER)
    truck = situation_object(TRUCK)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[truck],
        always_relations=[(bigger_than(truck, learner))],
    )
    assert generated_tokens(situation) == ("da4", "ka3 che1")


def test_tall_truck():
    learner = situation_object(LEARNER)
    truck = situation_object(TRUCK)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[truck],
        always_relations=[(bigger_than(truck, learner))],
        syntax_hints=[USE_VERTICAL_MODIFIERS],
    )
    assert generated_tokens(situation) == ("gau1 da4", "ka3 che1")


def test_small_truck():
    learner = situation_object(LEARNER)
    truck = situation_object(TRUCK)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[truck],
        always_relations=[(bigger_than(learner, truck))],
    )
    assert generated_tokens(situation) == ("syau3", "ka3 che1")


def test_short_truck():
    learner = situation_object(LEARNER)
    truck = situation_object(TRUCK)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[truck],
        always_relations=[(bigger_than(learner, truck))],
        syntax_hints=[USE_VERTICAL_MODIFIERS],
    )
    assert generated_tokens(situation) == ("dwan3", "ka3 che1")


# there is no under/below distinction in Chinese
def test_ball_under_below_cookie():
    ball = situation_object(BALL)
    cookie = situation_object(COOKIE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[ball, cookie],
        always_relations=[strictly_under(ball, cookie, dist=DISTAL)],
    )
    assert generated_tokens(situation) == (
        "dzai4",
        "chyu1 chi2 bing3",
        "sya4 myan4",
        "de",
        "chyou2",
    )


def test_ball_over_cookie():
    ball = situation_object(BALL)
    cookie = situation_object(COOKIE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[ball, cookie],
        always_relations=[strictly_over(ball, cookie, dist=DISTAL)],
    )
    assert generated_tokens(situation) == (
        "dzai4",
        "chyu1 chi2 bing3",
        "shang4 myan4",
        "de",
        "chyou2",
    )


def test_ball_above_cookie():
    ball = situation_object(BALL)
    cookie = situation_object(COOKIE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[ball, cookie],
        always_relations=[strictly_over(ball, cookie, dist=DISTAL)],
        syntax_hints=[USE_ABOVE_BELOW],
    )
    assert generated_tokens(situation) == (
        "dzai4",
        "chyu1 chi2 bing3",
        "shang4 fang1",
        "de",
        "chyou2",
    )


def test_ball_far_from_cookie():
    ball = situation_object(BALL)
    cookie = situation_object(COOKIE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[ball, cookie],
        always_relations=[far(ball, cookie)],
    )
    assert generated_tokens(situation) == (
        "li2",
        "chyu1 chi2 bing3",
        "hen3 ywan3",
        "de",
        "chyou2",
    )


def test_run():
    mom = situation_object(MOM, properties=[IS_SPEAKER])
    mom_runs = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mom],
        actions=[
            Action(
                WALK,
                auxiliary_variable_bindings=[(WALK_SURFACE_AUXILIARY, GROUND)],
                argument_roles_to_fillers=[(AGENT, mom)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            mom,
                            SpatialPath(
                                None,
                                reference_source_object=GROUND,
                                reference_destination_object=GROUND,
                                properties=[HARD_FORCE],
                            ),
                        )
                    ]
                ),
            )
        ],
    )
    assert generated_tokens(mom_runs) == ("wo3", "pau3")


def test_run_fast():
    mom = situation_object(MOM)
    mom_runs = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mom],
        actions=[
            Action(
                WALK,
                auxiliary_variable_bindings=[(WALK_SURFACE_AUXILIARY, GROUND)],
                argument_roles_to_fillers=[(AGENT, mom)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            mom,
                            SpatialPath(
                                None,
                                reference_source_object=GROUND,
                                reference_destination_object=GROUND,
                                properties=[HARD_FORCE, FAST],
                            ),
                        )
                    ]
                ),
            )
        ],
    )
    assert generated_tokens(mom_runs) == ("ma1 ma1", "kwai4 su4", "pau3")


def test_run_slow():
    mom = situation_object(MOM)
    mom_runs = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mom],
        actions=[
            Action(
                WALK,
                auxiliary_variable_bindings=[(WALK_SURFACE_AUXILIARY, GROUND)],
                argument_roles_to_fillers=[(AGENT, mom)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            mom,
                            SpatialPath(
                                None,
                                reference_source_object=GROUND,
                                reference_destination_object=GROUND,
                                properties=[HARD_FORCE, SLOW],
                            ),
                        )
                    ]
                ),
            )
        ],
    )
    assert generated_tokens(mom_runs) == ("ma1 ma1", "man4 man", "pau3")


def test_I_run_out_of_car_slowly():
    dad = situation_object(DAD, properties=[IS_SPEAKER])
    car = situation_object(CAR)
    inside_relation = inside(dad, car)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dad, car],
        actions=[
            Action(
                WALK,
                argument_roles_to_fillers=[
                    (AGENT, dad),
                    (GOAL, Region(car, distance=DISTAL)),
                ],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            dad,
                            SpatialPath(
                                None,
                                reference_source_object=GROUND,
                                reference_destination_object=GROUND,
                                properties=[HARD_FORCE, SLOW],
                            ),
                        )
                    ]
                ),
            )
        ],
        before_action_relations=flatten_relations(inside_relation),
        after_action_relations=flatten_relations(
            [relation.negated_copy() for relation in inside_relation]
        ),
        syntax_hints=[IGNORE_GOAL],
    )
    assert generated_tokens(situation) == ("wo3", "man4 man", "pau3", "chu1", "chi4 che1")


def test_dad_slowly_grabs_cookie():
    dad = situation_object(DAD)
    cookie = situation_object(COOKIE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dad, cookie],
        actions=[
            Action(
                TAKE,
                argument_roles_to_fillers=[(AGENT, dad), (THEME, cookie)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            dad,
                            SpatialPath(
                                None,
                                reference_source_object=GROUND,
                                reference_destination_object=GROUND,
                                properties=[HARD_FORCE, SLOW],
                            ),
                        )
                    ]
                ),
            )
        ],
        syntax_hints=[IGNORE_GOAL],
    )
    assert generated_tokens(situation) == (
        "ba4 ba4",
        "man4 man",
        "chyang3 na2 chi3",
        "chyu1 chi2 bing3",
    )


def test_i_shove_a_ball_on_a_table_dao():
    mum = situation_object(MOM, properties=[IS_SPEAKER])
    ball = situation_object(BALL)
    table = situation_object(TABLE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mum, ball, table],
        actions=[
            Action(
                action_type=PUSH,
                argument_roles_to_fillers=[
                    (AGENT, mum),
                    (THEME, ball),
                    (
                        GOAL,
                        Region(
                            reference_object=table,
                            distance=EXTERIOR_BUT_IN_CONTACT,
                            direction=GRAVITATIONAL_UP,
                        ),
                    ),
                ],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            mum,
                            SpatialPath(
                                None,
                                reference_source_object=GROUND,
                                reference_destination_object=GROUND,
                                properties=[HARD_FORCE],
                            ),
                        )
                    ]
                ),
            )
        ],
        after_action_relations=[on(ball, table)],
    )
    assert generated_tokens(situation) == (
        "wo3",
        "ba3",
        "chyou2",
        "yung4 li4 twei1",
        "dau4",
        "jwo1 dz",
        "shang4",
    )


def test_i_shove_a_table():
    mum = situation_object(MOM, properties=[IS_SPEAKER])
    table = situation_object(TABLE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mum, table],
        actions=[
            Action(
                action_type=PUSH,
                argument_roles_to_fillers=[(AGENT, mum), (THEME, table)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            mum,
                            SpatialPath(
                                None,
                                reference_source_object=GROUND,
                                reference_destination_object=GROUND,
                                properties=[HARD_FORCE],
                            ),
                        )
                    ]
                ),
            )
        ],
    )
    assert generated_tokens(situation) == ("wo3", "yung4 li4 twei1", "jwo1 dz")


def test_i_push_a_table():
    mum = situation_object(MOM, properties=[IS_SPEAKER])
    table = situation_object(TABLE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mum, table],
        actions=[
            Action(
                action_type=PUSH, argument_roles_to_fillers=[(AGENT, mum), (THEME, table)]
            )
        ],
    )
    assert generated_tokens(situation) == ("wo3", "twei1", "jwo1 dz")


def test_i_push_table_down_slowly():
    mum = situation_object(MOM, properties=[IS_SPEAKER])
    table = situation_object(TABLE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mum, table],
        actions=[
            Action(
                action_type=PUSH,
                argument_roles_to_fillers=[(AGENT, mum), (THEME, table)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            mum,
                            SpatialPath(
                                operator=TOWARD,
                                reference_source_object=situation_object(GROUND),
                                reference_destination_object=situation_object(GROUND),
                                properties=[SLOW],
                            ),
                        )
                    ]
                ),
            )
        ],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER],
    )
    assert generated_tokens(situation) == (
        "wo3",
        "man4 man",
        "ba3",
        "jwo1 dz",
        "twei1",
        "sya4 lai2",
    )


def test_i_throw_ball_down():
    mum = situation_object(MOM, properties=[IS_SPEAKER])
    ball = situation_object(BALL)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mum, ball],
        actions=[
            Action(
                action_type=THROW,
                argument_roles_to_fillers=[(AGENT, mum), (THEME, ball)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            mum,
                            SpatialPath(
                                operator=TOWARD,
                                reference_source_object=Region(
                                    situation_object(GROUND), distance=DISTAL
                                ),
                                reference_destination_object=situation_object(GROUND),
                            ),
                        )
                    ]
                ),
            )
        ],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER],
    )
    assert generated_tokens(situation) == ("wo3", "ba3", "chyou2", "reng1", "sya4 lai2")


def test_dad_passes_the_cookie_down_to_me():
    dad = situation_object(DAD)
    mum = situation_object(MOM, properties=[IS_SPEAKER])
    cookie = situation_object(COOKIE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mum, cookie, dad],
        actions=[
            Action(
                action_type=PASS,
                argument_roles_to_fillers=[(AGENT, dad), (THEME, cookie), (GOAL, mum)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            dad,
                            SpatialPath(
                                operator=TOWARD,
                                # TODO : fix ground checking in language generator
                                reference_source_object=Region(
                                    situation_object(GROUND), distance=DISTAL
                                ),
                                reference_destination_object=situation_object(GROUND),
                            ),
                        )
                    ]
                ),
            )
        ],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER],
    )
    assert generated_tokens(situation) == (
        "ba4 ba4",
        "ba3",
        "chyu1 chi2 bing3",
        "di4",
        "sya4 lai2",
        "gei3",
        "wo3",
    )


def test_toss():
    dad = situation_object(DAD)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dad],
        actions=[
            Action(
                PASS,
                argument_roles_to_fillers=[(AGENT, dad)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            dad,
                            SpatialPath(
                                None,
                                reference_source_object=GROUND,
                                reference_destination_object=GROUND,
                                properties=[HARD_FORCE],
                            ),
                        )
                    ]
                ),
            )
        ],
    )
    assert generated_tokens(situation) == ("ba4 ba4", "reng1")


def test_dad_tosses_the_cookie_up_to_me():
    dad = situation_object(DAD)
    mum = situation_object(MOM, properties=[IS_SPEAKER])
    cookie = situation_object(COOKIE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mum, cookie, dad],
        actions=[
            Action(
                action_type=PASS,
                argument_roles_to_fillers=[(AGENT, dad), (THEME, cookie), (GOAL, mum)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            dad,
                            SpatialPath(
                                operator=AWAY_FROM,
                                reference_source_object=situation_object(GROUND),
                                reference_destination_object=Region(
                                    situation_object(GROUND), distance=DISTAL
                                ),
                                properties=[HARD_FORCE],
                            ),
                        )
                    ]
                ),
            )
        ],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER],
    )
    assert generated_tokens(situation) == (
        "ba4 ba4",
        "ba3",
        "chyu1 chi2 bing3",
        "reng1",
        "chi3 lai2",
        "gei3",
        "wo3",
    )


def test_bird_flies_up_towards_mum():
    bird = situation_object(BIRD)
    mum = situation_object(MOM)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[bird, mum],
        actions=[
            Action(
                FLY,
                argument_roles_to_fillers=[(AGENT, bird)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            bird,
                            SpatialPath(
                                operator=TOWARD,
                                reference_source_object=Region(mum, distance=DISTAL),
                                reference_destination_object=mum,
                                reference_axis=HorizontalAxisOfObject(bird, 1),
                            ),
                        ),
                        (
                            bird,
                            SpatialPath(
                                operator=AWAY_FROM,
                                reference_source_object=situation_object(GROUND),
                                reference_destination_object=Region(
                                    situation_object(GROUND), distance=DISTAL
                                ),
                            ),
                        ),
                    ]
                ),
            )
        ],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER],
    )
    assert generated_tokens(situation) == (
        "nyau3",
        "chau2",
        "ma1 ma1",
        "fei1",
        "chi3 lai2",
    )


def test_bird_flies_away_from_mum():
    bird = situation_object(BIRD)
    mum = situation_object(MOM)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[bird, mum],
        actions=[
            Action(
                FLY,
                argument_roles_to_fillers=[(AGENT, bird)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            bird,
                            SpatialPath(
                                operator=AWAY_FROM,
                                reference_source_object=mum,
                                reference_destination_object=Region(mum, distance=DISTAL),
                                reference_axis=HorizontalAxisOfObject(bird, 1),
                            ),
                        )
                    ]
                ),
            )
        ],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER],
    )
    assert generated_tokens(situation) == ("nyau3", "li2", "ma1 ma1", "fei1")


def test_dad_passes_me_cookie():
    dad = situation_object(DAD)
    mum = situation_object(MOM, properties=[IS_SPEAKER])
    cookie = situation_object(COOKIE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mum, cookie, dad],
        actions=[
            Action(
                action_type=PASS,
                argument_roles_to_fillers=[(AGENT, dad), (THEME, cookie), (GOAL, mum)],
            )
        ],
    )
    assert generated_tokens(situation) == (
        "ba4 ba4",
        "ba3",
        "chyu1 chi2 bing3",
        "di4",
        "gei3",
        "wo3",
    )


def test_you_toss_me_cookie():
    dad = situation_object(DAD, properties=[IS_ADDRESSEE])
    mum = situation_object(MOM, properties=[IS_SPEAKER])
    cookie = situation_object(COOKIE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mum, cookie, dad],
        actions=[
            Action(
                action_type=PASS,
                argument_roles_to_fillers=[(AGENT, dad), (THEME, cookie), (GOAL, mum)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            dad,
                            SpatialPath(
                                operator=None,
                                reference_source_object=situation_object(GROUND),
                                reference_destination_object=situation_object(GROUND),
                                properties=[HARD_FORCE],
                            ),
                        )
                    ]
                ),
            )
        ],
    )
    assert generated_tokens(situation) == (
        "ni3",
        "ba3",
        "chyu1 chi2 bing3",
        "reng1",
        "gei3",
        "wo3",
    )


def test_you_toss_mum_cookie():
    dad = situation_object(DAD, properties=[IS_ADDRESSEE])
    mum = situation_object(MOM)
    cookie = situation_object(COOKIE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mum, cookie, dad],
        actions=[
            Action(
                action_type=PASS,
                argument_roles_to_fillers=[(AGENT, dad), (THEME, cookie), (GOAL, mum)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            dad,
                            SpatialPath(
                                operator=None,
                                reference_source_object=situation_object(GROUND),
                                reference_destination_object=situation_object(GROUND),
                            ),
                        )
                    ]
                ),
            )
        ],
    )
    assert generated_tokens(situation) == (
        "ni3",
        "ba3",
        "chyu1 chi2 bing3",
        "di4",
        "gei3",
        "ma1 ma1",
    )


def test_passing_for_phase1():
    agent = situation_object(DAD)
    theme = situation_object(COOKIE)
    goal = situation_object(MOM)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[agent, theme, goal],
        actions=[
            Action(
                PASS,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (THEME, theme),
                    (GOAL, Region(goal, distance=PROXIMAL)),
                ],
            )
        ],
    )
    assert generated_tokens(situation) == (
        "ba4 ba4",
        "ba3",
        "chyu1 chi2 bing3",
        "di4",
        "gei3",
        "ma1 ma1",
    )


def test_multiple_actions():
    agent = situation_object(DAD)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[agent],
        actions=[
            Action(DRINK, argument_roles_to_fillers=[(AGENT, agent)]),
            Action(EAT, argument_roles_to_fillers=[(AGENT, agent)]),
        ],
    )
    with pytest.raises(RuntimeError):
        generated_tokens(situation)


def test_multiple_subjects():
    agent = situation_object(DAD)
    other_agent = situation_object(MOM)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[agent, other_agent],
        actions=[
            Action(
                DRINK, argument_roles_to_fillers=[(AGENT, agent), (AGENT, other_agent)]
            )
        ],
    )
    with pytest.raises(RuntimeError):
        generated_tokens(situation)


def test_non_salient_in_path():
    dad = situation_object(DAD, properties=[IS_ADDRESSEE])
    mum = situation_object(MOM)
    cookie = situation_object(COOKIE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[cookie, dad],
        actions=[
            Action(
                action_type=EAT,
                argument_roles_to_fillers=[(AGENT, dad), (THEME, cookie)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            mum,
                            SpatialPath(
                                operator=None,
                                reference_source_object=cookie,
                                reference_destination_object=cookie,
                            ),
                        )
                    ]
                ),
            )
        ],
    )
    assert generated_tokens(situation) == ("ni3", "chr1", "chyu1 chi2 bing3")


def test_not_yet_used_operator():
    dad = situation_object(DAD, properties=[IS_ADDRESSEE])
    mum = situation_object(MOM)
    cookie = situation_object(COOKIE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[cookie, dad, mum],
        actions=[
            Action(
                action_type=EAT,
                argument_roles_to_fillers=[(AGENT, dad), (THEME, cookie)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            mum,
                            SpatialPath(
                                operator=FROM,
                                reference_source_object=cookie,
                                reference_destination_object=cookie,
                            ),
                        )
                    ]
                ),
            )
        ],
    )
    with pytest.raises(RuntimeError):
        generated_tokens(situation)


def test_bird_flies_away_from_mum_region():
    bird = situation_object(BIRD)
    mum = situation_object(MOM)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[bird, mum],
        actions=[
            Action(
                FLY,
                argument_roles_to_fillers=[(AGENT, bird)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            bird,
                            SpatialPath(
                                operator=AWAY_FROM,
                                reference_source_object=Region(mum, distance=PROXIMAL),
                                reference_destination_object=Region(mum, distance=DISTAL),
                                reference_axis=HorizontalAxisOfObject(bird, 1),
                            ),
                        )
                    ]
                ),
            )
        ],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER],
    )
    assert generated_tokens(situation) == ("nyau3", "li2", "ma1 ma1", "fei1")


def test_region_as_subject_should_fail():
    bird = situation_object(BIRD)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[bird],
        actions=[
            Action(
                FLY, argument_roles_to_fillers=[(AGENT, Region(bird, distance=PROXIMAL))]
            )
        ],
    )
    with pytest.raises(RuntimeError):
        generated_tokens(situation)


def test_near():
    mom = situation_object(MOM)
    dad = situation_object(DAD)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mom, dad],
        always_relations=[near(mom, dad)],
        syntax_hints=[USE_NEAR],
    )
    assert generated_tokens(situation) == (
        "dzai4",
        "ba4 ba4",
        "pang2 byan1",
        "de",
        "ma1 ma1",
    )


def test_far():
    mom = situation_object(MOM)
    dad = situation_object(DAD)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mom, dad],
        always_relations=[far(mom, dad)],
    )
    assert generated_tokens(situation) == (
        "li2",
        "ba4 ba4",
        "hen3 ywan3",
        "de",
        "ma1 ma1",
    )


def test_above():
    mom = situation_object(MOM)
    dad = situation_object(DAD)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mom, dad],
        always_relations=[strictly_over(mom, dad, dist=DISTAL)],
        syntax_hints=[USE_ABOVE_BELOW],
    )
    assert generated_tokens(situation) == (
        "dzai4",
        "ba4 ba4",
        "shang4 fang1",
        "de",
        "ma1 ma1",
    )


def test_go_above_goal():
    mom = situation_object(MOM)
    dad = situation_object(DAD)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mom, dad],
        actions=[
            Action(
                GO,
                argument_roles_to_fillers=[
                    (AGENT, dad),
                    (GOAL, Region(mom, distance=DISTAL, direction=GRAVITATIONAL_UP)),
                ],
            )
        ],
        after_action_relations=[strictly_over(mom, dad, dist=DISTAL)],
        syntax_hints=[USE_ABOVE_BELOW],
    )
    assert generated_tokens(situation) == (
        "ba4 ba4",
        "chyu4",
        "dau4",
        "ma1 ma1",
        "sya4 fang1",
    )


def test_go_near_goal():
    mom = situation_object(MOM)
    dad = situation_object(DAD)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mom, dad],
        actions=[
            Action(
                GO,
                argument_roles_to_fillers=[
                    (AGENT, dad),
                    (GOAL, Region(mom, distance=PROXIMAL)),
                ],
            )
        ],
        after_action_relations=[near(mom, dad)],
        syntax_hints=[USE_NEAR],
    )
    assert generated_tokens(situation) == (
        "ba4 ba4",
        "chyu4",
        "dau4",
        "ma1 ma1",
        "pang2 byan1",
    )


def test_go_far_from_goal():
    mom = situation_object(MOM)
    dad = situation_object(DAD)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mom, dad],
        actions=[
            Action(
                GO,
                argument_roles_to_fillers=[
                    (AGENT, dad),
                    (GOAL, Region(mom, distance=DISTAL)),
                ],
            )
        ],
        after_action_relations=[far(mom, dad)],
        syntax_hints=[USE_NEAR],
    )
    assert generated_tokens(situation) == (
        "ba4 ba4",
        "chyu4",
        "dau4",
        "ma1 ma1",
        "ywan3 li2",
    )


def test_error_for_goal():
    mom = situation_object(MOM)
    dad = situation_object(DAD)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mom, dad],
        actions=[
            Action(
                GO,
                argument_roles_to_fillers=[
                    (AGENT, dad),
                    (
                        GOAL,
                        Region(
                            mom,
                            direction=Direction(
                                positive=True,
                                relative_to_axis=HorizontalAxisOfObject(mom, index=0),
                            ),
                            distance=PROXIMAL,
                        ),
                    ),
                ],
            )
        ],
    )
    with pytest.raises(RuntimeError):
        generated_tokens(situation)


def test_beside_for_goal():
    mom = situation_object(MOM)
    dad = situation_object(DAD)
    learner = situation_object(LEARNER)
    situation = HighLevelSemanticsSituation(
        axis_info=AxesInfo(addressee=learner),
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mom, dad],
        actions=[
            Action(
                GO,
                argument_roles_to_fillers=[
                    (AGENT, dad),
                    (
                        GOAL,
                        Region(
                            mom,
                            direction=Direction(
                                positive=True,
                                relative_to_axis=HorizontalAxisOfObject(mom, index=0),
                            ),
                            distance=PROXIMAL,
                        ),
                    ),
                ],
            )
        ],
        after_action_relations=[near(mom, dad)],
    )
    assert generated_tokens(situation) == (
        "ba4 ba4",
        "chyu4",
        "dau4",
        "ma1 ma1",
        "pang2 byan1",
    )


def test_invalid_role():
    mom = situation_object(MOM)
    dad = situation_object(DAD)
    learner = situation_object(LEARNER)
    situation = HighLevelSemanticsSituation(
        axis_info=AxesInfo(addressee=learner),
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mom, dad],
        actions=[Action(GO, argument_roles_to_fillers=[(SEMANTIC_ROLE, dad)])],
        after_action_relations=[near(mom, dad)],
    )
    with pytest.raises(RuntimeError):
        generated_tokens(situation)


def test_no_x_is_y_dynamic():
    mom = situation_object(MOM)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mom],
        actions=[Action(EAT, argument_roles_to_fillers=[(AGENT, mom)])],
        syntax_hints=[ATTRIBUTES_AS_X_IS_Y],
    )
    with pytest.raises(RuntimeError):
        generated_tokens(situation)


def test_no_x_is_y_without_attributes():
    mom = situation_object(MOM)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mom],
        syntax_hints=[ATTRIBUTES_AS_X_IS_Y],
    )
    with pytest.raises(RuntimeError):
        generated_tokens(situation)


def test_multiple_possession_relations():
    mom = situation_object(MOM)
    dad = situation_object(DAD)
    cookie = situation_object(COOKIE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mom, dad, cookie],
        always_relations=[Relation(HAS, mom, cookie), Relation(HAS, dad, cookie)],
    )
    with pytest.raises(RuntimeError):
        generated_tokens(situation)


def test_invalid_relations():
    mom = situation_object(MOM)
    dad = situation_object(DAD)
    cookie = situation_object(COOKIE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mom, dad, cookie],
        always_relations=[Relation(SPATIAL_RELATION, mom, cookie)],
    )
    with pytest.raises(RuntimeError):
        generated_tokens(situation)


def test_drink_from():
    agent = situation_object(MOM)
    liquid = situation_object(WATER)
    container = situation_object(CUP)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[agent, liquid, container],
        actions=[
            Action(
                DRINK,
                argument_roles_to_fillers=[(AGENT, agent), (THEME, liquid)],
                auxiliary_variable_bindings=[(DRINK_CONTAINER_AUX, container)],
            )
        ],
        always_relations=[inside(liquid, container)],
    )
    assert generated_tokens(situation) == (
        "ma1 ma1",
        "tsung2",
        "bei1 dz",
        "he1",
        "shwei3",
    )


def test_order_ba_and_prep_phrase():
    agent = situation_object(MOM)
    theme = situation_object(BOOK)
    push_surface = situation_object(TABLE)
    push_goal = situation_object(DAD)
    situation = HighLevelSemanticsSituation(
        salient_objects=[agent, theme, push_surface],
        ontology=GAILA_PHASE_1_ONTOLOGY,
        actions=[
            Action(
                PUSH,
                argument_roles_to_fillers=[(AGENT, agent), (THEME, theme)],
                auxiliary_variable_bindings=[
                    (PUSH_SURFACE_AUX, push_surface),
                    (PUSH_GOAL, push_goal),
                ],
                during=DuringAction(
                    continuously=[on(theme, push_surface)],
                    objects_to_paths=[
                        (
                            agent,
                            SpatialPath(
                                None,
                                reference_source_object=Region(
                                    push_goal, distance=DISTAL
                                ),
                                reference_destination_object=push_goal,
                            ),
                        ),
                        (
                            agent,
                            SpatialPath(
                                operator=TOWARD,
                                reference_source_object=situation_object(GROUND),
                                reference_destination_object=situation_object(GROUND),
                            ),
                        ),
                    ],
                ),
            )
        ],
        always_relations=[on(theme, push_surface)],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER],
    )
    assert generated_tokens(situation) == (
        "ma1 ma1",
        "dzai4",
        "jwo1 dz",
        "shang4",
        "ba3",
        "shu1",
        "twei1",
        "sya4 lai2",
    )
