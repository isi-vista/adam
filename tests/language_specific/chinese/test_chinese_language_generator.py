"""This file contains test cases for the Chinese Language Generator,
which is still under development"""

from typing import Tuple
import pytest
from more_itertools import only
from adam.axes import (
    HorizontalAxisOfObject,
    FacingAddresseeAxis,
    AxesInfo,
    GRAVITATIONAL_AXIS_FUNCTION,
)

# TODO: update imports once Chinese syntactic info is updated
from adam.language_specific.chinese.chinese_language_generator import (
    PREFER_DITRANSITIVE,
    SimpleRuleBasedChineseLanguageGenerator,
    USE_ADVERBIAL_PATH_MODIFIER,
    IGNORE_HAS_AS_VERB,
)
from adam.language_specific.chinese.chinese_phase_1_lexicon import (
    GAILA_PHASE_1_CHINESE_LEXICON,
)
from adam.ontology import IN_REGION, IS_SPEAKER, IS_ADDRESSEE
from adam.ontology.during import DuringAction
from adam.ontology.phase1_ontology import (
    AGENT,
    COME,
    BABY,
    BALL,
    BIRD,
    BOX,
    CHAIR,
    COOKIE,
    CUP,
    DAD,
    DRINK,
    DRINK_CONTAINER_AUX,
    EAT,
    FALL,
    FLY,
    GAILA_PHASE_1_ONTOLOGY,
    GIVE,
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
    on,
    strictly_above,
    JUMP,
    JUMP_INITIAL_SUPPORTER_AUX,
    DOG,
    HOLLOW,
    GO,
    LEARNER,
    near,
    TAKE,
    CAR,
    ROLL_SURFACE_AUXILIARY,
    TRUCK,
    HOUSE,
)
from adam.ontology.phase1_spatial_relations import (
    AWAY_FROM,
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
from adam.relation import Relation
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


""" BASIC NOUN PHRASE TESTS"""

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


"""NOUN PHRASES WITH ADJECTIVAL MODIFIERS"""

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
    assert generated_tokens(situation) == ("wo3", "de", "lyu4 se4", "chyou2")


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
    assert generated_tokens(situation) == ("ni3", "de", "lyu4 se4", "chyou2")


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


"""COUNTING NOUN PHRASE TESTS (WITH CLASSIFIERS)"""


# a single object -- note that we don't use yi clf noun yet but this could be implemented later on
def test_single_item():
    dog = situation_object(DOG)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[dog]
    )
    assert generated_tokens(situation) == ("gou3",)


# two objects, which can be counted distinctly
@pytest.mark.skip(reason="https://github.com/isi-vista/adam/issues/782")
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


"""SALIENT VS. NOT SALIENT OBJECTS"""

# two trucks, only one is salient
# https://github.com/isi-vista/adam/issues/802")
def test_one_salient_object():
    truck1 = situation_object(TRUCK, debug_handle="truck1")
    truck2 = situation_object(TRUCK, debug_handle="truck2")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[truck1], other_objects=[truck2]
    )
    assert generated_tokens(situation) == ("ka3 che1",)


# many objects, only two are salient
@pytest.mark.skip(reason="saliency and classifiers aren't yet supported")
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


"""NOUN PHRASES WITH LOCALISER NOMINAL MODIFIERS"""

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


"""BASIC VP TESTING: SV, SVO, and SVIO"""


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


"""VP's WITH PERSONAL PRONOUNS"""

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


"""VP's WITH LOCALIZERS AND VARIOUS SPEAKERS"""
# TODO: handle zai/dao distinction in generator
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


"""TESTS POSSESSIVE: WO DE AND NI DE"""

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
        "wo3",
        "de",
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
        "wo3",
        "de",
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
        "ni3",
        "de",
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
        "wo3",
        "de",
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
        "ni3",
        "de",
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


"""TEST HAVE POSSESSION"""

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
    assert generated_tokens(situation) == ("wo3", "you3", "wo3", "de", "chyou2")


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
    assert generated_tokens(situation) == ("ni3", "you3", "ni3", "de", "chyou2")


"""PATH MODIFIERS"""

# this tests over, which is a special case since it doesn't use zai/dao
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


# this tests under, which is a regular case using 'dao'
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
        after_action_relations=[near(bird, table)],
    )
    assert generated_tokens(situation) == (
        "nyau3",
        "fei1",
        "dau4",
        "jwo1 dz",
        "sya4 myan4",
    )


# this is a different case for Chinese since there's no change in location so the PP is preverbal
# @pytest.mark.skip(reason="path modifiers have not been implemented yet")
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
                                reference_object=car_region,
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


"""ADV MODIFICATION"""
# TODO: check if adverb path modifiers are salient and should be implemented in Chinese
# https://github.com/isi-vista/adam/issues/797

# fall down testing -- this does translate but I'm not sure how much it's used
@pytest.mark.skip("advmods not yet implemented")
def test_falling_down():
    ball = situation_object(BALL)
    situation_without_modifier = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[ball],
        actions=[Action(FALL, argument_roles_to_fillers=[(THEME, ball)])],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER],
    )
    assert generated_tokens(situation) == ("chyou2", "dye2 dau3", "sya4lai2")


# direction of flight
@pytest.mark.skip("advmods not yet implemented")
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
                        (bird, SpatialPath(operator=AWAY_FROM, reference_object=ground))
                    ]
                ),
            )
        ],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER],
    )
    assert generated_tokens(situation) == ("nyau3", "fei1", "chi3lai2")


# direction of jumping
@pytest.mark.skip("advmods not yet implemented")
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
    assert generated_tokens(situation) == ("ba4 ba4", "tyau4", "chi3lai2")


"""GO WITH GOAL"""

# this tests dao for going to a region
def test_to_regions_as_goal():
    goal_object = situation_object(BOX, properties=[HOLLOW])
    assert generated_tokens(
        region_as_goal_situation(Region(goal_object, distance=PROXIMAL), goal_object)
    ) == ("gou3", "chyu4", "dau4", "syang1 dz", "shang4")


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


"""COME WITH GOAL"""

# this tests dao for going to a region
def test_to_regions_as_goal_come():
    goal_object = situation_object(BOX, properties=[HOLLOW])
    assert generated_tokens(
        region_as_goal_situation_come(Region(goal_object, distance=PROXIMAL), goal_object)
    ) == ("gou3", "lai2", "dau4", "syang1 dz", "shang4")


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


"""MISC TESTS REPLICATED FROM ENGLISH TESTING FILE"""


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
