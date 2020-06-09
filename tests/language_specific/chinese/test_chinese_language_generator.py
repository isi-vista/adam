"""This file contains test cases for the Chinese Language Generator,
which is still under development"""

from typing import Tuple
import pytest
from more_itertools import only
from adam.axes import HorizontalAxisOfObject, FacingAddresseeAxis, AxesInfo

# TODO: update imports once Chinese syntactic info is updated
from adam.language_specific.chinese.chinese_language_generator import (
    PREFER_DITRANSITIVE,
    SimpleRuleBasedChineseLanguageGenerator,
    USE_ADVERBIAL_PATH_MODIFIER,
)
from adam.language_specific.chinese.chinese_phase_1_lexicon import (
    GAILA_PHASE_1_CHINESE_LEXICON,
)
from adam.ontology import IN_REGION, IS_SPEAKER, IS_ADDRESSEE
from adam.ontology.during import DuringAction
from adam.ontology.phase1_ontology import (
    AGENT,
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


""" BASIC NOUN PHRASE TESTS"""

# just a single common noun
@pytest.mark.skip(reason="NPs aren't yet supported by our Chinese language generator")
def test_common_noun():
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[situation_object(BALL)]
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("chyou2",)


# a single mass noun. The distinction isn't nearly as salient in Chinese
@pytest.mark.skip(reason="NP's aren't yet supported by our Chinese language generator")
def test_mass_noun():
    situtation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[situation_object(WATER)]
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("shwei3",)


# a single proper noun
@pytest.mark.skip(reason="NP's aren't yet supported by our Chinese language generator")
def test_proper_noun():
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[situation_objects[DAD]]
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("ba4 ba4",)


"""COUNTING NOUN PHRASE TESTS (WITH CLASSIFIERS)"""


# a single object -- note that we don't use yi clf noun yet but this could be implemented later on
@pytest.mark.skip(reason="Classifiers aren't yet supported")
def test_single_item():
    dog = situation_object(DOG)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[dog]
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("gou3",)


# https://github.com/isi-vista/adam/issues/782
# TODO: get classifiers checked by a native speaker upon implementation
# two objects, which can be counted distinctly
@pytest.mark.skip(reason="Classifiers aren't yet supported")
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
@pytest.mark.skip(reason="Many object NPs aren't yet supported")
def test_many_items():
    ball1 = situation_object(BALL, debug_handle="ball1")
    ball2 = situation_object(BALL, debug_handle="ball2")
    ball3 = situation_object(BALL, debug_handle="ball3")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[ball1, ball2, ball3]
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("hen3dwo1", "chyou2")


"""SALIENT VS. NOT SALIENT OBJECTS"""

# two trucks, only one is salient
@pytest.mark.skip(reason="saliency not yet supported")
def test_one_salient_object():
    truck1 = situation_object(TRUCK, debug_handle="truck1")
    truck2 = situation_object(TRUCK, debug_handle="truck2")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[truck1], other_objects=[truck2]
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language((situation, FixedIndexChooser(0)))
    ).as_token_sequence() == ("ka3 che1")


# many objects, only two are salient
@pytest.mark.skip(reason="saliency and classifiers aren't yet supported")
def test_two_salient_objects():
    dog1 = situation_object(DOG, debug_handle="dog1")
    dog2 = situation_object(DOG, debug_handle="dog2")
    dog3 = situation_object(DOG, debug_handle="dog3")
    truck1 = situation_object(TRUCK, debug_handle="truck1")
    truck2 = situation_object(TRUCK, debug_handle="truck2")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dog1, dog2],
        other_objects=[dog3, truck1, truck2],
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("lyang3", "jr3", "gou3")


"""NOUN PHRASES WITH LOCALISER NOMINAL MODIFIERS"""
# TODO: native speaker should check the localisers once they are implemented

# tests mum being next to an object, a relation that is represented with a localiser phrase
@pytest.mark.skip(reason="localisers and NP's aren't yet supported")
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
    assert generated_tokens(situation) == ("ma1 ma1", "dzai4", "nyau3", "pang2 byan1")


# tests mum being under a bird
@pytest.mark.skip(reason="localisers and NP's aren't supported yet")
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
    assert generated_tokens(situation) == ("ma1 ma1", "dzai4", "nyau3", "sya4")


# tests mum being above an object
@pytest.mark.skip(reason="localisers and NP's aren't supported yet")
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
    assert generated_tokens(situation) == ("ma1 ma1", "dzai4", "nyau3", "shang4")


"""BASIC VP TESTING: SV, SVO, and SVIO"""

# test the simple subject-verb phrase "mum eats"
@pytest.mark.skip(reason="VP's aren't yet supported")
def test_simple_subject_verb():
    mum = situation_object(MOM, debug_handle="mum")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mum],
        actions=[Action(action_type=EAT, argument_roles_to_fillers=[(AGENT, mum)])],
    )
    assert generated_tokens(situation) == ("ma1 ma1", "chr1")


# test more complex SVO phrase "mum eats a cookie"
@pytest.mark.skip(reason="SVO structure isn't yet supported")
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
@pytest.mark.skip(reason="SVIO structure isn't yet supported")
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
                argument_roles_to_fillers=[(AGENT, mom), (GOAL, baby), (THEME, cookie)],
            )
        ],
    )
    assert generated_tokens(situation) == (
        "ma1 ma1",
        "gei3",
        "bau3 bau3",
        "chyu1 chi2 bing3",
    )


"""VP's WITH PERSONAL PRONOUNS"""

# test the simple subject-verb phrase "mum eats" with first person
@pytest.mark.skip(reason="pronouns and VP's aren't yet supported")
def test_simple_subject_verb_me():
    mum = situation_object(MOM, debug_handle="mum", properties=[IS_SPEAKER])
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mum],
        actions=[Action(action_type=EAT, argument_roles_to_fillers=[(AGENT, mum)])],
    )
    assert generated_tokens(situation) == ("wo3", "chr1")


# test the simple subject-verb phrase "mum eats" with second person
@pytest.mark.skip(reason="pronouns and VP's aren't yet supported")
def test_simple_subject_verb_you():
    mum = situation_object(MOM, debug_handle="mum", properties=[IS_ADDRESSEE])
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mum],
        actions=[Action(action_type=EAT, argument_roles_to_fillers=[(AGENT, mum)])],
    )
    assert generated_tokens(situation) == ("ni3", "chr1")


# test SVIO transfer of possession with personal pronouns
@pytest.mark.skip(reason="SVIO structure isn't yet supported")
def test_simple_SVIO_transfer():
    mum = situation_object(MOM, debug_handle="mum_subject", properties=[IS_SPEAKER])
    baby = situation_object(BABY, debug_handle="babyIO", properties=[IS_ADDRESSEE])
    cookie = situation_object(COOKIE, debug_handle="cookieDO")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mum, baby, cookie],
        actions=[
            Action(
                action_type=GIVE,
                argument_roles_to_fillers=[(AGENT, mom), (GOAL, baby), (THEME, cookie)],
            )
        ],
    )
    assert generated_tokens(situation) == ("wo3", "gei3", "ni3", "chyu1 chi2 bing3")


# test SVO with action/movement verb
@pytest.mark.skip(reason="SVO structure isn't supported yet")
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


"""VP's WITH LOCALIZERS AND VARIOUS SPEAKERS"""
# TODO: handle zai/dao distinction in generator
# use zai by default and dao with after-action relations based on https://github.com/isi-vista/adam/issues/796
# a list of verbs that currently don't accept goals is at https://github.com/isi-vista/adam/issues/582

# this situation doesn't have any after-action relations so it uses zai, which is valid
@pytest.mark.skip(reason="Localisers aren't yet implemented")
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
                    (AGENT, mom),
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
@pytest.mark.skip(reason="Localisers and dao aren't yet implemented")
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
                    (AGENT, mom),
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
@pytest.mark.skip(reason="pronouns and localisers aren't yet implemented")
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
                    (AGENT, mom),
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
@pytest.mark.skip(reason="We don't handle speaker or localisers yet")
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
                    (AGENT, mom),
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
@pytest.mark.skip(reason="pronouns and localisers aren't yet implemented")
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
                    (AGENT, mom),
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
@pytest.mark.skip(reason="We don't handle speaker or localisers yet")
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
                    (AGENT, mom),
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
@pytest.mark.skip(reason="localizers aren't yet handled")
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
        "syang1",
        "li3",
    )


# an additional test for localizers using a different localizer. This one specifies
# a change in location, so we use dao instead of zai to indicate this change
@pytest.mark.skip(reason="localizers aren't yet handled")
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
        "syang1",
        "li3",
    )


# test with another localiser and first person
@pytest.mark.skip(reason="we haven't implemented pronouns or localisers yet")
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
        "syang1",
        "li3",
    )


# test with another localiser and second person
@pytest.mark.skip(reason="we haven't implemented pronouns or localisers yet")
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
        "syang1",
        "li3",
    )


"""TESTS POSSESSIVE: WO DE AND NI DE"""

# tests the use of the first person possessive, 'wo de'
@pytest.mark.skip("we don't handle possessives or localisers yet")
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
    # TODO: have native speaker check whether wo de or wo is preferred here
    assert generated_tokens(situation) == (
        "wo3",
        "ba3",
        "chyu1 chi2 bing3",
        "fang4",
        "dzai4",
        "wo3",
        "de",
        "syang1",
        "li3",
    )


# this is where another person in the interaction has a box, but we don't note this in our langauge
@pytest.mark.skip("we don't handle possessives or localisers yet")
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
        "syang1",
        "li3",
    )
