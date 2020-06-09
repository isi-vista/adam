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

# a single object
# TODO: we need to mark whether we want to specify that there is one object
# TODO: these classifiers should be checked by a native speaker
@pytest.mark.skip(reason="Classifiers aren't yet supported")
def test_single_item():
    dog = situation_object(DOG)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[dog]
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("yi1", "jr3", "gou3")


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
# TODO: this can be implemented before classifier issue is resolved
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
