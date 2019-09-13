from more_itertools import only

from adam.language_specific.english.english_language_generator import (
    SimpleRuleBasedEnglishLanguageGenerator,
)
from adam.language_specific.english.english_phase_1_lexicon import (
    GAILA_PHASE_1_ENGLISH_LEXICON,
)
from adam.ontology.phase1_ontology import (
    AGENT,
    BALL,
    GAILA_PHASE_1_ONTOLOGY,
    MOM,
    PUSH,
    TABLE,
    THEME,
    BOX,
    WATER,
)
from adam.random_utils import FixedIndexChooser
from adam.situation import SituationAction, SituationObject
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from tests.situation.situation_test import make_mom_put_ball_on_table

_SIMPLE_GENERATOR = SimpleRuleBasedEnglishLanguageGenerator(
    ontology_lexicon=GAILA_PHASE_1_ENGLISH_LEXICON
)


def test_common_noun():
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, objects=[SituationObject(BALL)]
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("a", "ball")


def test_mass_noun():
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, objects=[SituationObject(WATER)]
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("water",)


def test_proper_noun():
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, objects=[SituationObject(MOM)]
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("Mom",)


def test_one_object():
    box = SituationObject(BOX)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, objects=[box]
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("a", "box")


def test_two_objects():
    box_1 = SituationObject(BOX)
    box_2 = SituationObject(BOX)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, objects=[box_1, box_2]
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("two", "boxes")


def test_many_objects():
    ball_1 = SituationObject(BALL)
    ball_2 = SituationObject(BALL)
    ball_3 = SituationObject(BALL)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, objects=[ball_1, ball_2, ball_3]
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("many", "balls")


def test_simple_verb():
    mom = SituationObject(MOM)
    table = SituationObject(TABLE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        objects=[mom, table],
        actions=[
            SituationAction(
                action_type=PUSH, argument_roles_to_fillers=[(AGENT, mom), (THEME, table)]
            )
        ],
    )
    # TODO: address morphology to capture verb conjugation here
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("Mom", "pushes", "a", "table")


def test_mom_put_a_ball_on_the_table():
    situation = make_mom_put_ball_on_table()
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("Mom", "puts", "a", "ball", "on", "a", "table")
