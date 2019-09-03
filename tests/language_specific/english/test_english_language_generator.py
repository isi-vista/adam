from more_itertools import only

from adam.language_specific.english.english_phase_1_lexicon import (
    GAILA_PHASE_1_ENGLISH_LEXICON,
)
from adam.language_specific.english.english_syntax import (
    SIMPLE_ENGLISH_DEPENDENCY_TREE_LINEARIZER,
)
from adam.language_specific.english.english_language_generator import (
    SimpleRuleBasedEnglishLanguageGenerator,
)
from adam.ontology.phase1_ontology import (
    GAILA_PHASE_1_ONTOLOGY,
    BALL,
    MOM,
    TABLE,
    PUSH,
    AGENT,
    THEME,
)
from adam.random_utils import FixedIndexChooser
from adam.situation import HighLevelSemanticsSituation, SituationObject, SituationAction
from language_specific.situation.situation_test import make_mom_put_ball_on_table

_SIMPLE_GENERATOR = SimpleRuleBasedEnglishLanguageGenerator(
    ontology_lexicon=GAILA_PHASE_1_ENGLISH_LEXICON)


def test_common_noun():
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, objects=[SituationObject(BALL)]
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("a", "ball")


def test_proper_noun():
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, objects=[SituationObject(MOM)]
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("Mom",)


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
    ).as_token_sequence() == ("Mom", "push", "a", "table")


def test_mom_put_a_ball_on_the_table():
    situation = make_mom_put_ball_on_table()
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("Mom", "put", "a", "ball", "on", "a", "table")
