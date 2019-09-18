from more_itertools import only

from adam.language_specific.english.english_language_generator import (
    SimpleRuleBasedEnglishLanguageGenerator,
)
from adam.language_specific.english.english_phase_1_lexicon import (
    GAILA_PHASE_1_ENGLISH_LEXICON,
)
from adam.ontology import Region
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
    COOKIE,
    DAD,
    PUT,
    GOAL,
    IS_SPEAKER,
    GREEN,
    IS_ADDRESSEE,
    PERSON, HAS)
from adam.ontology.phase1_spatial_relations import (
    INTERIOR,
    EXTERIOR_BUT_IN_CONTACT,
    Direction,
)
from adam.random_utils import FixedIndexChooser
from adam.relation import Relation
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
    box_1 = SituationObject(BOX, debug_handle="box_0")
    box_2 = SituationObject(BOX, debug_handle="box_1")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, objects=[box_1, box_2]
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("two", "boxes")


def test_many_objects():
    ball_1 = SituationObject(BALL, debug_handle="ball_0")
    ball_2 = SituationObject(BALL, debug_handle="ball_1")
    ball_3 = SituationObject(BALL, debug_handle="ball_2")
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


def test_mom_put_a_ball_on_a_table():
    situation = make_mom_put_ball_on_table()
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("Mom", "puts", "a", "ball", "on", "a", "table")


def test_mom_put_a_ball_on_a_table_using_i():
    mom = SituationObject(ontology_node=MOM, properties=[IS_SPEAKER])
    ball = SituationObject(ontology_node=BALL)
    table = SituationObject(ontology_node=TABLE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        objects=[mom, ball, table],
        relations=[],
        actions=[
            SituationAction(
                PUT,
                (
                    (AGENT, mom),
                    (THEME, ball),
                    (
                        GOAL,
                        Region(
                            reference_object=table,
                            distance=EXTERIOR_BUT_IN_CONTACT,
                            direction=Direction(
                                positive=True,
                                relative_to_axis="Vertical axis of table "
                                "relative to earth",
                            ),
                        ),
                    ),
                ),
            )
        ],
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("I", "put", "a", "ball", "on", "a", "table")


def test_mom_put_a_ball_on_a_table_using_you():
    mom = SituationObject(ontology_node=MOM, properties=[IS_ADDRESSEE])
    ball = SituationObject(ontology_node=BALL)
    table = SituationObject(ontology_node=TABLE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        objects=[mom, ball, table],
        relations=[],
        actions=[
            SituationAction(
                PUT,
                (
                    (AGENT, mom),
                    (THEME, ball),
                    (
                        GOAL,
                        Region(
                            reference_object=table,
                            distance=EXTERIOR_BUT_IN_CONTACT,
                            direction=Direction(
                                positive=True,
                                relative_to_axis="Vertical axis of table "
                                "relative to earth",
                            ),
                        ),
                    ),
                ),
            )
        ],
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("you", "put", "a", "ball", "on", "a", "table")


def test_dad_put_a_cookie_in_a_box():
    dad = SituationObject(DAD)
    cookie = SituationObject(COOKIE)
    box = SituationObject(BOX)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        objects=[dad, cookie, box],
        relations=[],
        actions=[
            SituationAction(
                PUT,
                (
                    (AGENT, dad),
                    (THEME, cookie),
                    (GOAL, Region(reference_object=box, distance=INTERIOR)),
                ),
            )
        ],
    )

    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("Dad", "puts", "a", "cookie", "in", "a", "box")


def test_dad_put_a_cookie_in_a_box_using_i():
    dad = SituationObject(DAD, properties=[IS_SPEAKER])
    cookie = SituationObject(COOKIE)
    box = SituationObject(BOX)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        objects=[dad, cookie, box],
        relations=[],
        actions=[
            SituationAction(
                PUT,
                (
                    (AGENT, dad),
                    (THEME, cookie),
                    (GOAL, Region(reference_object=box, distance=INTERIOR)),
                ),
            )
        ],
    )

    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("I", "put", "a", "cookie", "in", "a", "box")


def test_dad_put_a_cookie_in_a_box_using_you():
    dad = SituationObject(DAD, properties=[IS_ADDRESSEE])
    cookie = SituationObject(COOKIE)
    box = SituationObject(BOX)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        objects=[dad, cookie, box],
        relations=[],
        actions=[
            SituationAction(
                PUT,
                (
                    (AGENT, dad),
                    (THEME, cookie),
                    (GOAL, Region(reference_object=box, distance=INTERIOR)),
                ),
            )
        ],
    )

    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("you", "put", "a", "cookie", "in", "a", "box")


def test_dad_put_a_cookie_in_a_box_using_my_as_dad_speaker():
    dad = SituationObject(DAD, properties=[IS_SPEAKER])
    cookie = SituationObject(COOKIE)
    box = SituationObject(BOX)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        objects=[dad, cookie, box],
        relations=[Relation(HAS, dad, box)],
        actions=[
            SituationAction(
                PUT,
                (
                    (AGENT, dad),
                    (THEME, cookie),
                    (GOAL, Region(reference_object=box, distance=INTERIOR)),
                ),
            )
        ],
    )

    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("I", "put", "a", "cookie", "in", "my", "box")


def test_dad_put_a_cookie_in_a_box_using_dads():
    dad = SituationObject(DAD)
    cookie = SituationObject(COOKIE)
    box = SituationObject(BOX)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        objects=[dad, cookie, box],
        relations=[Relation(HAS, dad, box)],
        actions=[
            SituationAction(
                PUT,
                (
                    (AGENT, dad),
                    (THEME, cookie),
                    (GOAL, Region(reference_object=box, distance=INTERIOR)),
                ),
            )
        ],
    )

    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("Dad", "puts", "a", "cookie", "in", "Dad\'s", "box")


def test_dad_put_a_cookie_in_a_box_using_you_your():
    dad = SituationObject(DAD, properties=[IS_ADDRESSEE])
    cookie = SituationObject(COOKIE)
    box = SituationObject(BOX)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        objects=[dad, cookie, box],
        relations=[Relation(HAS, dad, box)],
        actions=[
            SituationAction(
                PUT,
                (
                    (AGENT, dad),
                    (THEME, cookie),
                    (GOAL, Region(reference_object=box, distance=INTERIOR)),
                ),
            )
        ],
    )

    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("you", "put", "a", "cookie", "in", "your", "box")


# def test_dad_put_a_cookie_in_a_box_using_my_as_mom_speaker():
#     dad = SituationObject(DAD)
#     cookie = SituationObject(COOKIE)
#     mom = SituationObject(MOM, properties=[IS_SPEAKER])
#     box = SituationObject(BOX)
#     situation = HighLevelSemanticsSituation(
#         ontology=GAILA_PHASE_1_ONTOLOGY,
#         objects=[dad, cookie, box, mom],
#         relations=[Relation(HAS, mom, box)],
#         actions=[
#             SituationAction(
#                 PUT,
#                 (
#                     (AGENT, dad),
#                     (THEME, cookie),
#                     (GOAL, Region(reference_object=box, distance=INTERIOR)),
#                 ),
#             )
#         ],
#     )
#
#     assert only(
#         _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
#     ).as_token_sequence() == ("Dad", "put", "a", "cookie", "in", "my", "box")


def test_dad_put_a_cookie_in_a_box_using_you():
    dad = SituationObject(DAD, properties=[IS_ADDRESSEE])
    cookie = SituationObject(COOKIE)
    box = SituationObject(BOX)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        objects=[dad, cookie, box],
        relations=[],
        actions=[
            SituationAction(
                PUT,
                (
                    (AGENT, dad),
                    (THEME, cookie),
                    (GOAL, Region(reference_object=box, distance=INTERIOR)),
                ),
            )
        ],
    )

    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("you", "put", "a", "cookie", "in", "a", "box")


def test_green_ball():
    ball = SituationObject(BALL, [GREEN])
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, objects=[ball], relations=[], actions=[]
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("a", "green", "ball")
