from typing import Tuple

import pytest
from more_itertools import only

from adam.axes import HorizontalAxisOfObject, FacingAddresseeAxis, AxesInfo
from adam.language_specific.english.english_language_generator import (
    PREFER_DITRANSITIVE,
    SimpleRuleBasedEnglishLanguageGenerator,
    USE_ADVERBIAL_PATH_MODIFIER,
)
from adam.language_specific.english.english_phase_1_lexicon import (
    GAILA_PHASE_1_ENGLISH_LEXICON,
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

_SIMPLE_GENERATOR = SimpleRuleBasedEnglishLanguageGenerator(
    ontology_lexicon=GAILA_PHASE_1_ENGLISH_LEXICON
)


def test_common_noun():
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[situation_object(BALL)]
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("a", "ball")


def test_mass_noun():
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[situation_object(WATER)]
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("water",)


def test_proper_noun():
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[situation_object(MOM)]
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("Mom",)


def test_one_object():
    box = situation_object(BOX)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[box]
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("a", "box")


def test_two_objects():
    box_1 = situation_object(BOX, debug_handle="box_0")
    box_2 = situation_object(BOX, debug_handle="box_1")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[box_1, box_2]
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("two", "boxes")


def test_two_objects_with_dad():
    table_1 = situation_object(TABLE, debug_handle="table_0")
    table_2 = situation_object(TABLE, debug_handle="table_1")
    dad = situation_object(DAD, debug_handle="dad")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[table_1, dad],
        other_objects=[table_2],
        always_relations=[
            Relation(
                IN_REGION,
                dad,
                Region(
                    table_1,
                    distance=PROXIMAL,
                    direction=Direction(
                        positive=True,
                        relative_to_axis=HorizontalAxisOfObject(table_1, index=0),
                    ),
                ),
            )
        ],
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("Dad", "beside", "a", "table")


def test_many_objects():
    ball_1 = situation_object(BALL, debug_handle="ball_0")
    ball_2 = situation_object(BALL, debug_handle="ball_1")
    ball_3 = situation_object(BALL, debug_handle="ball_2")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[ball_1, ball_2, ball_3]
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("many", "balls")


def test_simple_verb():
    mom = situation_object(MOM)
    table = situation_object(TABLE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mom, table],
        actions=[
            Action(
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
    mom = situation_object(ontology_node=MOM, properties=[IS_SPEAKER])
    ball = situation_object(ontology_node=BALL)
    table = situation_object(ontology_node=TABLE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mom, ball, table],
        actions=[
            Action(
                PUT,
                (
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
                ),
            )
        ],
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("I", "put", "a", "ball", "on", "a", "table")


def test_mom_put_a_ball_on_a_table_using_you():
    mom = situation_object(ontology_node=MOM, properties=[IS_ADDRESSEE])
    ball = situation_object(ontology_node=BALL)
    table = situation_object(ontology_node=TABLE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mom, ball, table],
        actions=[
            Action(
                PUT,
                (
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
                ),
            )
        ],
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("you", "put", "a", "ball", "on", "a", "table")


def test_dad_put_a_cookie_in_a_box():
    dad = situation_object(DAD)
    cookie = situation_object(COOKIE)
    box = situation_object(BOX)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dad, cookie, box],
        actions=[
            Action(
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
    dad = situation_object(DAD, properties=[IS_SPEAKER])
    cookie = situation_object(COOKIE)
    box = situation_object(BOX)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dad, cookie, box],
        actions=[
            Action(
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
    dad = situation_object(DAD, properties=[IS_ADDRESSEE])
    cookie = situation_object(COOKIE)
    box = situation_object(BOX)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dad, cookie, box],
        actions=[
            Action(
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
    dad = situation_object(DAD, properties=[IS_SPEAKER])
    cookie = situation_object(COOKIE)
    box = situation_object(BOX)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dad, cookie, box],
        always_relations=[Relation(HAS, dad, box)],
        actions=[
            Action(
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


def test_dad_put_a_cookie_in_a_box_using_possession():
    dad = situation_object(DAD)
    cookie = situation_object(COOKIE)
    box = situation_object(BOX)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dad, cookie, box],
        always_relations=[Relation(HAS, dad, box)],
        actions=[
            Action(
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


def test_dad_put_a_cookie_in_a_box_using_you_your():
    dad = situation_object(DAD, properties=[IS_ADDRESSEE])
    cookie = situation_object(COOKIE)
    box = situation_object(BOX)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dad, cookie, box],
        always_relations=[Relation(HAS, dad, box)],
        actions=[
            Action(
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


def test_dad_put_a_cookie_in_a_box_using_my_as_mom_speaker():
    dad = situation_object(DAD)
    cookie = situation_object(COOKIE)
    mom = situation_object(MOM, properties=[IS_SPEAKER])
    box = situation_object(BOX)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dad, cookie, box],
        always_relations=[Relation(HAS, mom, box)],
        actions=[
            Action(
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
    ).as_token_sequence() == ("Dad", "puts", "a", "cookie", "in", "my", "box")


def test_i_put_a_cookie_in_dads_box_using_my_as_mom_speaker():
    dad = situation_object(DAD)
    cookie = situation_object(COOKIE)
    mom = situation_object(MOM, properties=[IS_SPEAKER])
    box = situation_object(BOX)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mom, cookie, box, dad],
        always_relations=[Relation(HAS, dad, box)],
        actions=[
            Action(
                PUT,
                (
                    (AGENT, mom),
                    (THEME, cookie),
                    (GOAL, Region(reference_object=box, distance=INTERIOR)),
                ),
            )
        ],
    )

    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("I", "put", "a", "cookie", "in", "Dad", "'s", "box")


def test_i_have_my_ball():
    baby = situation_object(BABY, properties=[IS_SPEAKER])
    ball = situation_object(BALL)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[baby, ball],
        always_relations=[Relation(HAS, baby, ball)],
    )

    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("I", "have", "my", "ball")


def test_dad_has_a_cookie():
    dad = situation_object(DAD)
    cookie = situation_object(COOKIE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dad, cookie],
        always_relations=[Relation(HAS, dad, cookie)],
        actions=[],
    )

    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("Dad", "has", "a", "cookie")


def test_green_ball():
    ball = situation_object(BALL, properties=[GREEN])
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[ball]
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("a", "green", "ball")


def test_path_modifier():
    situation = make_bird_flies_over_a_house()
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("a", "bird", "flies", "over", "a", "house")


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
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("a", "bird", "flies", "under", "a", "table")


def test_path_modifier_on():
    mom = situation_object(MOM)
    ball = situation_object(BALL)
    table = situation_object(TABLE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mom, ball, table],
        actions=[
            Action(
                ROLL,
                argument_roles_to_fillers=[(AGENT, mom), (THEME, ball)],
                during=DuringAction(
                    at_some_point=[
                        Relation(
                            IN_REGION,
                            ball,
                            Region(
                                reference_object=table,
                                distance=EXTERIOR_BUT_IN_CONTACT,
                                direction=GRAVITATIONAL_UP,
                            ),
                        )
                    ]
                ),
            )
        ],
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("Mom", "rolls", "a", "ball", "on", "a", "table")


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
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("a", "baby", "rolls", "a", "cookie", "on", "a", "box")


def test_noun_with_modifier():
    table = situation_object(TABLE)
    ground = situation_object(GROUND)

    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[table, ground],
        always_relations=[on(table, ground)],
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("a", "table", "on", "the", "ground")


def test_fall_down_syntax_hint():
    ball = situation_object(BALL)

    situation_without_modifier = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[ball],
        actions=[Action(FALL, argument_roles_to_fillers=[(THEME, ball)])],
    )

    situation_with_modifier = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[ball],
        actions=[Action(FALL, argument_roles_to_fillers=[(THEME, ball)])],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER],
    )

    assert generated_tokens(situation_without_modifier) == ("a", "ball", "falls")
    assert generated_tokens(situation_with_modifier) == ("a", "ball", "falls", "down")


def test_action_with_advmod_and_preposition():
    mom = situation_object(MOM)
    chair = situation_object(CHAIR)

    situation_with_advmod_and_preposition = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mom, chair],
        actions=[
            Action(
                SIT,
                argument_roles_to_fillers=[
                    (AGENT, mom),
                    (
                        GOAL,
                        Region(
                            chair,
                            direction=GRAVITATIONAL_UP,
                            distance=EXTERIOR_BUT_IN_CONTACT,
                        ),
                    ),
                ],
            )
        ],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER],
    )

    assert generated_tokens(situation_with_advmod_and_preposition) == (
        "Mom",
        "sits",
        "down",
        "on",
        "a",
        "chair",
    )


def test_transfer_of_possession():
    mom = situation_object(MOM)
    baby = situation_object(BABY)
    cookie = situation_object(COOKIE)

    for (action, verb) in ((GIVE, "gives"), (THROW, "throws")):
        for prefer_ditransitive in (True, False):
            syntax_hints = [PREFER_DITRANSITIVE] if prefer_ditransitive else []
            situation = HighLevelSemanticsSituation(
                ontology=GAILA_PHASE_1_ONTOLOGY,
                salient_objects=[mom, baby, cookie],
                actions=[
                    Action(
                        action_type=action,
                        argument_roles_to_fillers=[
                            (AGENT, mom),
                            (GOAL, baby),
                            (THEME, cookie),
                        ],
                    )
                ],
                syntax_hints=syntax_hints,
            )

            reference_tokens: Tuple[str, ...]
            if prefer_ditransitive:
                reference_tokens = ("Mom", verb, "a", "baby", "a", "cookie")
            else:
                reference_tokens = ("Mom", verb, "a", "cookie", "to", "a", "baby")

            assert generated_tokens(situation) == reference_tokens


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
        "a",
        "baby",
        "takes",
        "a",
        "ball",
        "to",
        "a",
        "car",
    )


@pytest.mark.skip(
    "Disabling because BABY is now a recognized particular, "
    "and you can't have multiple recognized particulars in a situation"
)
def test_arguments_same_ontology_type():
    baby_0 = situation_object(BABY)
    baby_1 = situation_object(BABY)
    cookie = situation_object(COOKIE)

    for prefer_ditransitive in (True, False):
        syntax_hints = [PREFER_DITRANSITIVE] if prefer_ditransitive else []
        situation = HighLevelSemanticsSituation(
            ontology=GAILA_PHASE_1_ONTOLOGY,
            salient_objects=[baby_0, baby_1, cookie],
            actions=[
                Action(
                    action_type=GIVE,
                    argument_roles_to_fillers=[
                        (AGENT, baby_0),
                        (GOAL, baby_1),
                        (THEME, cookie),
                    ],
                )
            ],
            syntax_hints=syntax_hints,
        )

        reference_tokens: Tuple[str, ...]
        if prefer_ditransitive:
            reference_tokens = ("a", "baby", "gives", "a", "baby", "a", "cookie")
        else:
            reference_tokens = ("a", "baby", "gives", "a", "cookie", "to", "a", "baby")

        assert generated_tokens(situation) == reference_tokens


def test_bird_flies_over_dad():
    bird = situation_object(BIRD)
    dad = situation_object(DAD)

    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[bird, dad],
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
                                reference_object=dad,
                                distance=DISTAL,
                                direction=GRAVITATIONAL_UP,
                            ),
                        )
                    ]
                ),
            )
        ],
    )

    assert generated_tokens(situation) == ("a", "bird", "flies", "over", "Dad")


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

    assert generated_tokens(situation) == ("a", "bird", "flies", "beside", "a", "car")


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

    assert generated_tokens(situation) == ("a", "bird", "flies", "up")


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
    assert generated_tokens(situation) == ("Dad", "jumps", "up")


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
    assert generated_tokens(situation) == ("Dad", "jumps", "over", "a", "chair")


def test_mom_drinks_juice():
    mom = situation_object(MOM)
    juice = situation_object(JUICE)
    cup = situation_object(CUP)

    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mom, juice],
        actions=[
            Action(
                DRINK,
                argument_roles_to_fillers=[(AGENT, mom), (THEME, juice)],
                auxiliary_variable_bindings=[(DRINK_CONTAINER_AUX, cup)],
            )
        ],
    )

    assert generated_tokens(situation) == ("Mom", "drinks", "juice")


def test_mom_eats_cookie():
    mom = situation_object(MOM)
    cookie = situation_object(COOKIE)

    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mom, cookie],
        actions=[
            Action(EAT, argument_roles_to_fillers=[(AGENT, mom), (PATIENT, cookie)])
        ],
    )

    assert generated_tokens(situation) == ("Mom", "eats", "a", "cookie")


def test_ball_fell_on_ground():
    ball = situation_object(BALL)
    ground = situation_object(GROUND)

    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[ball, ground],
        actions=[Action(FALL, argument_roles_to_fillers=[(THEME, ball)])],
        after_action_relations=[on(ball, ground)],
    )

    assert generated_tokens(situation) == ("a", "ball", "falls", "on", "the", "ground")


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

    assert generated_tokens(situation) == ("Mom", "sits", "on", "a", "table")


def test_you_give_me_a_cookie():
    you = situation_object(DAD, properties=[IS_ADDRESSEE])
    baby = situation_object(BABY, properties=[IS_SPEAKER])
    cookie = situation_object(COOKIE)

    situation_to = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[you, baby, cookie],
        actions=[
            Action(
                GIVE,
                argument_roles_to_fillers=[(AGENT, you), (GOAL, baby), (THEME, cookie)],
            )
        ],
    )

    assert generated_tokens(situation_to) == ("you", "give", "a", "cookie", "to", "me")

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
        "you",
        "give",
        "me",
        "a",
        "cookie",
    )


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
    assert generated_tokens(situation) == ("a", "ball", "beside", "a", "table")


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
    assert generated_tokens(front_situation) == ("a", "box", "in front of", "a", "table")

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
    assert generated_tokens(behind_situation) == ("a", "box", "behind", "a", "table")


def test_to_regions_as_goal():
    goal_object = situation_object(BOX, properties=[HOLLOW])
    assert generated_tokens(
        region_as_goal_situation(Region(goal_object, distance=PROXIMAL), goal_object)
    ) == ("a", "dog", "goes", "to", "a", "box")


def test_in_region_as_goal():
    goal_object = situation_object(BOX, properties=[HOLLOW])
    assert generated_tokens(
        region_as_goal_situation(Region(goal_object, distance=INTERIOR), goal_object)
    ) == ("a", "dog", "goes", "in", "a", "box")


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
    ) == ("a", "dog", "goes", "beside", "a", "box")

    # Beside
    assert generated_tokens(
        region_as_goal_situation(
            Region(
                goal_object,
                distance=PROXIMAL,
                direction=Direction(
                    positive=False,
                    relative_to_axis=HorizontalAxisOfObject(goal_object, index=0),
                ),
            ),
            goal_object,
        )
    ) == ("a", "dog", "goes", "beside", "a", "box")


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
    ) == ("a", "dog", "goes", "behind", "a", "box")


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
    ) == ("a", "dog", "goes", "in front of", "a", "box")


def test_over_region_as_goal():
    goal_object = situation_object(TABLE)
    # Over
    assert generated_tokens(
        region_as_goal_situation(
            Region(goal_object, distance=PROXIMAL, direction=GRAVITATIONAL_UP),
            goal_object,
        )
    ) == ("a", "dog", "goes", "over", "a", "table")


def test_under_region_as_goal():
    goal_object = situation_object(TABLE)
    # Over
    assert generated_tokens(
        region_as_goal_situation(
            Region(goal_object, distance=PROXIMAL, direction=GRAVITATIONAL_DOWN),
            goal_object,
        )
    ) == ("a", "dog", "goes", "under", "a", "table")


def test_region_with_out_addressee():
    agent = situation_object(DOG)
    goal_object = situation_object(BOX, properties=[HOLLOW])
    with pytest.raises(RuntimeError):
        generated_tokens(
            HighLevelSemanticsSituation(
                ontology=GAILA_PHASE_1_ONTOLOGY,
                salient_objects=[agent, goal_object],
                actions=[
                    Action(
                        GO,
                        argument_roles_to_fillers=[
                            (AGENT, agent),
                            (
                                GOAL,
                                Region(
                                    goal_object,
                                    distance=PROXIMAL,
                                    direction=Direction(
                                        positive=True,
                                        relative_to_axis=FacingAddresseeAxis(goal_object),
                                    ),
                                ),
                            ),
                        ],
                    )
                ],
            )
        )


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
    )


def generated_tokens(situation):
    return only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence()
