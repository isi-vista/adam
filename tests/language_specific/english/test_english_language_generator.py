from typing import Tuple

from more_itertools import only, first

from adam.curriculum.phase1_curriculum import (
    make_jump_over_object_template,
    JUMPER,
    JUMPED_OVER,
    _GROUND_OBJECT,
)
from adam.language_specific.english.english_language_generator import (
    PREFER_DITRANSITIVE,
    SimpleRuleBasedEnglishLanguageGenerator,
    USE_ADVERBIAL_PATH_MODIFIER,
)
from adam.language_specific.english.english_phase_1_lexicon import (
    GAILA_PHASE_1_ENGLISH_LEXICON,
)
from adam.ontology import IN_REGION
from adam.ontology.during import DuringAction
from adam.ontology.phase1_ontology import (
    AGENT,
    BABY,
    BALL,
    BIRD,
    BOX,
    COOKIE,
    DAD,
    FALL,
    FLY,
    GAILA_PHASE_1_ONTOLOGY,
    GIVE,
    GOAL,
    GREEN,
    GROUND,
    HAS,
    IS_ADDRESSEE,
    IS_SPEAKER,
    MOM,
    PUSH,
    PUT,
    ROLL,
    TABLE,
    THEME,
    THROW,
    WATER,
    on,
    JUICE,
    CUP,
    DRINK,
    DRINK_CONTAINER_AUX,
    CHAIR,
    PATIENT,
    EAT,
    SIT,
)
from adam.ontology.phase1_spatial_relations import (
    AWAY_FROM,
    DISTAL,
    Direction,
    EXTERIOR_BUT_IN_CONTACT,
    GRAVITATIONAL_AXIS,
    GRAVITATIONAL_UP,
    INTERIOR,
    Region,
    SpatialPath,
)
from adam.random_utils import FixedIndexChooser, RandomChooser
from adam.relation import Relation
from adam.situation import Action, SituationObject
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.situation.templates.phase1_templates import (
    fixed_assignment,
    TemplateVariableAssignment,
)
from tests.sample_situations import make_bird_flies_over_a_house
from tests.situation.situation_test import make_mom_put_ball_on_table

_SIMPLE_GENERATOR = SimpleRuleBasedEnglishLanguageGenerator(
    ontology_lexicon=GAILA_PHASE_1_ENGLISH_LEXICON
)


def test_common_noun():
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[SituationObject(BALL)]
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("a", "ball")


def test_mass_noun():
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[SituationObject(WATER)]
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("water",)


def test_proper_noun():
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[SituationObject(MOM)]
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("Mom",)


def test_one_object():
    box = SituationObject(BOX)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[box]
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("a", "box")


def test_two_objects():
    box_1 = SituationObject(BOX, debug_handle="box_0")
    box_2 = SituationObject(BOX, debug_handle="box_1")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[box_1, box_2]
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("two", "boxes")


def test_many_objects():
    ball_1 = SituationObject(BALL, debug_handle="ball_0")
    ball_2 = SituationObject(BALL, debug_handle="ball_1")
    ball_3 = SituationObject(BALL, debug_handle="ball_2")
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[ball_1, ball_2, ball_3]
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("many", "balls")


def test_simple_verb():
    mom = SituationObject(MOM)
    table = SituationObject(TABLE)
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
    mom = SituationObject(ontology_node=MOM, properties=[IS_SPEAKER])
    ball = SituationObject(ontology_node=BALL)
    table = SituationObject(ontology_node=TABLE)
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
                            direction=Direction(
                                positive=True, relative_to_axis=GRAVITATIONAL_AXIS
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
                            direction=Direction(
                                positive=True, relative_to_axis=GRAVITATIONAL_AXIS
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


def test_situation_with_ground():
    dad = SituationObject(DAD)
    cookie = SituationObject(COOKIE)
    box = SituationObject(BOX)
    ground = SituationObject(GROUND)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[dad, cookie, box, ground],
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
    dad = SituationObject(DAD, properties=[IS_SPEAKER])
    cookie = SituationObject(COOKIE)
    box = SituationObject(BOX)
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
    dad = SituationObject(DAD, properties=[IS_ADDRESSEE])
    cookie = SituationObject(COOKIE)
    box = SituationObject(BOX)
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
    dad = SituationObject(DAD, properties=[IS_SPEAKER])
    cookie = SituationObject(COOKIE)
    box = SituationObject(BOX)
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
    dad = SituationObject(DAD)
    cookie = SituationObject(COOKIE)
    box = SituationObject(BOX)
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
    dad = SituationObject(DAD, properties=[IS_ADDRESSEE])
    cookie = SituationObject(COOKIE)
    box = SituationObject(BOX)
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
    dad = SituationObject(DAD)
    cookie = SituationObject(COOKIE)
    mom = SituationObject(MOM, properties=[IS_SPEAKER])
    box = SituationObject(BOX)
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
    dad = SituationObject(DAD)
    cookie = SituationObject(COOKIE)
    mom = SituationObject(MOM, properties=[IS_SPEAKER])
    box = SituationObject(BOX)
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


def test_dad_has_a_cookie():
    dad = SituationObject(DAD)
    cookie = SituationObject(COOKIE)
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
    ball = SituationObject(BALL, [GREEN])
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
    bird = SituationObject(BIRD)
    table = SituationObject(TABLE)
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
                                direction=Direction(
                                    positive=False, relative_to_axis=GRAVITATIONAL_AXIS
                                ),
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
    mom = SituationObject(MOM)
    ball = SituationObject(BALL)
    table = SituationObject(TABLE)
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
                                direction=Direction(
                                    positive=True, relative_to_axis=GRAVITATIONAL_AXIS
                                ),
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


def test_noun_with_modifier():
    table = SituationObject(TABLE)
    ground = SituationObject(GROUND)

    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[table, ground],
        always_relations=[on(table, ground)],
    )
    assert only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence() == ("a", "table", "on", "the", "ground")


def test_fall_down_syntax_hint():
    ball = SituationObject(BALL)

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


def test_transfer_of_possession():
    mom = SituationObject(MOM)
    baby = SituationObject(BABY)
    cookie = SituationObject(COOKIE)

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


def test_arguments_same_ontology_type():
    baby_0 = SituationObject(BABY)
    baby_1 = SituationObject(BABY)
    cookie = SituationObject(COOKIE)

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
    bird = SituationObject(BIRD)
    dad = SituationObject(DAD)

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


def test_bird_flies_up():
    bird = SituationObject(BIRD)
    ground = SituationObject(GROUND)

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


def test_jumps_over():
    template = make_jump_over_object_template()
    situation = first(
        fixed_assignment(
            template,
            TemplateVariableAssignment(
                object_variables_to_fillers=[
                    (JUMPER, DAD),
                    (JUMPED_OVER, CHAIR),
                    (_GROUND_OBJECT, GROUND),
                ]
            ),
            chooser=RandomChooser.for_seed(0),
            ontology=GAILA_PHASE_1_ONTOLOGY,
        )
    )
    assert generated_tokens(situation) == ("Dad", "jumps", "over", "a", "chair")


def test_mom_drinks_juice():
    mom = SituationObject(MOM)
    juice = SituationObject(JUICE)
    cup = SituationObject(CUP)

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
    mom = SituationObject(MOM)
    cookie = SituationObject(COOKIE)

    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mom, cookie],
        actions=[
            Action(EAT, argument_roles_to_fillers=[(AGENT, mom), (PATIENT, cookie)])
        ],
    )

    assert generated_tokens(situation) == ("Mom", "eats", "a", "cookie")


def test_ball_fell_on_ground():
    ball = SituationObject(BALL)
    ground = SituationObject(GROUND)

    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[ball, ground],
        actions=[Action(FALL, argument_roles_to_fillers=[(THEME, ball)])],
        after_action_relations=[on(ball, ground)],
    )

    assert generated_tokens(situation) == ("a", "ball", "falls", "on", "the", "ground")


def test_mom_sits_on_a_table():
    mom = SituationObject(MOM)
    table = SituationObject(TABLE)

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


def generated_tokens(situation):
    return only(
        _SIMPLE_GENERATOR.generate_language(situation, FixedIndexChooser(0))
    ).as_token_sequence()
