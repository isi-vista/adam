import pytest

from adam.ontology.phase1_ontology import (
    AGENT,
    BALL,
    BOX,
    DAD,
    GAILA_PHASE_1_ONTOLOGY,
    GIVE,
    GOAL,
    MOM,
    PUT,
    TABLE,
    THEME,
)
from adam.ontology.phase1_spatial_relations import (
    EXTERIOR_BUT_IN_CONTACT,
    GRAVITATIONAL_UP,
    Region,
)
from adam.situation import Action
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam_test_utils import situation_object


def test_mom_put_ball_on_the_table():
    make_mom_put_ball_on_table()


def make_mom_put_ball_on_table():
    mom = situation_object(ontology_node=MOM)
    ball = situation_object(ontology_node=BALL)
    table = situation_object(ontology_node=TABLE)
    return HighLevelSemanticsSituation(
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


def test_multiple_trecognized_particulars_banned():
    with pytest.raises(RuntimeError):
        dad_0 = situation_object(DAD)
        dad_1 = situation_object(DAD)
        box = situation_object(BOX)
        HighLevelSemanticsSituation(
            salient_objects=[dad_0, dad_1, box],
            actions=[
                Action(
                    GIVE,
                    argument_roles_to_fillers=[
                        (AGENT, dad_0),
                        (THEME, box),
                        (GOAL, dad_1),
                    ],
                )
            ],
            ontology=GAILA_PHASE_1_ONTOLOGY,
        )
