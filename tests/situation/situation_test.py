import pytest

from adam.ontology.phase1_ontology import (
    AGENT,
    BALL,
    GAILA_PHASE_1_ONTOLOGY,
    GOAL,
    MOM,
    PUT,
    TABLE,
    THEME,
    DAD,
    BOX,
    GIVE,
)
from adam.ontology.phase1_spatial_relations import (
    Direction,
    EXTERIOR_BUT_IN_CONTACT,
    GRAVITATIONAL_AXIS,
    Region,
)
from adam.situation import Action, SituationObject
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation


def test_mom_put_ball_on_the_table():
    make_mom_put_ball_on_table()


def make_mom_put_ball_on_table():
    mom = SituationObject(ontology_node=MOM)
    ball = SituationObject(ontology_node=BALL)
    table = SituationObject(ontology_node=TABLE)
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
                            direction=Direction(
                                positive=True, relative_to_axis=GRAVITATIONAL_AXIS
                            ),
                        ),
                    ),
                ),
            )
        ],
    )


def test_multiple_trecognized_particulars_banned():
    with pytest.raises(RuntimeError):
        dad_0 = SituationObject(DAD)
        dad_1 = SituationObject(DAD)
        box = SituationObject(BOX)
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
