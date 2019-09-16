from adam.ontology import Region
from adam.ontology.phase1_ontology import (
    AGENT,
    BALL,
    GOAL,
    MOM,
    ON,
    PUT,
    TABLE,
    THEME,
    GAILA_PHASE_1_ONTOLOGY,
)
from adam.ontology.phase1_spatial_relations import EXTERIOR_BUT_IN_CONTACT, Direction
from adam.situation import SituationAction, SituationObject, SituationRelation
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation


def test_mom_put_ball_on_the_table():
    make_mom_put_ball_on_table()


def make_mom_put_ball_on_table():
    mom = SituationObject(ontology_node=MOM)
    ball = SituationObject(ontology_node=BALL)
    table = SituationObject(ontology_node=TABLE)
    return HighLevelSemanticsSituation(
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
                                axis="Vertical axis of table " "relative to earth",
                            ),
                        ),
                    ),
                ),
            )
        ],
    )
