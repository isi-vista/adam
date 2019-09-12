from adam.ontology.phase1_ontology import (
    AGENT,
    BALL,
    DESTINATION,
    MOM,
    ON,
    PUT,
    TABLE,
    THEME,
    GAILA_PHASE_1_ONTOLOGY,
)
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
            # What is the best way of representing the destination in the high-level semantics?
            # Here we represent it as indicating a relation which should be true.
            SituationAction(
                PUT,
                (
                    (AGENT, mom),
                    (THEME, ball),
                    (DESTINATION, SituationRelation(ON, ball, table)),
                ),
            )
        ],
    )
