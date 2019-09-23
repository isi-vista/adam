from adam.ontology.phase1_ontology import (
    AGENT,
    BALL,
    GAILA_PHASE_1_ONTOLOGY,
    GOAL,
    MOM,
    PUT,
    TABLE,
    THEME,
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
