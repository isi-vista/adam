from adam.ontology import IN_REGION
from adam.ontology.during import DuringAction
from adam.ontology.phase1_ontology import BIRD, HOUSE, GAILA_PHASE_1_ONTOLOGY, FLY, AGENT
from adam.ontology.phase1_spatial_relations import (
    DISTAL,
    Direction,
    GRAVITATIONAL_AXIS,
    Region,
)
from adam.relation import Relation
from adam.situation import SituationObject, Action
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation


def make_bird_flies_over_a_house():
    bird = SituationObject(BIRD)
    house = SituationObject(HOUSE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        objects=[bird, house],
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
                                reference_object=house,
                                distance=DISTAL,
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
    return situation
