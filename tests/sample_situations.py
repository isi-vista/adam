from adam.axes import GRAVITATIONAL_AXIS_FUNCTION
from adam.ontology import IN_REGION
from adam.ontology.during import DuringAction
from adam.ontology.phase1_ontology import AGENT, BIRD, FLY, GAILA_PHASE_1_ONTOLOGY, HOUSE
from adam.ontology.phase1_spatial_relations import DISTAL, Direction, Region
from adam.relation import Relation
from adam.situation import Action, SituationObject
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation


def make_bird_flies_over_a_house():
    bird = SituationObject(BIRD)
    house = SituationObject(HOUSE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[bird, house],
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
                                    positive=True,
                                    relative_to_axis=GRAVITATIONAL_AXIS_FUNCTION,
                                ),
                            ),
                        )
                    ]
                ),
            )
        ],
    )
    return situation
