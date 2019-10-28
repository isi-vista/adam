from adam.axes import GRAVITATIONAL_AXIS_FUNCTION
from adam.ontology import IN_REGION
from adam.ontology.during import DuringAction
from adam.ontology.phase1_ontology import AGENT, BIRD, FLY, GAILA_PHASE_1_ONTOLOGY, HOUSE
from adam.ontology.phase1_spatial_relations import DISTAL, Direction, Region
from adam.relation import Relation
from adam.situation import Action
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam_test_utils import situation_object


def make_bird_flies_over_a_house():
    bird = situation_object(BIRD)
    house = situation_object(HOUSE)
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
