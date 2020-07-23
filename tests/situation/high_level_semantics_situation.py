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
    EXTERIOR_BUT_IN_CONTACT,
    GRAVITATIONAL_UP,
    Region,
)
from adam.situation import Action
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from tests.adam_test_utils import situation_object


def test_no_region_in_gazed_objects():
    putter = situation_object(MOM)
    object_put = situation_object(BALL)
    on_region_object = situation_object(TABLE)
    situation = HighLevelSemanticsSituation(
        salient_objects=[putter, object_put, on_region_object],
        actions=[
            Action(
                PUT,
                argument_roles_to_fillers=[
                    (AGENT, putter),
                    (THEME, object_put),
                    (
                        GOAL,
                        Region(
                            on_region_object,
                            distance=EXTERIOR_BUT_IN_CONTACT,
                            direction=GRAVITATIONAL_UP,
                        ),
                    ),
                ],
            )
        ],
        ontology=GAILA_PHASE_1_ONTOLOGY,
    )

    assert situation.gazed_objects
    for object_ in situation.gazed_objects:
        assert not isinstance(object_, Region)
