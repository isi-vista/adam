from immutablecollections import immutableset

from adam.axes import HorizontalAxisOfObject, AxesInfo, FacingAddresseeAxis
from adam.ontology import IN_REGION, IS_ADDRESSEE, IS_SPEAKER
from adam.ontology.phase1_ontology import GAILA_PHASE_1_ONTOLOGY, COOKIE, DAD, MOM, CHAIR
from adam.ontology.phase1_spatial_relations import Direction, PROXIMAL, Region
from adam.relation import Relation
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from tests.adam_test_utils import situation_object


def test_axis_facing_addressee():
    # There's a type of situation that's causing `curriculum_to_html` to crash,
    # so I am in the process of replicating that situation here.
    chair = situation_object(CHAIR)
    person_0 = situation_object(MOM, properties=[IS_SPEAKER])
    person_1 = situation_object(DAD, properties=[IS_ADDRESSEE])
    cookie = situation_object(COOKIE)

    object_facing_situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[chair, cookie],
        other_objects=[person_0, person_1],
        always_relations=[
            Relation(
                IN_REGION,
                cookie,
                Region(
                    chair,
                    distance=PROXIMAL,
                    direction=Direction(
                        positive=False, relative_to_axis=FacingAddresseeAxis(chair)
                    ),
                ),
            )
        ],
        axis_info=AxesInfo(
            addressee=person_1,
            axes_facing=[
                (
                    person_1,
                    # TODO: fix this hack
                    HorizontalAxisOfObject(obj, index=1).to_concrete_axis(  # type: ignore
                        None
                    ),
                )
                for obj in [chair, cookie, person_0, person_1]
                if obj.axes
            ],
        ),
    )

    axes_facing_person = object_facing_situation.axis_info.axes_facing[person_1]
    # print("axes_facing_person:", axes_facing_person)  # debugging
    chair_axes = immutableset(chair.axes.all_axes)
    chair_axes_facing_person = axes_facing_person.intersection(chair_axes)
    # print("chair_axes_facing_person:", chair_axes_facing_person)  # debugging
    assert chair_axes_facing_person
    assert len(chair_axes_facing_person) == 1
