from adam.ontology import IN_REGION
from adam.ontology.phase1_ontology import IS_DAD, IS_MOM, SENTIENT, above
from adam.ontology.phase1_spatial_relations import EXTERIOR_BUT_IN_CONTACT, Region
from adam.perception import PerceptualRepresentation, ObjectPerception
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
    HasBinaryProperty,
    HasColor,
    RgbColorPerception,
)
from adam.perception.perception_frame_difference import diff_primitive_perception_frames
from adam.relation import Relation


def test_recognized_particular():
    # create a simple situation consisting of only Mom and Dad
    mom = ObjectPerception("mom")
    dad = ObjectPerception("dad")

    PerceptualRepresentation.single_frame(
        DevelopmentalPrimitivePerceptionFrame(
            perceived_objects=[mom, dad],
            property_assertions=[
                HasBinaryProperty(mom, SENTIENT),
                HasBinaryProperty(dad, SENTIENT),
                HasBinaryProperty(mom, IS_MOM),
                HasBinaryProperty(dad, IS_DAD),
            ],
        )
    )


def test_color():
    # create a situation with a red ball
    red = RgbColorPerception(255, 0, 0)
    ball = ObjectPerception("ball")

    PerceptualRepresentation.single_frame(
        DevelopmentalPrimitivePerceptionFrame(
            perceived_objects=[ball], property_assertions=[HasColor(ball, red)]
        )
    )


def test_relations():
    # ball on a table
    ball = ObjectPerception("ball")
    table = ObjectPerception("table")

    PerceptualRepresentation.single_frame(
        DevelopmentalPrimitivePerceptionFrame(
            perceived_objects=[ball, table],
            relations=[
                above(ball, table),
                Relation(
                    IN_REGION, ball, Region(table, distance=EXTERIOR_BUT_IN_CONTACT)
                ),
                Relation(
                    IN_REGION, table, Region(ball, distance=EXTERIOR_BUT_IN_CONTACT)
                ),
            ],
        )
    )


def test_difference():
    ball = ObjectPerception("ball")
    table = ObjectPerception("table")

    first_frame = DevelopmentalPrimitivePerceptionFrame(
        perceived_objects=[ball, table],
        relations=[
            above(ball, table),
            Relation(IN_REGION, ball, Region(table, distance=EXTERIOR_BUT_IN_CONTACT)),
            Relation(IN_REGION, table, Region(ball, distance=EXTERIOR_BUT_IN_CONTACT)),
        ],
    )

    second_frame = DevelopmentalPrimitivePerceptionFrame(
        perceived_objects=[ball, table], relations=[above(ball, table)]
    )

    diff = diff_primitive_perception_frames(before=first_frame, after=second_frame)
    assert len(diff.removed_relations) == 2
