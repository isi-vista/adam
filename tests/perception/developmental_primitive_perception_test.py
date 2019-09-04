from adam.ontology.phase1_ontology import GAILA_PHASE_1_ONTOLOGY, MOM, DAD
from adam.perception import PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
    ObjectPerception,
    SENTIENT,
    HasFlagProperty,
    RgbColorPerception,
    HasColor,
    SUPPORTS,
    RelationPerception,
    CONTACTS,
    BELOW,
    ABOVE,
    IsRecognizedParticular,
)
from adam.perception.perception_frame_difference import diff_primitive_perception_frames


def test_recognized_particular():
    # create a simple situation consisting of only Mom and Dad
    mom = ObjectPerception("mom")
    dad = ObjectPerception("dad")

    PerceptualRepresentation.single_frame(
        DevelopmentalPrimitivePerceptionFrame(
            perceived_objects=[mom, dad],
            property_assertions=[
                HasFlagProperty(mom, SENTIENT),
                HasFlagProperty(dad, SENTIENT),
                IsRecognizedParticular(mom, MOM, ontology=GAILA_PHASE_1_ONTOLOGY),
                IsRecognizedParticular(dad, DAD, ontology=GAILA_PHASE_1_ONTOLOGY),
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
                RelationPerception(SUPPORTS, table, ball),
                RelationPerception(ABOVE, ball, table),
                RelationPerception(BELOW, table, ball),
                RelationPerception(CONTACTS, ball, table),
                RelationPerception(CONTACTS, table, ball),
            ],
        )
    )


def test_difference():
    ball = ObjectPerception("ball")
    table = ObjectPerception("table")

    first_frame = DevelopmentalPrimitivePerceptionFrame(
            perceived_objects=[ball, table],
            relations=[
                RelationPerception(SUPPORTS, table, ball),
                RelationPerception(ABOVE, ball, table),
                RelationPerception(BELOW, table, ball),
                RelationPerception(CONTACTS, ball, table),
                RelationPerception(CONTACTS, table, ball),
            ],
        )

    second_frame = DevelopmentalPrimitivePerceptionFrame(
            perceived_objects=[ball, table],
            relations=[
                RelationPerception(SUPPORTS, table, ball),
                RelationPerception(ABOVE, ball, table),
                RelationPerception(BELOW, table, ball)
            ],
        )

    diff = diff_primitive_perception_frames(before=first_frame, after=second_frame)
    assert len(diff.removed_relations) == 2
