from adam.ontology.phase1_ontology import (
    GAILA_PHASE_1_ONTOLOGY,
    MOM,
    DAD,
    SENTIENT,
    SUPPORTS,
    CONTACTS,
    ABOVE,
    BELOW,
)
from adam.perception import PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
    ObjectPerception,
    HasBinaryProperty,
    RgbColorPerception,
    HasColor,
    RelationPerception,
    IsRecognizedParticular,
)


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
