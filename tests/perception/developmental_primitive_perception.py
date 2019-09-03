from adam.perception import PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerception,
    DevelopmentalPrimitiveObject,
    SENTIENT,
    HasProperty,
    Color,
    HasColor,
    SUPPORTS,
    DevelopmentalPrimitiveRelation,
    CONTACTS,
    BELOW,
    ABOVE,
)


def test_recognized_particular():
    # create a simple situation consisting of only Mom and Dad
    mom = DevelopmentalPrimitiveObject("mom")
    dad = DevelopmentalPrimitiveObject("dad")

    PerceptualRepresentation.single_frame(
        DevelopmentalPrimitivePerception(
            perceived_objects=[mom, dad],
            property_assertions=[HasProperty(mom, SENTIENT), HasProperty(dad, SENTIENT)],
        )
    )


def test_color():
    # create a situation with a red ball
    red = Color(255, 0, 0)
    ball = DevelopmentalPrimitiveObject("ball")

    PerceptualRepresentation.single_frame(
        DevelopmentalPrimitivePerception(
            perceived_objects=[ball], property_assertions=[HasColor(ball, red)]
        )
    )


def test_relations():
    # ball on a table
    ball = DevelopmentalPrimitiveObject("ball")
    table = DevelopmentalPrimitiveObject("table")

    PerceptualRepresentation.single_frame(
        DevelopmentalPrimitiveObject(
            perceived_objects=[ball, table],
            relations=[
                DevelopmentalPrimitiveRelation(SUPPORTS, table, ball),
                DevelopmentalPrimitiveRelation(ABOVE, ball, table),
                DevelopmentalPrimitiveObject(BELOW, table, ball),
                DevelopmentalPrimitiveRelation(CONTACTS, ball, table),
                DevelopmentalPrimitiveRelation(CONTACTS, table, ball),
            ],
        )
    )
