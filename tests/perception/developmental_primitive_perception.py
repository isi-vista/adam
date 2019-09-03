from adam.perception import PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerception,
    DevelopmentalPrimitiveObject,
    SENTIENT,
    HasProperty,
)


def test_recognized_particular():
    # create a simple situation consisting of only Mom and Dad
    mom = DevelopmentalPrimitiveObject("mom")
    dad = DevelopmentalPrimitiveObject("dad")

    PerceptualRepresentation(
        frames=[
            DevelopmentalPrimitivePerception(
                perceived_objects=[mom, dad],
                property_assertions=[
                    HasProperty(mom, SENTIENT),
                    HasProperty(dad, SENTIENT),
                ],
            )
        ]
    )
