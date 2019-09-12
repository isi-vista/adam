from adam.ontology.phase1_ontology import BALL, GAILA_PHASE_1_ONTOLOGY, PERSON, RED
from adam.perception.developmental_primitive_perception import (
    PropertyPerception,
    HasBinaryProperty,
    HasColor,
)
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator,
)
from adam.random_utils import RandomChooser
from adam.situation import SituationObject
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation

_PERCEPTION_GENERATOR = HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator(
    GAILA_PHASE_1_ONTOLOGY
)


def test_person_and_ball():
    person = SituationObject(PERSON)
    ball = SituationObject(BALL)

    person_and_ball_situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, objects=[person, ball]
    )

    person_and_ball_perception = _PERCEPTION_GENERATOR.generate_perception(
        person_and_ball_situation, chooser=RandomChooser.for_seed(0)
    )

    perceived_objects = person_and_ball_perception.frames[0].perceived_objects
    print(perceived_objects)
    object_handles = set(obj.debug_handle for obj in perceived_objects)
    assert len(person_and_ball_perception.frames) == 1
    assert object_handles == {
        "ball_0",
        "person_0",
        "head_0",
        "torso_0",
        "arm_0",
        "armsegment_0",
        "armsegment_1",
        "hand_0",
        "arm_1",
        "armsegment_2",
        "armsegment_3",
        "hand_1",
        "leg_0",
        "leg_1",
    }
    print(person_and_ball_perception.frames[0].relations)
    assert person_and_ball_perception.frames[0].relations
    print(person_and_ball_perception.frames[0].property_assertions)
    assert len(person_and_ball_perception.frames[0].property_assertions) == 2


def test_person_and_ball_color():
    person = SituationObject(PERSON)
    ball = SituationObject(BALL, properties=[RED])

    person_and_ball_situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, objects=[person, ball]
    )

    person_and_ball_perception = _PERCEPTION_GENERATOR.generate_perception(
        person_and_ball_situation, chooser=RandomChooser.for_seed(0)
    )

    assert len(person_and_ball_perception.frames) == 1
    assert len(person_and_ball_perception.frames[0].property_assertions) == 3

    property_assertions = person_and_ball_perception.frames[0].property_assertions
    assert (
        prop_assertion is PropertyPerception for prop_assertion in property_assertions
    )
    # Check binary properties
    assert {
        (prop.perceived_object.debug_handle, prop.binary_property.handle)
        for prop in property_assertions
        if isinstance(prop, HasBinaryProperty)
    } == {("person_0", "animate"), ("ball_0", "inanimate")}
    # Check colors
    assert {
        (
            prop.perceived_object.debug_handle,
            f"#{prop.color.red:02x}{prop.color.green:02x}{prop.color.blue:02x}",
        )
        for prop in property_assertions
        if isinstance(prop, HasColor)
    } == {("ball_0", "#f2003c")}
