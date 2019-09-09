from adam.ontology.phase1_ontology import BALL, GAILA_PHASE_1_ONTOLOGY, PERSON
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator,
)
from adam.random_utils import RandomChooser
from adam.situation import HighLevelSemanticsSituation, SituationObject

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
        "eye_0",
        "eye_1",
        "mouth_0",
        "nose_0",
        "ear_0",
        "ear_1",
        "torso_0",
        "arm_0",
        "bone_0",
        "bone_1",
        "hand_0",
        "palm_0",
        "finger_0",
        "finger_1",
        "finger_2",
        "finger_3",
        "finger_4",
        "arm_1",
        "bone_2",
        "bone_3",
        "hand_1",
        "palm_1",
        "finger_5",
        "finger_6",
        "finger_7",
        "finger_8",
        "finger_9",
        "leg_0",
        "leg_1",
    }
    print(person_and_ball_perception.frames[0].relations)
    assert person_and_ball_perception.frames[0].relations
    print(person_and_ball_perception.frames[0].property_assertions)
    assert len(person_and_ball_perception.frames[0].property_assertions) == 2
