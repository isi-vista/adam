import pytest
from more_itertools import quantify

from adam.ontology import OntologyNode
from adam.ontology.phase1_ontology import AGENT, BALL, DESTINATION, GAILA_PHASE_1_ONTOLOGY, \
    IS_LEARNER, IS_SPEAKER, JUICE, LEARNER, LIQUID, PERSON, PUT, RED, TABLE, THEME
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
    HasBinaryProperty,
    HasColor,
    PropertyPerception,
)
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator,
    TooManySpeakersException,
)
from adam.random_utils import RandomChooser
from adam.situation import SituationAction, SituationObject
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


def test_person_put_ball_on_table():
    person = SituationObject(ontology_node=PERSON)
    ball = SituationObject(ontology_node=BALL)
    table = SituationObject(ontology_node=TABLE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        objects=[person, ball, table],
        relations=[],
        actions=[
            # What is the best way of representing the destination in the high-level semantics?
            # Here we represent it as indicating a relation which should be true.
            SituationAction(PUT, ((AGENT, person), (THEME, ball), (DESTINATION, table)))
        ],
    )

    perception = _PERCEPTION_GENERATOR.generate_perception(
        situation, chooser=RandomChooser.for_seed(0)
    )
    assert len(perception.frames) == 2
    assert len(perception.frames[0].property_assertions) == 3

    first_frame_relations = perception.frames[0].relations
    second_frame_relations = perception.frames[1].relations

    assert len(first_frame_relations) == 61
    assert len(second_frame_relations) == 62

    assert "smallerThan(ball_0, person_0)" in {
        f"{r.relation_type}({r.arg1}, {r.arg2})" for r in first_frame_relations
    }
    assert "smallerThan(ball_0, person_0)" in {
        f"{r.relation_type}({r.arg1}, {r.arg2})" for r in second_frame_relations
    }
    assert "contacts(ball_0, table_0)" in {
        f"{r.relation_type}({r.arg1}, {r.arg2})" for r in second_frame_relations
    }

def _some_object_has_binary_property(
    perception_frame: DevelopmentalPrimitivePerceptionFrame, query_property: OntologyNode
) -> bool:
    return (
        quantify(
            isinstance(property_assertion, HasBinaryProperty)
            and property_assertion.binary_property == query_property
            for property_assertion in perception_frame.property_assertions
        )
        > 0
    )


def test_speaker_perceivable():
    speaker_situation_perception = _PERCEPTION_GENERATOR.generate_perception(
        HighLevelSemanticsSituation(
            ontology=GAILA_PHASE_1_ONTOLOGY,
            objects=[SituationObject(PERSON, [IS_SPEAKER])],
        ),
        chooser=RandomChooser.for_seed(0),
    )
    assert _some_object_has_binary_property(
        speaker_situation_perception.frames[0], IS_SPEAKER
    )


def test_not_two_speakers():
    with pytest.raises(TooManySpeakersException):
        _PERCEPTION_GENERATOR.generate_perception(
            HighLevelSemanticsSituation(
                ontology=GAILA_PHASE_1_ONTOLOGY,
                objects=[
                    SituationObject(PERSON, [IS_SPEAKER]),
                    SituationObject(PERSON, [IS_SPEAKER]),
                ],
            ),
            chooser=RandomChooser.for_seed(0),
        )


def test_liquid_perceivable():
    juice_perception = _PERCEPTION_GENERATOR.generate_perception(
        HighLevelSemanticsSituation(
            ontology=GAILA_PHASE_1_ONTOLOGY, objects=[SituationObject(JUICE)]
        ),
        chooser=RandomChooser.for_seed(0),
    ).frames[0]
    assert _some_object_has_binary_property(juice_perception, LIQUID)


def test_learner_perceivable():
    learner_perception = _PERCEPTION_GENERATOR.generate_perception(
        HighLevelSemanticsSituation(
            ontology=GAILA_PHASE_1_ONTOLOGY, objects=[SituationObject(LEARNER)]
        ),
        chooser=RandomChooser.for_seed(0),
    ).frames[0]

    assert _some_object_has_binary_property(learner_perception, IS_LEARNER)
