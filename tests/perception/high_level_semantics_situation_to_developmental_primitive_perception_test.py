import pytest
from more_itertools import quantify

from adam.ontology import OntologyNode, Region
from adam.ontology.phase1_ontology import (
    AGENT,
    BALL,
    GOAL,
    GAILA_PHASE_1_ONTOLOGY,
    IS_LEARNER,
    IS_SPEAKER,
    JUICE,
    LEARNER,
    LIQUID,
    PERSON,
    PUT,
    RED,
    TABLE,
    THEME,
    ANIMATE, SELF_MOVING, INANIMATE)
from adam.ontology.phase1_spatial_relations import EXTERIOR_BUT_IN_CONTACT, Direction
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
    HasBinaryProperty,
    HasColor,
    PropertyPerception,
    ObjectPerception)
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


def perception_with_handle(frame: DevelopmentalPrimitivePerceptionFrame, handle: str) -> \
        ObjectPerception:
    for object_perception in frame.perceived_objects:
        if object_perception.debug_handle == handle:
            return object_perception
    raise RuntimeError(f"Could not find object perception with handle {handle} "
                       f"in {frame}")


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

    assert person_and_ball_perception.frames[0].relations

    person_perception = perception_with_handle(person_and_ball_perception.frames[0],
                                               "person_0")
    ball_perception = perception_with_handle(person_and_ball_perception.frames[0],
                                             "ball_0")

    assert (set(person_and_ball_perception.frames[0].property_assertions)
            == {HasBinaryProperty(person_perception, ANIMATE),
                HasBinaryProperty(person_perception, SELF_MOVING),
                HasBinaryProperty(ball_perception, INANIMATE)})


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
    frame = person_and_ball_perception.frames[0]

    assert (
        prop_assertion is PropertyPerception for prop_assertion in frame.property_assertions
    )

    person_perception = perception_with_handle(frame, "person_0")
    ball_perception = perception_with_handle(frame, "ball_0")
    assert HasBinaryProperty(person_perception, ANIMATE) in frame.property_assertions
    assert HasBinaryProperty(person_perception, SELF_MOVING) in frame.property_assertions
    assert HasBinaryProperty(ball_perception, INANIMATE) in frame.property_assertions
    assert any(isinstance(property_, HasColor) and property_.perceived_object == ball_perception
               for property_ in frame.property_assertions)


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
            SituationAction(
                PUT,
                (
                    (AGENT, person),
                    (THEME, ball),
                    (
                        GOAL,
                        Region(
                            reference_object=table,
                            distance=EXTERIOR_BUT_IN_CONTACT,
                            direction=Direction(
                                positive=True, relative_to_axis="vertical w.r.t. gravity"
                            ),
                        ),
                    ),
                ),
            )
        ],
    )

    perception = _PERCEPTION_GENERATOR.generate_perception(
        situation, chooser=RandomChooser.for_seed(0)
    )
    assert len(perception.frames) == 2
    first_frame = perception.frames[0]
    person_perception = perception_with_handle(first_frame, "person_0")
    ball_perception = perception_with_handle(first_frame, "ball_0")
    table_perception = perception_with_handle(first_frame, "table_0")
    assert HasBinaryProperty(person_perception, ANIMATE) in first_frame.property_assertions
    assert HasBinaryProperty(person_perception, SELF_MOVING) in first_frame.property_assertions
    assert HasBinaryProperty(ball_perception, INANIMATE) in first_frame.property_assertions
    assert HasBinaryProperty(table_perception, INANIMATE) in first_frame.property_assertions

    first_frame_relations = first_frame.relations
    second_frame_relations = perception.frames[1].relations

    # assert we generate at least some relations in each frame
    assert first_frame_relations
    assert second_frame_relations

    first_frame_relations_strings = {
        f"{r.relation_type}({r.first_slot}, {r.second_slot})" for r in first_frame.relations
    }
    second_frame_relations_strings = {
        f"{r.relation_type}({r.first_slot}, {r.second_slot})" for r in perception.frames[1].relations
    }
    assert "smallerThan(ball_0, person_0)" in first_frame_relations_strings
    assert "partOf(hand_0, person_0)" in first_frame_relations_strings
    assert (
        "in-region(ball_0, Region(reference_object=hand_0, "
        "distance=Distance(name='exterior-but-in-contact'), direction=None))"
        in first_frame_relations_strings
    )

    # continuing relations:
    assert "smallerThan(ball_0, person_0)" in second_frame_relations_strings

    # new relations:
    assert (
        "in-region(ball_0, Region(reference_object=table_0, "
        "distance=Distance(name='exterior-but-in-contact'), "
        "direction=Direction(positive=True, relative_to_axis='vertical w.r.t. gravity')))"
        in second_frame_relations_strings
    )

    # removed relations:
    assert (
        "in-region(ball_0, Region(reference_object=hand_0, "
        "distance=Distance(name='exterior-but-in-contact'), direction=None))"
        not in second_frame_relations_strings
    )


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
