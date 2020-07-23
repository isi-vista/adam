from adam.ontology import IN_REGION
from adam.ontology.phase1_ontology import (
    IS_DAD,
    IS_MOM,
    SENTIENT,
    _BALL_SCHEMA,
    _PERSON_SCHEMA,
    _TABLE_SCHEMA,
    _make_cup_schema,
    above,
)
from adam.ontology.phase1_spatial_relations import EXTERIOR_BUT_IN_CONTACT, Region
from adam.perception import ObjectPerception, PerceptualRepresentation
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
    mom = ObjectPerception("mom", axes=_PERSON_SCHEMA.axes.copy())
    dad = ObjectPerception("dad", axes=_PERSON_SCHEMA.axes.copy())

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
    ball = ObjectPerception("ball", _BALL_SCHEMA.geon.copy())

    PerceptualRepresentation.single_frame(
        DevelopmentalPrimitivePerceptionFrame(
            perceived_objects=[ball], property_assertions=[HasColor(ball, red)]
        )
    )


def test_relations():
    # ball on a table
    ball = ObjectPerception("ball", _BALL_SCHEMA.geon.copy())
    table = ObjectPerception("table", axes=_TABLE_SCHEMA.axes.copy())

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
    ball = ObjectPerception("ball", _BALL_SCHEMA.geon.copy())
    cup = ObjectPerception("cup", _make_cup_schema().geon.copy())

    table = ObjectPerception("table", axes=_TABLE_SCHEMA.axes.copy())

    first_frame = DevelopmentalPrimitivePerceptionFrame(
        perceived_objects=[ball, table],
        relations=[
            above(ball, table),
            Relation(IN_REGION, ball, Region(table, distance=EXTERIOR_BUT_IN_CONTACT)),
            Relation(IN_REGION, table, Region(ball, distance=EXTERIOR_BUT_IN_CONTACT)),
        ],
    )

    second_frame = DevelopmentalPrimitivePerceptionFrame(
        perceived_objects=[ball, table, cup], relations=[above(ball, table)]
    )

    diff = diff_primitive_perception_frames(before=first_frame, after=second_frame)
    assert len(diff.removed_relations) == 2
    assert not diff.added_relations
    assert len(diff.added_objects) == 1
    assert not diff.removed_objects
    assert diff.before_axis_info == first_frame.axis_info
    assert diff.after_axis_info == second_frame.axis_info
    assert not diff.added_property_assertions
    assert not diff.removed_property_assertions

    # Reversed
    diff_2 = diff_primitive_perception_frames(before=second_frame, after=first_frame)
    assert len(diff_2.added_relations) == 2
    assert not diff_2.removed_relations
    assert not diff_2.added_objects
    assert len(diff_2.removed_objects) == 1
    assert diff_2.before_axis_info == second_frame.axis_info
    assert diff_2.after_axis_info == first_frame.axis_info
    assert not diff_2.added_property_assertions
    assert not diff_2.removed_property_assertions
