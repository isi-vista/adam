import pytest
from more_itertools import only, quantify

from adam.axes import HorizontalAxisOfObject
from adam.ontology import IN_REGION, OntologyNode, IS_SPEAKER
from adam.ontology.phase1_ontology import (
    AGENT,
    ANIMATE,
    BALL,
    BABY,
    BLACK,
    BLUE,
    BOX,
    CAUSES_CHANGE,
    COLORS_TO_RGBS,
    COOKIE,
    DAD,
    EAT,
    FALL,
    GAILA_PHASE_1_ONTOLOGY,
    GAZED_AT,
    GOAL,
    GROUND,
    INANIMATE,
    IS_LEARNER,
    JUICE,
    LEARNER,
    LIQUID,
    MOM,
    PART_OF,
    PATIENT,
    PERSON,
    PUT,
    RED,
    SELF_MOVING,
    SMALLER_THAN,
    STATIONARY,
    TABLE,
    THEME,
    TRUCK,
    TWO_DIMENSIONAL,
    UNDERGOES_CHANGE,
    VOLITIONALLY_INVOLVED,
    far,
    near,
    on,
)
from adam.ontology.phase1_spatial_relations import (
    DISTAL,
    EXTERIOR_BUT_IN_CONTACT,
    GRAVITATIONAL_UP,
    INTERIOR,
    Region,
    TOWARD,
    PROXIMAL,
    Direction,
)
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
    HasBinaryProperty,
    HasColor,
    PropertyPerception,
    RgbColorPerception,
)
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator,
)
from adam.random_utils import RandomChooser
from adam.relation import Relation
from adam.situation import Action
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam_test_utils import perception_with_handle, situation_object
from sample_situations import make_bird_flies_over_a_house

_PERCEPTION_GENERATOR = HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator(
    GAILA_PHASE_1_ONTOLOGY
)


def test_person_and_ball():
    person = situation_object(PERSON)
    ball = situation_object(BALL)

    person_and_ball_situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[person, ball]
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
        "(animal) leg_0",
        "leg-segment_0",
        "leg-segment_1",
        "foot_0",
        "(animal) leg_1",
        "leg-segment_2",
        "leg-segment_3",
        "foot_1",
        "the ground",
    }

    assert person_and_ball_perception.frames[0].relations

    person_perception = perception_with_handle(
        person_and_ball_perception.frames[0], "person_0"
    )
    ball_perception = perception_with_handle(
        person_and_ball_perception.frames[0], "ball_0"
    )

    assert set(person_and_ball_perception.frames[0].property_assertions).issuperset(
        {
            HasBinaryProperty(person_perception, ANIMATE),
            HasBinaryProperty(person_perception, SELF_MOVING),
            HasBinaryProperty(ball_perception, INANIMATE),
        }
    )


def test_person_and_ball_color():
    person = situation_object(PERSON)
    ball = situation_object(BALL, properties=[RED])

    person_and_ball_situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[person, ball]
    )

    person_and_ball_perception = _PERCEPTION_GENERATOR.generate_perception(
        person_and_ball_situation, chooser=RandomChooser.for_seed(0)
    )

    assert len(person_and_ball_perception.frames) == 1
    frame = person_and_ball_perception.frames[0]

    assert (
        prop_assertion is PropertyPerception
        for prop_assertion in frame.property_assertions
    )

    person_perception = perception_with_handle(frame, "person_0")
    ball_perception = perception_with_handle(frame, "ball_0")
    assert HasBinaryProperty(person_perception, ANIMATE) in frame.property_assertions
    assert HasBinaryProperty(person_perception, SELF_MOVING) in frame.property_assertions
    assert HasBinaryProperty(ball_perception, INANIMATE) in frame.property_assertions
    assert any(
        isinstance(property_, HasColor) and property_.perceived_object == ball_perception
        for property_ in frame.property_assertions
    )


def test_person_put_ball_on_table():
    person = situation_object(ontology_node=PERSON)
    ball = situation_object(ontology_node=BALL)
    table = situation_object(ontology_node=TABLE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[person, ball, table],
        actions=[
            # What is the best way of representing the destination in the high-level semantics?
            # Here we represent it as indicating a relation which should be true.
            Action(
                PUT,
                (
                    (AGENT, person),
                    (THEME, ball),
                    (
                        GOAL,
                        Region(
                            reference_object=table,
                            distance=EXTERIOR_BUT_IN_CONTACT,
                            direction=GRAVITATIONAL_UP,
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
    hand_perception = perception_with_handle(first_frame, "hand_0")
    assert (
        HasBinaryProperty(person_perception, ANIMATE) in first_frame.property_assertions
    )
    assert (
        HasBinaryProperty(person_perception, SELF_MOVING)
        in first_frame.property_assertions
    )
    assert (
        HasBinaryProperty(ball_perception, INANIMATE) in first_frame.property_assertions
    )
    assert (
        HasBinaryProperty(table_perception, INANIMATE) in first_frame.property_assertions
    )

    first_frame_relations = first_frame.relations
    second_frame_relations = perception.frames[1].relations

    # assert we generate at least some relations in each frame
    assert first_frame_relations
    assert second_frame_relations

    assert (
        Relation(SMALLER_THAN, ball_perception, person_perception)
        in first_frame_relations
    )
    assert Relation(PART_OF, hand_perception, person_perception) in first_frame_relations
    assert (
        Relation(
            IN_REGION,
            ball_perception,
            Region(reference_object=hand_perception, distance=EXTERIOR_BUT_IN_CONTACT),
        )
        in first_frame_relations
    )

    # continuing relations:
    assert (
        Relation(SMALLER_THAN, ball_perception, person_perception)
        in second_frame_relations
    )

    # new relations:
    ball_on_table_relation = Relation(
        IN_REGION,
        ball_perception,
        Region(
            table_perception, distance=EXTERIOR_BUT_IN_CONTACT, direction=GRAVITATIONAL_UP
        ),
    )
    assert ball_on_table_relation in second_frame_relations

    # removed relations:
    assert (
        Relation(
            IN_REGION,
            ball_perception,
            Region(reference_object=hand_perception, distance=EXTERIOR_BUT_IN_CONTACT),
        )
        not in second_frame_relations
    )

    # proto-role properties
    assert (
        HasBinaryProperty(person_perception, VOLITIONALLY_INVOLVED)
        in first_frame.property_assertions
    )
    assert (
        HasBinaryProperty(person_perception, CAUSES_CHANGE)
        in first_frame.property_assertions
    )
    assert (
        HasBinaryProperty(ball_perception, UNDERGOES_CHANGE)
        in first_frame.property_assertions
    )
    assert (
        HasBinaryProperty(table_perception, STATIONARY) in first_frame.property_assertions
    )


def test_relations_between_objects_and_ground():
    # person_put_ball_on_table
    person = situation_object(ontology_node=PERSON)
    ball = situation_object(ontology_node=BALL)
    table = situation_object(ontology_node=TABLE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[person, ball, table],
        actions=[
            # What is the best way of representing the destination in the high-level semantics?
            # Here we represent it as indicating a relation which should be true.
            Action(
                PUT,
                (
                    (AGENT, person),
                    (THEME, ball),
                    (
                        GOAL,
                        Region(
                            reference_object=table,
                            distance=EXTERIOR_BUT_IN_CONTACT,
                            direction=GRAVITATIONAL_UP,
                        ),
                    ),
                ),
            )
        ],
    )

    perception = _PERCEPTION_GENERATOR.generate_perception(
        situation, chooser=RandomChooser.for_seed(0)
    )
    first_frame = perception.frames[0]
    ball_perception = perception_with_handle(first_frame, "ball_0")
    ground_perception = perception_with_handle(first_frame, "the ground")
    table_perception = perception_with_handle(first_frame, "table_0")

    first_frame_relations = first_frame.relations
    second_frame_relations = perception.frames[1].relations

    assert on(ball_perception, ground_perception)[0] in first_frame_relations
    assert on(ball_perception, ground_perception)[0] not in second_frame_relations
    assert on(table_perception, ground_perception)[0] in first_frame_relations
    assert on(table_perception, ground_perception)[0] in second_frame_relations


def test_grounding_of_unsupported_objects():
    box = situation_object(ontology_node=BOX)
    ball = situation_object(ontology_node=BALL)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[box, ball],
        always_relations=[
            Relation(
                IN_REGION,
                ball,
                Region(
                    box,
                    distance=INTERIOR,
                ),
            )
        ],
    )

    perception = _PERCEPTION_GENERATOR.generate_perception(
        situation, chooser=RandomChooser.for_seed(0)
    )
    first_frame = perception.frames[0]
    ball_perception = perception_with_handle(first_frame, "ball_0")
    ground_perception = perception_with_handle(first_frame, "the ground")
    box_perception = perception_with_handle(first_frame, "box_0")

    first_frame_relations = first_frame.relations

    assert on(ball_perception, ground_perception)[0] not in first_frame_relations
    assert on(box_perception, ground_perception)[0] in first_frame_relations


def test_objects_in_something_not_implcitly_grounded():
    box = situation_object(ontology_node=BOX)
    ball = situation_object(ontology_node=BALL)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[box, ball],
        always_relations=[
            Relation(
                IN_REGION,
                ball,
                Region(
                    box,
                    distance=PROXIMAL,
                    direction=Direction(
                        positive=True,
                        relative_to_axis=HorizontalAxisOfObject(box, index=0),
                    ),
                ),
            )
        ],
    )

    perception = _PERCEPTION_GENERATOR.generate_perception(
        situation, chooser=RandomChooser.for_seed(0)
    )
    first_frame = perception.frames[0]
    ball_perception = perception_with_handle(first_frame, "ball_0")
    ground_perception = perception_with_handle(first_frame, "the ground")
    box_perception = perception_with_handle(first_frame, "box_0")

    first_frame_relations = first_frame.relations

    assert on(ball_perception, ground_perception)[0] in first_frame_relations
    assert on(box_perception, ground_perception)[0] in first_frame_relations


def test_dynamic_prepositions_implicit_grounding():
    # person_put_ball_on_table
    truck = situation_object(ontology_node=TRUCK)
    baby = situation_object(ontology_node=BABY)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[baby, truck],
        actions=[Action(FALL,  argument_roles_to_fillers=[(THEME, baby)])],
        after_action_relations=[on(baby, truck)],
    )

    perception = _PERCEPTION_GENERATOR.generate_perception(
        situation, chooser=RandomChooser.for_seed(0)
    )
    first_frame = perception.frames[0]
    baby_perception = perception_with_handle(first_frame, "person_0")
    ground_perception = perception_with_handle(first_frame, "the ground")
    truck_perception = perception_with_handle(first_frame, "truck_0")

    first_frame_relations = first_frame.relations
    second_frame_relations = perception.frames[1].relations

    assert on(baby_perception, truck_perception)[0] not in first_frame_relations
    assert on(baby_perception, truck_perception)[0] in second_frame_relations
    assert on(baby_perception, ground_perception)[0] not in second_frame_relations
    assert on(truck_perception, ground_perception)[0] in first_frame_relations
    assert on(truck_perception, ground_perception)[0] in second_frame_relations


def test_liquid_in_and_out_of_container():
    juice = situation_object(JUICE)
    box = situation_object(ontology_node=BOX)
    table = situation_object(ontology_node=TABLE)
    two_d_situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[juice, table],
        always_relations=[on(juice, table)],
    )
    two_d_perception = _PERCEPTION_GENERATOR.generate_perception(
        two_d_situation, chooser=RandomChooser.for_seed(0)
    )

    three_d_situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[juice, box],
        always_relations=[
            Relation(IN_REGION, juice, Region(box, distance=INTERIOR, direction=None))
        ],
    )
    three_d_perception = _PERCEPTION_GENERATOR.generate_perception(
        three_d_situation, chooser=RandomChooser.for_seed(0)
    )

    two_perceived_objects = two_d_perception.frames[0].perceived_objects
    two_object_handles = set(obj.debug_handle for obj in two_perceived_objects)
    assert all(handle in two_object_handles for handle in {"juice_0", "table_0"})
    three_perceived_objects = three_d_perception.frames[0].perceived_objects
    three_object_handles = set(obj.debug_handle for obj in three_perceived_objects)
    assert all(handle in three_object_handles for handle in {"juice_0", "box_0"})

    assert any(
        isinstance(p, HasBinaryProperty)
        and perception_with_handle(two_d_perception.frames[0], "juice_0")
        == p.perceived_object
        and p.binary_property == TWO_DIMENSIONAL
        for p in two_d_perception.frames[0].property_assertions
    )
    assert not any(
        isinstance(p, HasBinaryProperty)
        and perception_with_handle(three_d_perception.frames[0], "juice_0")
        == p.perceived_object
        and p.binary_property == TWO_DIMENSIONAL
        for p in three_d_perception.frames[0].property_assertions
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
            salient_objects=[situation_object(PERSON, properties=[IS_SPEAKER])],
        ),
        chooser=RandomChooser.for_seed(0),
    )
    assert _some_object_has_binary_property(
        speaker_situation_perception.frames[0], IS_SPEAKER
    )


def test_not_two_speakers():
    with pytest.raises(RuntimeError):
        _PERCEPTION_GENERATOR.generate_perception(
            HighLevelSemanticsSituation(
                ontology=GAILA_PHASE_1_ONTOLOGY,
                salient_objects=[
                    situation_object(PERSON, properties=[IS_SPEAKER]),
                    situation_object(PERSON, properties=[IS_SPEAKER]),
                ],
            ),
            chooser=RandomChooser.for_seed(0),
        )


def test_liquid_perceivable():
    juice_perception = _PERCEPTION_GENERATOR.generate_perception(
        HighLevelSemanticsSituation(
            ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[situation_object(JUICE)]
        ),
        chooser=RandomChooser.for_seed(0),
    ).frames[0]
    assert _some_object_has_binary_property(juice_perception, LIQUID)


def test_learner_perceivable():
    learner_perception = _PERCEPTION_GENERATOR.generate_perception(
        HighLevelSemanticsSituation(
            ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[situation_object(LEARNER)]
        ),
        chooser=RandomChooser.for_seed(0),
    ).frames[0]

    assert _some_object_has_binary_property(learner_perception, IS_LEARNER)


def test_implicit_ground():
    table_perception = _PERCEPTION_GENERATOR.generate_perception(
        HighLevelSemanticsSituation(
            ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[situation_object(TABLE)]
        ),
        chooser=RandomChooser.for_seed(0),
    )

    perceived_objects = table_perception.frames[0].perceived_objects
    object_handles = set(obj.debug_handle for obj in perceived_objects)

    # assert that a "ground" object is perceived
    assert "the ground" in object_handles


def test_explicit_ground():
    ground_perception = _PERCEPTION_GENERATOR.generate_perception(
        HighLevelSemanticsSituation(
            ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[situation_object(GROUND)]
        ),
        chooser=RandomChooser.for_seed(0),
    )

    perceived_objects = ground_perception.frames[0].perceived_objects
    object_handles = set(obj.debug_handle for obj in perceived_objects)

    # assert that a second "ground" object was not generated
    assert object_handles == {"the ground"}


def test_perceive_relations_during():
    learner_perception = _PERCEPTION_GENERATOR.generate_perception(
        make_bird_flies_over_a_house(), chooser=RandomChooser.for_seed(0)
    )
    assert learner_perception.during

    bird = perception_with_handle(learner_perception.frames[0], "bird_0")
    house = perception_with_handle(learner_perception.frames[0], "house_0")

    bird_over_the_house = Relation(
        IN_REGION,
        bird,
        Region(reference_object=house, distance=DISTAL, direction=GRAVITATIONAL_UP),
    )

    assert bird_over_the_house in learner_perception.during.at_some_point


def test_perceive_explicit_relations():
    # we want to test that relations explicitly called out in the situation are perceived.
    # Such relations fall into three buckets:
    # those which hold before an action,
    # those which hold after an action,
    # and those which hold both before and after an action.
    # To test all three of these at once, we use a situation where
    # (a) Mom is putting a ball on a table
    # (b) before the action the ball is far from a box,
    # (c) but after the action the ball is near the box,
    # (d) throughout the action the box is on the table.
    mom = situation_object(MOM)
    ball = situation_object(BALL)
    box = situation_object(BOX)
    table = situation_object(TABLE)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mom, box, ball, table],
        always_relations=[on(ball, table)],
        before_action_relations=[far(ball, box)],
        after_action_relations=[near(ball, box)],
        actions=[
            Action(
                PUT,
                argument_roles_to_fillers=[
                    (AGENT, mom),
                    (THEME, ball),
                    (
                        GOAL,
                        Region(
                            reference_object=table,
                            distance=EXTERIOR_BUT_IN_CONTACT,
                            direction=GRAVITATIONAL_UP,
                        ),
                    ),
                ],
            )
        ],
    )
    perception = _PERCEPTION_GENERATOR.generate_perception(
        situation, chooser=RandomChooser.for_seed(0)
    )

    ball_perception = perception_with_handle(perception.frames[0], "ball_0")
    box_perception = perception_with_handle(perception.frames[0], "box_0")
    table_perception = perception_with_handle(perception.frames[0], "table_0")

    assert only(on(ball_perception, table_perception)) in perception.frames[0].relations
    assert only(on(ball_perception, table_perception)) in perception.frames[0].relations

    assert only(far(ball_perception, box_perception)) in perception.frames[0].relations
    assert (
        only(far(ball_perception, box_perception)) not in perception.frames[1].relations
    )

    assert (
        only(near(ball_perception, box_perception)) not in perception.frames[0].relations
    )
    assert only(near(ball_perception, box_perception)) in perception.frames[1].relations


def test_path_from_action_description():
    ball = situation_object(BALL)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[ball],
        actions=[Action(FALL, argument_roles_to_fillers=[(THEME, ball)])],
    )
    perception = _PERCEPTION_GENERATOR.generate_perception(
        situation, chooser=RandomChooser.for_seed(0)
    )
    ball_perception = perception_with_handle(perception.frames[0], "ball_0")
    ground_perception = perception_with_handle(perception.frames[0], "the ground")
    assert perception.during
    assert perception.during.objects_to_paths
    assert len(perception.during.objects_to_paths) == 1
    path = only(perception.during.objects_to_paths[ball_perception])
    assert path.reference_object == ground_perception
    assert path.operator == TOWARD


def test_gaze_default():
    cookie = situation_object(COOKIE)
    table = situation_object(TABLE)
    dad = situation_object(DAD)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[cookie, table, dad],
        actions=[
            Action(EAT, argument_roles_to_fillers=[(AGENT, dad), (PATIENT, cookie)])
        ],
    )
    perception = _PERCEPTION_GENERATOR.generate_perception(
        situation, chooser=RandomChooser.for_seed(0)
    )
    frame = perception.frames[0]
    cookie_perception = perception_with_handle(frame, "cookie_0")
    dad_perception = perception_with_handle(frame, "dad_0")
    table_perception = perception_with_handle(frame, "table_0")
    assert HasBinaryProperty(cookie_perception, GAZED_AT) in frame.property_assertions
    assert HasBinaryProperty(dad_perception, GAZED_AT) in frame.property_assertions
    # because the table does not occupy a semantic role in the situation
    assert HasBinaryProperty(table_perception, GAZED_AT) not in frame.property_assertions


def test_gaze_specified():
    cookie = situation_object(COOKIE)
    table = situation_object(TABLE)
    dad = situation_object(DAD)
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[cookie, table, dad],
        actions=[
            Action(EAT, argument_roles_to_fillers=[(AGENT, dad), (PATIENT, cookie)])
        ],
        gazed_objects=[cookie],
    )
    perception = _PERCEPTION_GENERATOR.generate_perception(
        situation, chooser=RandomChooser.for_seed(0)
    )
    frame = perception.frames[0]
    cookie_perception = perception_with_handle(frame, "cookie_0")
    dad_perception = perception_with_handle(frame, "dad_0")
    table_perception = perception_with_handle(frame, "table_0")
    # only the cookie is gazed at, because the user said so.
    assert HasBinaryProperty(cookie_perception, GAZED_AT) in frame.property_assertions
    assert HasBinaryProperty(dad_perception, GAZED_AT) not in frame.property_assertions
    assert HasBinaryProperty(table_perception, GAZED_AT) not in frame.property_assertions


def test_colors_across_part_of_relations():
    """
    Intended to test color inheritance across part-of relations
    with objects that have a prototypical color
    """
    learner_perception = _PERCEPTION_GENERATOR.generate_perception(
        HighLevelSemanticsSituation(
            ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[situation_object(TRUCK)]
        ),
        chooser=RandomChooser.for_seed(0),
    )
    frame = learner_perception.frames[0]
    property_assertions = frame.property_assertions

    truck_perception = perception_with_handle(frame, "truck_0")
    flatbed_perception = perception_with_handle(frame, "flatbed_0")
    tire0_perception = perception_with_handle(frame, "tire_0")
    tire1_perception = perception_with_handle(frame, "tire_1")
    tire2_perception = perception_with_handle(frame, "tire_2")
    tire3_perception = perception_with_handle(frame, "tire_3")
    tire4_perception = perception_with_handle(frame, "tire_4")
    tire5_perception = perception_with_handle(frame, "tire_5")
    tire6_perception = perception_with_handle(frame, "tire_6")
    tire7_perception = perception_with_handle(frame, "tire_7")
    blue_options = COLORS_TO_RGBS[BLUE]
    blue_perceptions = [RgbColorPerception(r, g, b) for r, g, b in blue_options]
    red_options = COLORS_TO_RGBS[RED]
    red_perceptions = [RgbColorPerception(r, g, b) for r, g, b in red_options]
    black_options = COLORS_TO_RGBS[BLACK]
    black_perceptions = [RgbColorPerception(r, g, b) for r, g, b in black_options]

    assert any(
        HasColor(truck_perception, blue_perception) in property_assertions
        for blue_perception in blue_perceptions
    ) or any(
        HasColor(truck_perception, red_perception) in property_assertions
        for red_perception in red_perceptions
    )
    assert any(
        HasColor(flatbed_perception, blue_perception) in property_assertions
        for blue_perception in blue_perceptions
    ) or any(
        HasColor(flatbed_perception, red_perception) in property_assertions
        for red_perception in red_perceptions
    )
    assert any(
        (
            HasColor(tire0_perception, black_perception)
            and HasColor(tire1_perception, black_perception)
            and HasColor(tire2_perception, black_perception)
            and HasColor(tire3_perception, black_perception)
            and HasColor(tire4_perception, black_perception)
            and HasColor(tire5_perception, black_perception)
            and HasColor(tire6_perception, black_perception)
            and HasColor(tire7_perception, black_perception)
        )
        in property_assertions
        for black_perception in black_perceptions
    )
