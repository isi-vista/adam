"""
The `Ontology` for use in ISI's GAILA Phase 1 effort.

Note that this `Ontology` is only used for training and testing example generation; the learner
has no access to it.

The following will eventually end up here:

- Objects: mommy, daddy, baby, book, house, car, water, ball, juice, cup, box, chair, head,
  milk, hand, dog, truck, door, hat, table, cookie, bird
- Actions/Verbs: go, put, come, take, eat, give, turn, sit, drink, push, fall, throw, move, jump,
  has (possessive), give, roll, fly
- Relations, Modifiers, Function Words: basic color terms (red, blue, green, white, black…), one,
  two, I, me, my, you, your, to, in, on, [beside, behind, in front of, over, under], up, down
"""
from typing import Iterable, Optional, Sequence, Tuple

from immutablecollections import ImmutableDict, immutabledict, immutableset
from more_itertools import flatten

from adam.geon import (
    CIRCULAR,
    CONSTANT,
    Geon,
    IRREGULAR,
    LARGE_TO_SMALL,
    OVALISH,
    RECTANGULAR,
    SMALL_TO_LARGE,
    SMALL_TO_LARGE_TO_SMALL,
    directed,
    straight_up,
    symmetric,
    symmetric_verical,
)
from adam.ontology import (
    ACTION,
    CAN_FILL_TEMPLATE_SLOT,
    IN_REGION,
    OntologyNode,
    PROPERTY,
    RELATION,
    THING,
    minimal_ontology_graph,
)
from adam.ontology.action_description import ActionDescription, ActionDescriptionFrame
from adam.ontology.during import DuringAction
from adam.ontology.ontology import Ontology
from adam.ontology.phase1_size_relationships import build_size_relationships
from adam.ontology.phase1_spatial_relations import (
    AWAY_FROM,
    Axis,
    DISTAL,
    Direction,
    EXTERIOR_BUT_IN_CONTACT,
    FROM,
    GRAVITATIONAL_AXIS,
    GRAVITATIONAL_DOWN,
    GRAVITATIONAL_UP,
    INTERIOR,
    PROXIMAL,
    Region,
    SpatialPath,
    TO,
    TOWARD,
)
from adam.ontology.structural_schema import ObjectStructuralSchema, SubObject
from adam.relation import (
    ObjectT,
    Relation,
    flatten_relations,
    make_dsl_region_relation,
    make_dsl_relation,
    make_opposite_dsl_region_relation,
    make_opposite_dsl_relation,
    make_symmetric_dsl_region_relation,
    negate,
)
from adam.situation import SituationObject

_ontology_graph = minimal_ontology_graph()  # pylint:disable=invalid-name


def subtype(sub: OntologyNode, _super: OntologyNode) -> None:
    _ontology_graph.add_edge(sub, _super)


# Semantic Roles

SEMANTIC_ROLE = OntologyNode("semantic-role")
AGENT = OntologyNode("agent")
subtype(AGENT, SEMANTIC_ROLE)
PATIENT = OntologyNode("patient")
subtype(PATIENT, SEMANTIC_ROLE)
THEME = OntologyNode("theme")
subtype(THEME, SEMANTIC_ROLE)
GOAL = OntologyNode("goal")
subtype(GOAL, SEMANTIC_ROLE)


# these are "properties of properties" (e.g. whether a property is perceivable by the learner)

META_PROPERTY = OntologyNode("meta-property")
PERCEIVABLE = OntologyNode("perceivable")
subtype(PERCEIVABLE, META_PROPERTY)
BINARY = OntologyNode("binary")
subtype(BINARY, META_PROPERTY)

IS_HUMAN = OntologyNode("is-human", [BINARY])
subtype(IS_HUMAN, PROPERTY)

# properties of objects which can be perceived by the learner
PERCEIVABLE_PROPERTY = OntologyNode("perceivable-property", [PERCEIVABLE])
subtype(PERCEIVABLE_PROPERTY, PROPERTY)
SELF_MOVING = OntologyNode("self-moving", [BINARY])
subtype(SELF_MOVING, PERCEIVABLE_PROPERTY)
ANIMATE = OntologyNode("animate", [BINARY])
subtype(ANIMATE, PERCEIVABLE_PROPERTY)
INANIMATE = OntologyNode("inanimate", [BINARY])
subtype(INANIMATE, PERCEIVABLE_PROPERTY)
SENTIENT = OntologyNode("sentient", [BINARY])
subtype(SENTIENT, PERCEIVABLE_PROPERTY)
TWO_DIMENSIONAL = OntologyNode("two-dimensional", [BINARY])
subtype(TWO_DIMENSIONAL, PERCEIVABLE_PROPERTY)
LIQUID = OntologyNode("liquid", [BINARY])
subtype(LIQUID, PERCEIVABLE_PROPERTY)
HOLLOW = OntologyNode("hollow", [BINARY])
"""
Whether an object should be though of as empty on the inside.
In particular, hollow objects may serve as containers.

Jackendoff and Landau argue this should be regarded as a primitive of object perception.
"""
subtype(HOLLOW, PERCEIVABLE_PROPERTY)

RECOGNIZED_PARTICULAR_PROPERTY = OntologyNode("recognized-particular", [BINARY])
"""
Indicates that a property in the ontology indicates the identity of an object
as a known particular object (rather than a class)
which is assumed to be known to the `LanguageLearner`. 
The prototypical cases here are *Mom* and *Dad*.
"""

subtype(RECOGNIZED_PARTICULAR_PROPERTY, PERCEIVABLE_PROPERTY)

IS_SPEAKER = OntologyNode("is-speaker", [BINARY])
"""
Indicates that the marked object is the one who is speaking 
the linguistic description of the situation. 
This will not be present for all situations.
It only makes sense to apply this to sub-types of PERSON,
but this is not currently enforced.
"""
subtype(IS_SPEAKER, PERCEIVABLE_PROPERTY)
IS_ADDRESSEE = OntologyNode("is-addressee", [BINARY])
"""
Indicates that the marked object is the one who is addressed.
This will not be present for all situations.
It only makes sense to apply this to sub-types of PERSON,
but this is not currently enforced. E.g. 'You put the ball on the table.'
"""
subtype(IS_ADDRESSEE, PERCEIVABLE_PROPERTY)
GAZED_AT = OntologyNode("gazed-at", [BINARY])
"""
Indicates the object of the focus of the speaker. This is not currently strictly enforced and is
implicity generated in the perception step if not explicit in a situation.
"""
subtype(GAZED_AT, PERCEIVABLE_PROPERTY)


# Dowty's Proto-Roles: Issue 104; Dotwy, 91, page 572.
# Agent Proto-Roles properties:
VOLITIONALLY_INVOLVED = OntologyNode("volitionally-involved", [BINARY])
subtype(VOLITIONALLY_INVOLVED, PERCEIVABLE_PROPERTY)
SENTIENT_OR_PERCEIVES = OntologyNode("sentient-or-perceives", [BINARY])
subtype(SENTIENT_OR_PERCEIVES, PERCEIVABLE_PROPERTY)
CAUSES_CHANGE = OntologyNode("causes-change", [BINARY])
subtype(CAUSES_CHANGE, PERCEIVABLE_PROPERTY)
MOVES = OntologyNode("moves", [BINARY])
subtype(MOVES, PERCEIVABLE_PROPERTY)

# Patient Proto-Roles:
UNDERGOES_CHANGE = OntologyNode("undergoes-change", [BINARY])
subtype(UNDERGOES_CHANGE, PERCEIVABLE_PROPERTY)
INCREMENTAL_THEME = OntologyNode("incremental-theme", [BINARY])
subtype(INCREMENTAL_THEME, PERCEIVABLE_PROPERTY)
CAUSALLY_AFFECTED = OntologyNode("causally-affected", [BINARY])
subtype(CAUSALLY_AFFECTED, PERCEIVABLE_PROPERTY)
STATIONARY = OntologyNode("stationary", [BINARY])
subtype(STATIONARY, PERCEIVABLE_PROPERTY)


# Properties not perceived by the learner, but useful for situation generation

CAN_MANIPULATE_OBJECTS = OntologyNode("can-manipulate-objects")
subtype(CAN_MANIPULATE_OBJECTS, PROPERTY)
EDIBLE = OntologyNode("edible")
subtype(EDIBLE, PROPERTY)
ROLLABLE = OntologyNode("rollable")
subtype(ROLLABLE, PROPERTY)
CAN_HAVE_THINGS_RESTING_ON_THEM = OntologyNode("can-have-things-on-them")
subtype(CAN_HAVE_THINGS_RESTING_ON_THEM, PROPERTY)
IS_BODY_PART = OntologyNode("is-body-part")
subtype(IS_BODY_PART, PROPERTY)
PERSON_CAN_HAVE = OntologyNode("person-can-have")
subtype(PERSON_CAN_HAVE, PROPERTY)
TRANSFER_OF_POSSESSION = OntologyNode("transfer-of-possession")
subtype(TRANSFER_OF_POSSESSION, PROPERTY)
CAN_JUMP = OntologyNode("can-jump")
subtype(CAN_JUMP, PROPERTY)
CAN_FLY = OntologyNode("can-fly")
subtype(CAN_FLY, PROPERTY)
HAS_SPACE_UNDER = OntologyNode("has-space-under")
subtype(HAS_SPACE_UNDER, PROPERTY)
EDIBLE = OntologyNode("edible")
subtype(EDIBLE, PROPERTY)
CAN_BE_SAT_ON_BY_PEOPLE = OntologyNode("can-be-sat-on")
subtype(CAN_BE_SAT_ON_BY_PEOPLE, PROPERTY)

COLOR = OntologyNode("color")
subtype(COLOR, PERCEIVABLE_PROPERTY)
RED = OntologyNode("red", [CAN_FILL_TEMPLATE_SLOT])
BLUE = OntologyNode("blue", [CAN_FILL_TEMPLATE_SLOT])
GREEN = OntologyNode("green", [CAN_FILL_TEMPLATE_SLOT])
BLACK = OntologyNode("black", [CAN_FILL_TEMPLATE_SLOT])
WHITE = OntologyNode("white", [CAN_FILL_TEMPLATE_SLOT])
LIGHT_BROWN = OntologyNode("light-brown", [CAN_FILL_TEMPLATE_SLOT])
DARK_BROWN = OntologyNode("dark-brown", [CAN_FILL_TEMPLATE_SLOT])
TRANSPARENT = OntologyNode("transparent", [CAN_FILL_TEMPLATE_SLOT])
subtype(RED, COLOR)
subtype(BLUE, COLOR)
subtype(GREEN, COLOR)
subtype(BLACK, COLOR)
subtype(WHITE, COLOR)
subtype(LIGHT_BROWN, COLOR)
subtype(DARK_BROWN, COLOR)
subtype(TRANSPARENT, COLOR)
_RED_HEX = [
    (255, 0, 0),
    (237, 28, 36),
    (196, 2, 51),
    (242, 0, 60),
    (237, 41, 57),
    (238, 32, 77),
]
_BLUE_HEX = [
    (0, 0, 255),
    (51, 51, 153),
    (0, 135, 189),
    (0, 147, 175),
    (0, 24, 168),
    (31, 117, 254),
]
_GREEN_HEX = [(0, 255, 0), (75, 111, 68), (86, 130, 3), (34, 139, 34)]
_BLACK_HEX = [(0, 0, 0), (12, 2, 15), (53, 56, 57), (52, 52, 52)]
_WHITE_HEX = [(255, 255, 255), (248, 248, 255), (245, 245, 245), (254, 254, 250)]
_LIGHT_BROWN_HEX = [(219, 191, 33), (222, 205, 111), (222, 212, 160)]
_DARK_BROWN_HEX = [(110, 95, 19), (105, 88, 6), (87, 76, 26)]

COLORS_TO_RGBS: ImmutableDict[
    OntologyNode, Optional[Sequence[Tuple[int, int, int]]]
] = immutabledict(
    [
        (RED, _RED_HEX),
        (BLUE, _BLUE_HEX),
        (GREEN, _GREEN_HEX),
        (BLACK, _BLACK_HEX),
        (WHITE, _WHITE_HEX),
        (TRANSPARENT, None),
        (LIGHT_BROWN, _LIGHT_BROWN_HEX),
        (DARK_BROWN, _DARK_BROWN_HEX),
    ]
)

# Objects
# Information about the hierarchical structure of objects
# is given at the end of this module because it is so bulky.

INANIMATE_OBJECT = OntologyNode("inanimate-object", inheritable_properties=[INANIMATE])
subtype(INANIMATE_OBJECT, THING)

IS_GROUND = OntologyNode("is-ground")
subtype(IS_GROUND, RECOGNIZED_PARTICULAR_PROPERTY)
GROUND = OntologyNode(
    "ground",
    non_inheritable_properties=[
        IS_GROUND,
        CAN_HAVE_THINGS_RESTING_ON_THEM,
        CAN_BE_SAT_ON_BY_PEOPLE,
    ],
)
subtype(GROUND, INANIMATE_OBJECT)

TABLE = OntologyNode(
    "table",
    [
        CAN_FILL_TEMPLATE_SLOT,
        CAN_HAVE_THINGS_RESTING_ON_THEM,
        HAS_SPACE_UNDER,
        CAN_BE_SAT_ON_BY_PEOPLE,
        LIGHT_BROWN,
        DARK_BROWN,
    ],
)
subtype(TABLE, INANIMATE_OBJECT)
BALL = OntologyNode("ball", [CAN_FILL_TEMPLATE_SLOT, PERSON_CAN_HAVE, ROLLABLE])
subtype(BALL, INANIMATE_OBJECT)
BOOK = OntologyNode(
    "book",
    [
        CAN_FILL_TEMPLATE_SLOT,
        CAN_HAVE_THINGS_RESTING_ON_THEM,
        PERSON_CAN_HAVE,
        RED,
        BLUE,
        GREEN,
    ],
)
subtype(BOOK, INANIMATE_OBJECT)
HOUSE = OntologyNode("house", [HOLLOW, CAN_FILL_TEMPLATE_SLOT])
subtype(HOUSE, INANIMATE_OBJECT)
CAR = OntologyNode(
    "car",
    [
        HOLLOW,
        CAN_FILL_TEMPLATE_SLOT,
        SELF_MOVING,
        CAN_HAVE_THINGS_RESTING_ON_THEM,
        ROLLABLE,
    ],
)
subtype(CAR, INANIMATE_OBJECT)
WATER = OntologyNode(
    "water",
    [LIQUID],
    non_inheritable_properties=[TRANSPARENT, CAN_FILL_TEMPLATE_SLOT, EDIBLE],
)
subtype(WATER, INANIMATE_OBJECT)
JUICE = OntologyNode(
    "juice", [LIQUID], non_inheritable_properties=[RED, CAN_FILL_TEMPLATE_SLOT, EDIBLE]
)
subtype(JUICE, INANIMATE_OBJECT)
CUP = OntologyNode("cup", [HOLLOW, CAN_FILL_TEMPLATE_SLOT, PERSON_CAN_HAVE])
subtype(CUP, INANIMATE_OBJECT)
BOX = OntologyNode(
    "box",
    [HOLLOW, CAN_FILL_TEMPLATE_SLOT, CAN_HAVE_THINGS_RESTING_ON_THEM, PERSON_CAN_HAVE],
)
subtype(BOX, INANIMATE_OBJECT)
CHAIR = OntologyNode(
    "chair",
    [CAN_FILL_TEMPLATE_SLOT, CAN_HAVE_THINGS_RESTING_ON_THEM, CAN_BE_SAT_ON_BY_PEOPLE],
)
subtype(CHAIR, INANIMATE_OBJECT)
# should a HEAD be hollow? We are answering yes for now,
# because food and liquids can enter it,
# but we eventually want something more sophisticated.
HEAD = OntologyNode(
    "head",
    [HOLLOW, CAN_FILL_TEMPLATE_SLOT, CAN_HAVE_THINGS_RESTING_ON_THEM, IS_BODY_PART],
)
subtype(HEAD, INANIMATE_OBJECT)
MILK = OntologyNode(
    "milk", [LIQUID], non_inheritable_properties=[WHITE, CAN_FILL_TEMPLATE_SLOT, EDIBLE]
)
subtype(MILK, INANIMATE_OBJECT)
HAND = OntologyNode(
    "hand", [CAN_MANIPULATE_OBJECTS, CAN_FILL_TEMPLATE_SLOT, IS_BODY_PART]
)
subtype(HAND, INANIMATE_OBJECT)
TRUCK = OntologyNode(
    "truck",
    [
        BLUE,
        RED,
        HOLLOW,
        CAN_FILL_TEMPLATE_SLOT,
        SELF_MOVING,
        CAN_HAVE_THINGS_RESTING_ON_THEM,
    ],
)
subtype(TRUCK, INANIMATE_OBJECT)
DOOR = OntologyNode("door", [CAN_FILL_TEMPLATE_SLOT])
subtype(DOOR, INANIMATE_OBJECT)
HAT = OntologyNode("hat", [CAN_FILL_TEMPLATE_SLOT, PERSON_CAN_HAVE])
subtype(HAT, INANIMATE_OBJECT)
COOKIE = OntologyNode(
    "cookie", [CAN_FILL_TEMPLATE_SLOT, PERSON_CAN_HAVE, ROLLABLE, EDIBLE]
)
subtype(COOKIE, INANIMATE_OBJECT)

PERSON = OntologyNode(
    "person", inheritable_properties=[ANIMATE, SELF_MOVING, CAN_JUMP, IS_HUMAN]
)
subtype(PERSON, THING)
IS_MOM = OntologyNode("is-mom")
subtype(IS_MOM, RECOGNIZED_PARTICULAR_PROPERTY)
MOM = OntologyNode("mom", non_inheritable_properties=[IS_MOM, CAN_FILL_TEMPLATE_SLOT])
subtype(MOM, PERSON)

IS_DAD = OntologyNode("is-dad")
subtype(IS_DAD, RECOGNIZED_PARTICULAR_PROPERTY)
DAD = OntologyNode("dad", non_inheritable_properties=[IS_DAD, CAN_FILL_TEMPLATE_SLOT])
subtype(DAD, PERSON)

BABY = OntologyNode("baby", non_inheritable_properties=[CAN_FILL_TEMPLATE_SLOT])
subtype(BABY, PERSON)

IS_LEARNER = OntologyNode("is-learner")
subtype(IS_LEARNER, RECOGNIZED_PARTICULAR_PROPERTY)
LEARNER = OntologyNode("learner", [IS_LEARNER])
"""
We represent the language learner itself in the situation,
because the size or position of objects relative to the learner itself
may be significant for learning.
"""
subtype(LEARNER, BABY)

NONHUMAN_ANIMAL = OntologyNode("animal", inheritable_properties=[ANIMATE])
subtype(NONHUMAN_ANIMAL, THING)
DOG = OntologyNode("dog", [CAN_FILL_TEMPLATE_SLOT, CAN_JUMP])
subtype(DOG, NONHUMAN_ANIMAL)
BIRD = OntologyNode("bird", [CAN_FILL_TEMPLATE_SLOT, CAN_FLY])
subtype(BIRD, NONHUMAN_ANIMAL)

PHASE_1_CURRICULUM_OBJECTS = immutableset(
    [
        BABY,
        BALL,
        BIRD,
        BOOK,
        BOX,
        CAR,
        CHAIR,
        COOKIE,
        CUP,
        DAD,
        DOG,
        DOOR,
        HAND,
        HAT,
        HEAD,
        HOUSE,
        JUICE,
        MILK,
        MOM,
        TABLE,
        TRUCK,
        WATER,
    ]
)

# Terms below are internal and can only be accessed as parts of other objects
_BODY_PART = OntologyNode("body-part", [IS_BODY_PART])
subtype(_BODY_PART, THING)
_ARM = OntologyNode("arm")
subtype(_ARM, INANIMATE_OBJECT)
_TORSO = OntologyNode("torso")
subtype(_TORSO, _BODY_PART)
_LEG = OntologyNode("leg")
subtype(_LEG, _BODY_PART)
_CHAIR_BACK = OntologyNode("chairback")
subtype(_CHAIR_BACK, INANIMATE_OBJECT)
_CHAIR_SEAT = OntologyNode("chairseat")
subtype(_CHAIR_SEAT, INANIMATE_OBJECT)
_TABLETOP = OntologyNode("tabletop")
subtype(_TABLETOP, INANIMATE_OBJECT)
_TAIL = OntologyNode("tail")
subtype(_TAIL, _BODY_PART)
_WING = OntologyNode("wing")
subtype(_WING, _BODY_PART)
_ARM_SEGMENT = OntologyNode("armsegment")
subtype(_ARM_SEGMENT, _BODY_PART)
_WALL = OntologyNode("wall")
subtype(_WALL, INANIMATE_OBJECT)
_ROOF = OntologyNode("roof")
subtype(_ROOF, INANIMATE_OBJECT)
_TIRE = OntologyNode("tire", [BLACK])
subtype(_TIRE, INANIMATE_OBJECT)
_TRUCK_CAB = OntologyNode("truckcab")
subtype(_TRUCK_CAB, INANIMATE_OBJECT)
_TRAILER = OntologyNode("trailer")
subtype(_TRAILER, INANIMATE_OBJECT)
_FLATBED = OntologyNode("flatbed")
subtype(_FLATBED, INANIMATE_OBJECT)
_BODY = OntologyNode("body")
subtype(_BODY, _BODY_PART)
_DOG_HEAD = OntologyNode("dog-head", [CAN_MANIPULATE_OBJECTS])
subtype(_DOG_HEAD, _BODY_PART)
_BIRD_HEAD = OntologyNode("dog-head", [CAN_MANIPULATE_OBJECTS])
subtype(_BIRD_HEAD, _BODY_PART)

# Verbs

STATE = OntologyNode("state")
CONSUME = OntologyNode("consume")
subtype(CONSUME, ACTION)
PUT = OntologyNode("put")
PUSH = OntologyNode("push")
subtype(PUT, ACTION)
subtype(PUSH, ACTION)
GO = OntologyNode("go")
subtype(GO, ACTION)
COME = OntologyNode("come")
subtype(COME, ACTION)
TAKE = OntologyNode("take")
subtype(TAKE, ACTION)
EAT = OntologyNode("eat")
subtype(EAT, CONSUME)
GIVE = OntologyNode("give", [TRANSFER_OF_POSSESSION])
subtype(GIVE, ACTION)
SPIN = OntologyNode("spin")
subtype(SPIN, ACTION)
SIT = OntologyNode("sit")
subtype(SIT, ACTION)
DRINK = OntologyNode("drink")
subtype(DRINK, CONSUME)
FALL = OntologyNode("fall")
subtype(FALL, ACTION)  # ?
THROW = OntologyNode("throw", [TRANSFER_OF_POSSESSION])
subtype(THROW, ACTION)
MOVE = OntologyNode("move")
subtype(MOVE, ACTION)
JUMP = OntologyNode("jump")
subtype(JUMP, ACTION)
ROLL = OntologyNode("roll")
subtype(ROLL, ACTION)
FLY = OntologyNode("fly")
subtype(FLY, ACTION)


# Relations
# These are used both for situations and in the perceptual representation

SPATIAL_RELATION = OntologyNode("spatial-relation")
subtype(SPATIAL_RELATION, RELATION)


# On is an English-specific bundle of semantics, but that's okay, because this is just for
# data generation, and it will get decomposed before being presented as perceptions to the
# learner.
def _on_region_factory(reference_object: ObjectT) -> Region[ObjectT]:
    return Region(
        reference_object=reference_object,
        distance=EXTERIOR_BUT_IN_CONTACT,
        direction=Direction(positive=True, relative_to_axis=GRAVITATIONAL_AXIS),
    )


on = make_dsl_region_relation(_on_region_factory)  # pylint:disable=invalid-name


def _near_region_factory(reference_object: ObjectT) -> Region[ObjectT]:
    return Region(reference_object=reference_object, distance=PROXIMAL)


near = make_dsl_region_relation(_near_region_factory)  # pylint:disable=invalid-name


def _far_region_factory(reference_object: ObjectT) -> Region[ObjectT]:
    return Region(reference_object=reference_object, distance=DISTAL)


far = make_dsl_region_relation(_far_region_factory)  # pylint:disable=invalid-name


PART_OF = OntologyNode("partOf")
"""
A relation indicating that one object is part of another object.
"""
subtype(PART_OF, RELATION)
partOf = make_dsl_relation(PART_OF)  # pylint:disable=invalid-name

SIZE_RELATION = OntologyNode("size-relation")
subtype(SIZE_RELATION, RELATION)

BIGGER_THAN = OntologyNode("biggerThan")
"""
A relation indicating that one object is bigger than another object.

This is a placeholder for a more sophisticated representation of size:
https://github.com/isi-vista/adam/issues/70
"""
subtype(BIGGER_THAN, SIZE_RELATION)

SMALLER_THAN = OntologyNode("smallerThan")
"""
A relation indicating that one object is smaller than another object.

This is a placeholder for a more sophisticated representation of size:
https://github.com/isi-vista/adam/issues/70
"""
subtype(SMALLER_THAN, SIZE_RELATION)

bigger_than = make_opposite_dsl_relation(  # pylint:disable=invalid-name
    BIGGER_THAN, opposite_type=SMALLER_THAN
)

AXIS_RELATION = OntologyNode("axis-relation")
subtype(AXIS_RELATION, RELATION)

MUCH_BIGGER_THAN = OntologyNode("muchBiggerThan")
"""
A relation indicating one axis of a geon is much bigger than another.
This should only be used for geon axis, relations, not general object relations.
"""
subtype(MUCH_BIGGER_THAN, AXIS_RELATION)

MUCH_SMALLER_THAN = OntologyNode("muchSmallerThan")
"""
A relation indicating one axis of a geon is much smaller than another.
This should only be used for geon axis, relations, not general object relations.
"""
subtype(MUCH_SMALLER_THAN, AXIS_RELATION)

much_bigger_than = make_opposite_dsl_relation(  # pylint:disable=invalid-name
    MUCH_BIGGER_THAN, opposite_type=MUCH_SMALLER_THAN
)


HAS = OntologyNode("has")
subtype(HAS, RELATION)
has = make_dsl_relation(HAS)  # pylint:disable=invalid-name


def _contact_region_factory(reference_object: ObjectT) -> Region[ObjectT]:
    return Region(reference_object=reference_object, distance=EXTERIOR_BUT_IN_CONTACT)


# mypy's reveal_type says the type of "contacts" is
# 'def (Union[ObjectT`-1, typing.Iterable[ObjectT`-1]], Union[ObjectT`-1, typing.Iterable[
# ObjectT`-1]]) -> builtins.tuple[adam.relation.Relation[ObjectT`-1]]'
# but `ObjectT`-1 won't bind, so when called below we get things like
# Argument 2 has incompatible type "SubObject"; expected "Union[ObjectT, Iterable[ObjectT]]"
# For now I'm just suppressing the typing and I'll look more into this later.
contacts = make_symmetric_dsl_region_relation(  # pylint:disable=invalid-name
    _contact_region_factory
)


def _inside_region_factory(reference_object: ObjectT) -> Region[ObjectT]:
    return Region(reference_object=reference_object, distance=INTERIOR)


inside = make_dsl_region_relation(_inside_region_factory)  # pylint:disable=invalid-name


def _above_region_factory(reference_object: ObjectT) -> Region[ObjectT]:
    return Region(
        reference_object=reference_object,
        direction=Direction(positive=True, relative_to_axis=GRAVITATIONAL_AXIS),
    )


def _below_region_factory(reference_object: ObjectT) -> Region[ObjectT]:
    return Region(
        reference_object=reference_object,
        direction=Direction(positive=False, relative_to_axis=GRAVITATIONAL_AXIS),
    )


above = make_opposite_dsl_region_relation(  # pylint:disable=invalid-name
    _above_region_factory, _below_region_factory
)


def _strictly_above_region_factory(reference_object: ObjectT) -> Region[ObjectT]:
    return Region(
        reference_object=reference_object, distance=DISTAL, direction=GRAVITATIONAL_UP
    )


def _strictly_below_region_factory(reference_object: ObjectT) -> Region[ObjectT]:
    return Region(
        reference_object=reference_object, distance=DISTAL, direction=GRAVITATIONAL_DOWN
    )


strictly_above = make_opposite_dsl_region_relation(  # pylint:disable=invalid-name
    _strictly_above_region_factory, _strictly_below_region_factory
)


_GROUND_SCHEMA = ObjectStructuralSchema(ontology_node=GROUND)
_LEARNER_SCHEMA = ObjectStructuralSchema(ontology_node=LEARNER)

# Structural Objects without Sub-Parts which are part of our Phase 1 Vocabulary
# These may need to evolve to reflect the changes for visualization of phase 1


def _make_door_schema() -> ObjectStructuralSchema:
    hinges_to_edge = directed("hinges-to-edge")
    bottom_to_top = directed("bottom-to-top")
    interior_to_exterior = directed("interior-to-exterior")

    return ObjectStructuralSchema(
        ontology_node=DOOR,
        geon=Geon(
            cross_section=RECTANGULAR,
            cross_section_size=CONSTANT,
            generating_axis=interior_to_exterior,
            orienting_axes=[hinges_to_edge, bottom_to_top],
            axis_relations=[
                bigger_than(bottom_to_top, hinges_to_edge),
                much_bigger_than(bottom_to_top, interior_to_exterior),
                bigger_than(hinges_to_edge, interior_to_exterior),
            ],
        ),
    )


_DOOR_SCHEMA = _make_door_schema()


def _make_ball_schema() -> ObjectStructuralSchema:
    generating_axis = symmetric_verical("ball-generating")
    orienting_axis = symmetric("ball-orienting")

    return ObjectStructuralSchema(
        ontology_node=BALL,
        geon=Geon(
            cross_section=CIRCULAR,
            cross_section_size=SMALL_TO_LARGE_TO_SMALL,
            generating_axis=generating_axis,
            orienting_axes=[orienting_axis, orienting_axis],
        ),
    )


_BALL_SCHEMA = _make_ball_schema()


_WATER_SCHEMA = ObjectStructuralSchema(WATER)
_JUICE_SCHEMA = ObjectStructuralSchema(JUICE)


def _make_box_schema() -> ObjectStructuralSchema:
    top_to_bottom = straight_up("top-to-bottom")
    side_to_side = symmetric("side-to-side")

    return ObjectStructuralSchema(
        ontology_node=BOX,
        geon=Geon(
            cross_section=RECTANGULAR,
            cross_section_size=CONSTANT,
            generating_axis=top_to_bottom,
            orienting_axes=[side_to_side, side_to_side],
        ),
    )


_BOX_SCHEMA = _make_box_schema()
_MILK_SCHEMA = ObjectStructuralSchema(MILK)


def _make_hat_schema() -> ObjectStructuralSchema:
    brim_to_top = straight_up("brim-to-top")
    side_to_side = symmetric("side-to-side")

    return ObjectStructuralSchema(
        ontology_node=HAT,
        geon=Geon(
            cross_section=OVALISH,
            cross_section_size=LARGE_TO_SMALL,
            generating_axis=brim_to_top,
            orienting_axes=[side_to_side, side_to_side],
        ),
    )


_HAT_SCHEMA = _make_hat_schema()


def _make_cookie_schema() -> ObjectStructuralSchema:
    bottom_to_top = straight_up("bottom-to-top")
    side_to_side = symmetric("side-to-side")

    return ObjectStructuralSchema(
        ontology_node=COOKIE,
        geon=Geon(
            cross_section=CIRCULAR,
            cross_section_size=CONSTANT,
            generating_axis=bottom_to_top,
            orienting_axes=[side_to_side, side_to_side],
        ),
    )


_COOKIE_SCHEMA = _make_cookie_schema()


def _make_cup_schema() -> ObjectStructuralSchema:
    bottom_to_top = straight_up("bottom-to-top")
    side_to_side = symmetric("side-to-side")

    return ObjectStructuralSchema(
        ontology_node=CUP,
        geon=Geon(
            cross_section=CIRCULAR,
            cross_section_size=SMALL_TO_LARGE,
            generating_axis=bottom_to_top,
            orienting_axes=[side_to_side, side_to_side],
        ),
    )


_CUP_SCHEMA = _make_cup_schema()


def _make_book_schema() -> ObjectStructuralSchema:
    back_cover_to_front_cover = directed("back-cover-to-front-cover")
    spine_to_edges = directed("spine-to-edges")
    edges_to_edges = straight_up("edges-to-edges")

    return ObjectStructuralSchema(
        ontology_node=BOOK,
        geon=Geon(
            cross_section=RECTANGULAR,
            cross_section_size=CONSTANT,
            generating_axis=edges_to_edges,
            orienting_axes=[back_cover_to_front_cover, spine_to_edges],
        ),
    )


_BOOK_SCHEMA = _make_book_schema()


def _make_hand_schema() -> ObjectStructuralSchema:
    wrist_to_fingertips = directed("wrist-to-fingertips")
    thumb_to_pinky = directed("thumb-to-pinky")
    top_to_palm = directed("top-to-palm")

    return ObjectStructuralSchema(
        ontology_node=HAND,
        # we do not currently represent fingers
        geon=Geon(
            cross_section=RECTANGULAR,
            cross_section_size=CONSTANT,
            generating_axis=wrist_to_fingertips,
            orienting_axes=[thumb_to_pinky, top_to_palm],
            axis_relations=[
                bigger_than(wrist_to_fingertips, thumb_to_pinky),
                much_bigger_than(wrist_to_fingertips, top_to_palm),
                much_bigger_than(thumb_to_pinky, top_to_palm),
            ],
        ),
    )


_HAND_SCHEMA = _make_hand_schema()


def _make_head_schema():
    chin_to_scalp = straight_up("chin-to-scalp")
    back_to_front = directed("back-to-front")
    left_to_right = symmetric("left-to-right")
    return ObjectStructuralSchema(
        HEAD,
        geon=Geon(
            cross_section=OVALISH,
            cross_section_size=SMALL_TO_LARGE_TO_SMALL,
            generating_axis=chin_to_scalp,
            orienting_axes=[back_to_front, left_to_right],
            axis_relations=[
                bigger_than(chin_to_scalp, back_to_front),
                bigger_than(chin_to_scalp, left_to_right),
            ],
        ),
    )


_HEAD_SCHEMA = _make_head_schema()


def _make_torso_schema():
    waist_to_shoulders = straight_up("waist-to-shoulders")
    front_to_back = directed("front-to-back")
    left_to_right = symmetric("left-to-right")

    return ObjectStructuralSchema(
        _TORSO,
        geon=Geon(
            cross_section=OVALISH,
            cross_section_size=CONSTANT,
            generating_axis=waist_to_shoulders,
            orienting_axes=[front_to_back, left_to_right],
            axis_relations=[
                bigger_than(waist_to_shoulders, left_to_right),
                much_bigger_than(waist_to_shoulders, front_to_back),
                much_bigger_than(left_to_right, front_to_back),
            ],
        ),
    )


_DOG_HEAD_SCHEMA = ObjectStructuralSchema(_DOG_HEAD)
_BIRD_HEAD_SCHEMA = ObjectStructuralSchema(_BIRD_HEAD)

# Hierarchical structure of objects
_TORSO_SCHEMA = _make_torso_schema()


def _make_human_leg_schema():
    hip_to_foot = directed("hip-to-foot")
    diameter_0 = symmetric("diameter_0")
    diameter_1 = symmetric("diameter_1")

    return ObjectStructuralSchema(
        _LEG,
        geon=Geon(
            cross_section=OVALISH,
            cross_section_size=CONSTANT,
            generating_axis=hip_to_foot,
            orienting_axes=[diameter_0, diameter_1],
            axis_relations=[
                much_bigger_than(hip_to_foot, diameter_0),
                much_bigger_than(hip_to_foot, diameter_1),
            ],
        ),
    )


# TODO: we shouldn't share a leg schema between humans and tables
# https://github.com/isi-vista/adam/issues/265
_LEG_SCHEMA = _make_human_leg_schema()


def _make_chair_back_schema() -> ObjectStructuralSchema:
    bottom_to_top = straight_up("bottom-to-top")
    front_to_back = directed("front-to-back")
    side_to_side = directed("side-to-side")

    return ObjectStructuralSchema(
        ontology_node=_CHAIR_BACK,
        geon=Geon(
            cross_section=IRREGULAR,
            cross_section_size=CONSTANT,
            generating_axis=front_to_back,
            orienting_axes=[bottom_to_top, side_to_side],
            axis_relations=[
                bigger_than(bottom_to_top, front_to_back),
                bigger_than(bottom_to_top, side_to_side),
            ],
        ),
    )


_CHAIRBACK_SCHEMA = _make_chair_back_schema()


def _make_chair_seat_schema() -> ObjectStructuralSchema:
    bottom_to_top = straight_up("bottom-to-top")
    front_edge_to_back_edge = directed("front-to-back")
    side_to_side = directed("side-to-side")

    return ObjectStructuralSchema(
        ontology_node=_CHAIR_SEAT,
        geon=Geon(
            cross_section=OVALISH,
            cross_section_size=CONSTANT,
            generating_axis=bottom_to_top,
            orienting_axes=[front_edge_to_back_edge, side_to_side],
        ),
    )


_CHAIR_SEAT_SCHEMA = _make_chair_seat_schema()


def _make_table_top_schema() -> ObjectStructuralSchema:
    bottom_to_top = straight_up("bottom-to-top")
    side_to_side = directed("side-to-side")
    front_to_back = directed("front-to-back")

    return ObjectStructuralSchema(
        ontology_node=_TABLETOP,
        geon=Geon(
            cross_section=RECTANGULAR,
            cross_section_size=CONSTANT,
            generating_axis=bottom_to_top,
            orienting_axes=[side_to_side, front_to_back],
            axis_relations=[
                bigger_than(side_to_side, bottom_to_top),
                bigger_than(front_to_back, bottom_to_top),
            ],
        ),
    )


_TABLETOP_SCHEMA = _make_table_top_schema()


def _make_tail_schema() -> ObjectStructuralSchema:
    edge_to_tip = directed("edge-to-tip")
    side_to_side = symmetric("side-to-side")

    return ObjectStructuralSchema(
        ontology_node=_TAIL,
        geon=Geon(
            cross_section=OVALISH,
            cross_section_size=LARGE_TO_SMALL,
            generating_axis=edge_to_tip,
            orienting_axes=[side_to_side, side_to_side],
            axis_relations=[bigger_than(edge_to_tip, side_to_side)],
        ),
    )


_TAIL_SCHEMA = _make_tail_schema()


def _make_wing_schema() -> ObjectStructuralSchema:
    edge_to_tip = directed("edge-to-tip")
    bottom_to_top = straight_up("bottom-to-top")
    front_to_back = directed("front-to-back")

    return ObjectStructuralSchema(
        ontology_node=_WING,
        geon=Geon(
            cross_section=IRREGULAR,
            cross_section_size=CONSTANT,
            generating_axis=edge_to_tip,
            orienting_axes=[bottom_to_top, front_to_back],
            axis_relations=[
                bigger_than(edge_to_tip, bottom_to_top),
                bigger_than(front_to_back, bottom_to_top),
            ],
        ),
    )


_WING_SCHEMA = _make_wing_schema()


def _make_roof_schema() -> ObjectStructuralSchema:
    bottom_to_shingles = straight_up("bottom-to-shingles")
    front_to_back = directed("front-to-back")
    side_to_side = directed("side-to-side")

    return ObjectStructuralSchema(
        ontology_node=_ROOF,
        geon=Geon(
            cross_section=RECTANGULAR,
            cross_section_size=LARGE_TO_SMALL,
            generating_axis=bottom_to_shingles,
            orienting_axes=[front_to_back, side_to_side],
        ),
    )


_ROOF_SCHEMA = _make_roof_schema()


def _make_wall_schema() -> ObjectStructuralSchema:
    bottom_to_top = straight_up("bottom-to-top")
    edge_to_edge = directed("edge-to-edge")
    face_to_face = directed("face-to-face")

    return ObjectStructuralSchema(
        ontology_node=_WALL,
        geon=Geon(
            cross_section=RECTANGULAR,
            cross_section_size=CONSTANT,
            generating_axis=bottom_to_top,
            orienting_axes=[edge_to_edge, face_to_face],
            axis_relations=[
                bigger_than(edge_to_edge, face_to_face),
                bigger_than(bottom_to_top, face_to_face),
            ],
        ),
    )


_WALL_SCHEMA = _make_wall_schema()


def _make_tire_schema() -> ObjectStructuralSchema:
    front_to_back = directed("front-to-back")
    tread_to_tread = symmetric("tread-to-tread")

    return ObjectStructuralSchema(
        ontology_node=_TIRE,
        geon=Geon(
            cross_section=CIRCULAR,
            cross_section_size=CONSTANT,
            generating_axis=front_to_back,
            orienting_axes=[tread_to_tread, tread_to_tread],
        ),
    )


_TIRE_SCHEMA = _make_tire_schema()


def _make_flat_bed_schema() -> ObjectStructuralSchema:
    bottom_to_bed = straight_up("bottom-to-bed")
    front_to_back = directed("front-to-back")
    side_to_side = directed("side-to-side")

    return ObjectStructuralSchema(
        ontology_node=_FLATBED,
        geon=Geon(
            cross_section=RECTANGULAR,
            cross_section_size=CONSTANT,
            generating_axis=bottom_to_bed,
            orienting_axes=[front_to_back, side_to_side],
            axis_relations=[
                bigger_than(front_to_back, side_to_side),
                bigger_than(front_to_back, bottom_to_bed),
                bigger_than(side_to_side, bottom_to_bed),
            ],
        ),
    )


_FLATBED_SCHEMA = _make_flat_bed_schema()


def _make_body_schema() -> ObjectStructuralSchema:
    bottom_to_top = straight_up("bottom-to-top")
    front_to_back = directed("front-to-back")
    side_to_side = directed("side-to-side")

    return ObjectStructuralSchema(
        ontology_node=_BODY,
        geon=Geon(
            cross_section=IRREGULAR,
            cross_section_size=CONSTANT,
            generating_axis=bottom_to_top,
            orienting_axes=[front_to_back, side_to_side],
        ),
    )


_BODY_SCHEMA = _make_body_schema()


def _make_human_arm_segment():
    upper_to_lower = directed("upper-to-lower")
    diameter_0 = symmetric("diameter_0")
    diameter_1 = symmetric("diameter_1")

    return ObjectStructuralSchema(
        _ARM_SEGMENT,
        geon=Geon(
            cross_section=OVALISH,
            cross_section_size=CONSTANT,
            generating_axis=upper_to_lower,
            orienting_axes=[diameter_0, diameter_1],
            axis_relations=[
                much_bigger_than(upper_to_lower, diameter_0),
                much_bigger_than(upper_to_lower, diameter_1),
            ],
        ),
    )


_ARM_SEGMENT_SCHEMA = _make_human_arm_segment()

# schemata describing the sub-object structural nature of a Human Arm
_ARM_SCHEMA_HAND = SubObject(_HAND_SCHEMA)
_ARM_SCHEMA_UPPER = SubObject(
    _ARM_SEGMENT_SCHEMA
)  # Is that the correct sub-object we want?
_ARM_SCHEMA_LOWER = SubObject(_ARM_SEGMENT_SCHEMA)

_ARM_SCHEMA = ObjectStructuralSchema(
    ontology_node=_ARM,
    sub_objects=[_ARM_SCHEMA_HAND, _ARM_SCHEMA_LOWER, _ARM_SCHEMA_UPPER],
    sub_object_relations=flatten_relations(
        [contacts([_ARM_SCHEMA_UPPER, _ARM_SCHEMA_HAND], _ARM_SCHEMA_LOWER)]
    ),
)

# schemata describing the sub-object structural nature of a Person
_PERSON_SCHEMA_HEAD = SubObject(_HEAD_SCHEMA)
_PERSON_SCHEMA_TORSO = SubObject(_TORSO_SCHEMA)
_PERSON_SCHEMA_LEFT_ARM = SubObject(_ARM_SCHEMA)
_PERSON_SCHEMA_RIGHT_ARM = SubObject(_ARM_SCHEMA)
_PERSON_SCHEMA_LEFT_LEG = SubObject(_LEG_SCHEMA)
_PERSON_SCHEMA_RIGHT_LEG = SubObject(_LEG_SCHEMA)

_PERSON_SCHEMA_APPENDAGES = [
    _PERSON_SCHEMA_LEFT_ARM,
    _PERSON_SCHEMA_LEFT_LEG,
    _PERSON_SCHEMA_RIGHT_ARM,
    _PERSON_SCHEMA_RIGHT_LEG,
    _PERSON_SCHEMA_HEAD,
]
_PERSON_SCHEMA = ObjectStructuralSchema(
    ontology_node=PERSON,
    sub_objects=[
        _PERSON_SCHEMA_HEAD,
        _PERSON_SCHEMA_TORSO,
        _PERSON_SCHEMA_LEFT_ARM,
        _PERSON_SCHEMA_RIGHT_ARM,
        _PERSON_SCHEMA_LEFT_LEG,
        _PERSON_SCHEMA_RIGHT_LEG,
    ],
    sub_object_relations=flatten_relations(
        [
            above(_PERSON_SCHEMA_HEAD, _PERSON_SCHEMA_TORSO),
            bigger_than(_PERSON_SCHEMA_TORSO, _PERSON_SCHEMA_HEAD),
            contacts(_PERSON_SCHEMA_TORSO, _PERSON_SCHEMA_APPENDAGES),
        ]
    ),
)


# schemata describing the sub-object structural nature of a Chair
_CHAIR_SCHEMA_BACK = SubObject(_CHAIRBACK_SCHEMA)
_CHAIR_SCHEMA_LEG_1 = SubObject(_LEG_SCHEMA)
_CHAIR_SCHEMA_LEG_2 = SubObject(_LEG_SCHEMA)
_CHAIR_SCHEMA_LEG_3 = SubObject(_LEG_SCHEMA)
_CHAIR_SCHEMA_LEG_4 = SubObject(_LEG_SCHEMA)
_CHAIR_SCHEMA_SEAT = SubObject(_CHAIR_SEAT_SCHEMA)
_CHAIR_LEGS = [
    _CHAIR_SCHEMA_LEG_1,
    _CHAIR_SCHEMA_LEG_2,
    _CHAIR_SCHEMA_LEG_3,
    _CHAIR_SCHEMA_LEG_4,
]

_CHAIR_SCHEMA = ObjectStructuralSchema(
    CHAIR,
    sub_objects=[
        _CHAIR_SCHEMA_BACK,
        _CHAIR_SCHEMA_SEAT,
        _CHAIR_SCHEMA_LEG_1,
        _CHAIR_SCHEMA_LEG_2,
        _CHAIR_SCHEMA_LEG_3,
        _CHAIR_SCHEMA_LEG_4,
    ],
    sub_object_relations=flatten_relations(
        [
            contacts(_CHAIR_LEGS, _CHAIR_SCHEMA_SEAT),
            above(_CHAIR_SCHEMA_SEAT, _CHAIR_LEGS),
            contacts(_CHAIR_SCHEMA_BACK, _CHAIR_SCHEMA_SEAT),
            above(_CHAIR_SCHEMA_BACK, _CHAIR_SCHEMA_SEAT),
        ]
    ),
)

# schemata describing the sub-object structural nature of a Table
_TABLE_SCHEMA_LEG_1 = SubObject(_LEG_SCHEMA)
_TABLE_SCHEMA_LEG_2 = SubObject(_LEG_SCHEMA)
_TABLE_SCHEMA_LEG_3 = SubObject(_LEG_SCHEMA)
_TABLE_SCHEMA_LEG_4 = SubObject(_LEG_SCHEMA)
_TABLE_SCHEMA_TABLETOP = SubObject(_TABLETOP_SCHEMA)
_TABLE_LEGS = [
    _TABLE_SCHEMA_LEG_1,
    _TABLE_SCHEMA_LEG_2,
    _TABLE_SCHEMA_LEG_3,
    _TABLE_SCHEMA_LEG_4,
]

_TABLE_SCHEMA = ObjectStructuralSchema(
    TABLE,
    sub_objects=[
        _TABLE_SCHEMA_LEG_1,
        _TABLE_SCHEMA_LEG_2,
        _TABLE_SCHEMA_LEG_3,
        _TABLE_SCHEMA_LEG_4,
        _TABLE_SCHEMA_TABLETOP,
    ],
    sub_object_relations=flatten_relations(
        [
            # Relationship of tabletop to the legs
            contacts(_TABLE_SCHEMA_TABLETOP, _TABLE_LEGS),
            above(_TABLE_SCHEMA_TABLETOP, _TABLE_LEGS),
        ]
    ),
)

# schemata describing the sub-object structural nature of a dog
_DOG_SCHEMA_LEG_1 = SubObject(_LEG_SCHEMA)
_DOG_SCHEMA_LEG_2 = SubObject(_LEG_SCHEMA)
_DOG_SCHEMA_LEG_3 = SubObject(_LEG_SCHEMA)
_DOG_SCHEMA_LEG_4 = SubObject(_LEG_SCHEMA)
_DOG_SCHEMA_TORSO = SubObject(_TORSO_SCHEMA)
_DOG_SCHEMA_HEAD = SubObject(_DOG_HEAD_SCHEMA)
_DOG_SCHEMA_TAIL = SubObject(_TAIL_SCHEMA)

_DOG_LEGS = [_DOG_SCHEMA_LEG_1, _DOG_SCHEMA_LEG_2, _DOG_SCHEMA_LEG_3, _DOG_SCHEMA_LEG_4]

_DOG_APPENDAGES = [
    _DOG_SCHEMA_LEG_1,
    _DOG_SCHEMA_LEG_2,
    _DOG_SCHEMA_LEG_3,
    _DOG_SCHEMA_LEG_4,
    _DOG_SCHEMA_HEAD,
    _DOG_SCHEMA_TAIL,
]

_DOG_SCHEMA = ObjectStructuralSchema(
    DOG,
    sub_objects=[
        _DOG_SCHEMA_HEAD,
        _DOG_SCHEMA_TORSO,
        _DOG_SCHEMA_TAIL,
        _DOG_SCHEMA_LEG_1,
        _DOG_SCHEMA_LEG_2,
        _DOG_SCHEMA_LEG_3,
        _DOG_SCHEMA_LEG_4,
    ],
    sub_object_relations=flatten_relations(
        [
            contacts(_DOG_SCHEMA_TORSO, _DOG_APPENDAGES),
            above(_DOG_SCHEMA_HEAD, _DOG_SCHEMA_TORSO),
            above(_DOG_SCHEMA_TORSO, _DOG_LEGS),
            bigger_than(_DOG_SCHEMA_TORSO, _DOG_SCHEMA_TAIL),
        ]
    ),
)

# schemata describing the sub-object structural nature of a bird
_BIRD_SCHEMA_HEAD = SubObject(_BIRD_HEAD_SCHEMA)
_BIRD_SCHEMA_TORSO = SubObject(_TORSO_SCHEMA)
_BIRD_SCHEMA_LEFT_LEG = SubObject(_LEG_SCHEMA)
_BIRD_SCHEMA_RIGHT_LEG = SubObject(_LEG_SCHEMA)
_BIRD_SCHEMA_TAIL = SubObject(_TAIL_SCHEMA)
_BIRD_SCHEMA_LEFT_WING = SubObject(_WING_SCHEMA)
_BIRD_SCHEMA_RIGHT_WING = SubObject(_WING_SCHEMA)
_BIRD_LEGS = [_BIRD_SCHEMA_LEFT_LEG, _BIRD_SCHEMA_RIGHT_LEG]
_BIRD_WINGS = [_BIRD_SCHEMA_LEFT_WING, _BIRD_SCHEMA_RIGHT_WING]
_BIRD_APPENDAGES = flatten(
    [_BIRD_LEGS, _BIRD_WINGS, [_BIRD_SCHEMA_HEAD, _BIRD_SCHEMA_TAIL]]
)

# Bird designed with a Robin or similar garden bird in mind
_BIRD_SCHEMA = ObjectStructuralSchema(
    BIRD,
    sub_objects=[
        _BIRD_SCHEMA_HEAD,
        _BIRD_SCHEMA_TORSO,
        _BIRD_SCHEMA_LEFT_LEG,
        _BIRD_SCHEMA_RIGHT_LEG,
        _BIRD_SCHEMA_LEFT_WING,
        _BIRD_SCHEMA_RIGHT_WING,
        _BIRD_SCHEMA_TAIL,
    ],
    sub_object_relations=flatten_relations(
        [
            contacts(_BIRD_SCHEMA_TORSO, _BIRD_APPENDAGES),
            above(_BIRD_SCHEMA_HEAD, _BIRD_SCHEMA_TORSO),
            above(_BIRD_SCHEMA_TORSO, _BIRD_LEGS),
            bigger_than(_BIRD_SCHEMA_TORSO, _BIRD_SCHEMA_HEAD),
            bigger_than(_BIRD_SCHEMA_TORSO, _BIRD_LEGS),
        ]
    ),
)

# schemata describing the sub-object structural nature of a house
_HOUSE_SCHEMA_ROOF = SubObject(_ROOF_SCHEMA)
_HOUSE_SCHEMA_GROUND_FLOOR = SubObject(_WALL_SCHEMA)

# House modeled after a simple 1 story home as commonly seen in child's books
# Stick example below -- ASCII art perhaps isn't the best demonstration form
#      / \
#    /     \
#  /         \
# /           \
# -------------
# | []  _  [] |
# [----| |----]
_HOUSE_SCHEMA = ObjectStructuralSchema(
    HOUSE,
    sub_objects=[_HOUSE_SCHEMA_ROOF, _HOUSE_SCHEMA_GROUND_FLOOR],
    sub_object_relations=flatten_relations(
        [
            contacts(_HOUSE_SCHEMA_ROOF, _HOUSE_SCHEMA_GROUND_FLOOR),
            above(_HOUSE_SCHEMA_ROOF, _HOUSE_SCHEMA_GROUND_FLOOR),
        ]
    ),
)

# schemata describing the sub-object structural nature of a car
_CAR_SCHEMA_FRONT_LEFT_TIRE = SubObject(_TIRE_SCHEMA)
_CAR_SCHEMA_FRONT_RIGHT_TIRE = SubObject(_TIRE_SCHEMA)
_CAR_SCHEMA_REAR_LEFT_TIRE = SubObject(_TIRE_SCHEMA)
_CAR_SCHEMA_REAR_RIGHT_TIRE = SubObject(_TIRE_SCHEMA)
_CAR_SCHEMA_BODY = SubObject(_BODY_SCHEMA)
_CAR_SCHEMA_TIRES = [
    _CAR_SCHEMA_FRONT_LEFT_TIRE,
    _CAR_SCHEMA_FRONT_RIGHT_TIRE,
    _CAR_SCHEMA_REAR_LEFT_TIRE,
    _CAR_SCHEMA_REAR_RIGHT_TIRE,
]

# Improve Car Stuctural Schema once surfaces are introduced
# Git Issue: https://github.com/isi-vista/adam/issues/69
_CAR_SCHEMA = ObjectStructuralSchema(
    CAR,
    sub_objects=[
        _CAR_SCHEMA_FRONT_LEFT_TIRE,
        _CAR_SCHEMA_FRONT_RIGHT_TIRE,
        _CAR_SCHEMA_REAR_LEFT_TIRE,
        _CAR_SCHEMA_REAR_RIGHT_TIRE,
        _CAR_SCHEMA_BODY,
    ],
    sub_object_relations=flatten_relations(
        [contacts(_CAR_SCHEMA_TIRES, _CAR_SCHEMA_BODY)]
    ),
)

# schemata describing the sub-object structural nature of a truck cab
_TRUCK_CAB_TIRE_1 = SubObject(_TIRE_SCHEMA)
_TRUCK_CAB_TIRE_2 = SubObject(_TIRE_SCHEMA)
_TRUCK_CAB_TIRE_3 = SubObject(_TIRE_SCHEMA)
_TRUCK_CAB_TIRE_4 = SubObject(_TIRE_SCHEMA)
_TRUCK_CAB_BODY = SubObject(_BODY_SCHEMA)

_TRUCK_CAB_TIRES = [
    _TRUCK_CAB_TIRE_1,
    _TRUCK_CAB_TIRE_2,
    _TRUCK_CAB_TIRE_3,
    _TRUCK_CAB_TIRE_4,
]

_TRUCK_CAB_SCHEMA = ObjectStructuralSchema(
    _TRUCK_CAB,
    sub_objects=[
        _TRUCK_CAB_TIRE_1,
        _TRUCK_CAB_TIRE_2,
        _TRUCK_CAB_TIRE_3,
        _TRUCK_CAB_TIRE_4,
        _TRUCK_CAB_BODY,
    ],
    sub_object_relations=flatten_relations(
        [
            above(_TRUCK_CAB_BODY, _TRUCK_CAB_TIRES),
            contacts(_TRUCK_CAB_BODY, _TRUCK_CAB_TIRES),
        ]
    ),
)

# schemata describing the sub-object structural nature of a truck trailer
_TRUCK_TRAILER_TIRE_1 = SubObject(_TIRE_SCHEMA)
_TRUCK_TRAILER_TIRE_2 = SubObject(_TIRE_SCHEMA)
_TRUCK_TRAILER_TIRE_3 = SubObject(_TIRE_SCHEMA)
_TRUCK_TRAILER_TIRE_4 = SubObject(_TIRE_SCHEMA)
_TRUCK_TRAILER_FLATBED = SubObject(_FLATBED_SCHEMA)
_TRUCK_TRAILER_TIRES = [
    _TRUCK_TRAILER_TIRE_1,
    _TRUCK_TRAILER_TIRE_2,
    _TRUCK_TRAILER_TIRE_3,
    _TRUCK_TRAILER_TIRE_4,
]

_TRUCK_TRAILER_SCHEMA = ObjectStructuralSchema(
    _TRAILER,
    sub_objects=[
        _TRUCK_TRAILER_TIRE_1,
        _TRUCK_TRAILER_TIRE_2,
        _TRUCK_TRAILER_TIRE_3,
        _TRUCK_TRAILER_TIRE_4,
        _TRUCK_TRAILER_FLATBED,
    ],
    sub_object_relations=flatten_relations(
        [
            contacts(_TRUCK_TRAILER_FLATBED, _TRUCK_TRAILER_TIRES),
            above(_TRUCK_TRAILER_FLATBED, _TRUCK_TRAILER_TIRES),
            bigger_than(_TRUCK_TRAILER_FLATBED, _TRUCK_TRAILER_TIRES),
        ]
    ),
)

# Truck in mind is a Semi Trailer with flat bed trailer
# Schemata describing the sub-object structural nature of a truck
_TRUCK_SCHEMA_CAB = SubObject(_TRUCK_CAB_SCHEMA)
_TRUCK_SCHEMA_TRAILER = SubObject(_TRUCK_TRAILER_SCHEMA)

_TRUCK_SCHEMA = ObjectStructuralSchema(
    TRUCK,
    sub_objects=[_TRUCK_SCHEMA_CAB, _TRUCK_SCHEMA_TRAILER],
    sub_object_relations=flatten_relations(
        [
            contacts(_TRUCK_SCHEMA_CAB, _TRUCK_SCHEMA_TRAILER),
            bigger_than(_TRUCK_SCHEMA_TRAILER, _TRUCK_SCHEMA_CAB),
        ]
    ),
)

_PUT_AGENT = SituationObject(THING, properties=[ANIMATE], debug_handle="put_agent")
_PUT_THEME = SituationObject(THING, debug_handle="put_theme")
_PUT_GOAL = SituationObject(THING, debug_handle="put_goal")
_PUT_MANIPULATOR = SituationObject(
    THING, properties=[CAN_MANIPULATE_OBJECTS], debug_handle="put_manipulator"
)

_CONTACTING_MANIPULATOR = Region(
    reference_object=_PUT_MANIPULATOR, distance=EXTERIOR_BUT_IN_CONTACT
)

_PUT_ACTION_DESCRIPTION = ActionDescription(
    frame=ActionDescriptionFrame({AGENT: _PUT_AGENT, THEME: _PUT_THEME, GOAL: _PUT_GOAL}),
    during=DuringAction(
        objects_to_paths=[
            (_PUT_THEME, SpatialPath(FROM, _CONTACTING_MANIPULATOR)),
            (_PUT_THEME, SpatialPath(TO, _PUT_GOAL)),
        ]
    ),
    enduring_conditions=[
        Relation(SMALLER_THAN, _PUT_THEME, _PUT_AGENT),
        Relation(PART_OF, _PUT_MANIPULATOR, _PUT_AGENT),
    ],
    preconditions=[
        Relation(IN_REGION, _PUT_THEME, _CONTACTING_MANIPULATOR),
        # THEME is not already located in GOAL
        Relation(IN_REGION, _PUT_THEME, _PUT_GOAL, negated=True),
    ],
    postconditions=[
        Relation(IN_REGION, _PUT_THEME, _CONTACTING_MANIPULATOR, negated=True),
        Relation(IN_REGION, _PUT_THEME, _PUT_GOAL),
    ],
    asserted_properties=[
        (_PUT_AGENT, VOLITIONALLY_INVOLVED),
        (_PUT_AGENT, CAUSES_CHANGE),
        (_PUT_THEME, UNDERGOES_CHANGE),
        (_PUT_GOAL, STATIONARY),
    ],
)

_PUSH_AGENT = SituationObject(THING, properties=[ANIMATE], debug_handle="push-agent")
_PUSH_THEME = SituationObject(INANIMATE_OBJECT, debug_handle="push-theme")
PUSH_GOAL = SituationObject(THING, debug_handle="push_goal")
_PUSH_MANIPULATOR = SituationObject(
    THING, properties=[CAN_MANIPULATE_OBJECTS], debug_handle="push-manipulator"
)
PUSH_SURFACE_AUX = SituationObject(
    THING, properties=[CAN_HAVE_THINGS_RESTING_ON_THEM], debug_handle="push-surface"
)


def _make_push_descriptions() -> Iterable[Tuple[OntologyNode, ActionDescription]]:
    during: DuringAction[SituationObject] = DuringAction(
        objects_to_paths=[(_PUSH_THEME, SpatialPath(TO, PUSH_GOAL))]
    )
    enduring = [
        partOf(_PUSH_MANIPULATOR, _PUSH_AGENT),
        bigger_than(_PUSH_AGENT, _PUSH_THEME),
        bigger_than(PUSH_SURFACE_AUX, _PUSH_THEME),
        contacts(_PUSH_MANIPULATOR, _PUSH_THEME),
        on(_PUSH_THEME, PUSH_SURFACE_AUX),
    ]
    preconditions = [Relation(IN_REGION, _PUSH_THEME, PUSH_GOAL, negated=True)]
    postconditions = [Relation(IN_REGION, _PUSH_THEME, PUSH_GOAL)]
    asserted_properties = [
        (_PUSH_AGENT, VOLITIONALLY_INVOLVED),
        (_PUSH_AGENT, CAUSES_CHANGE),
        (_PUSH_THEME, UNDERGOES_CHANGE),
    ]
    # explicit goal
    yield PUSH, ActionDescription(
        frame=ActionDescriptionFrame(
            {AGENT: _PUSH_AGENT, THEME: _PUSH_THEME, GOAL: PUSH_GOAL}
        ),
        during=during,
        enduring_conditions=enduring,
        preconditions=preconditions,
        postconditions=postconditions,
        asserted_properties=asserted_properties,
    )
    # implicit goal
    yield PUSH, ActionDescription(
        frame=ActionDescriptionFrame({AGENT: _PUSH_AGENT, THEME: _PUSH_THEME}),
        during=during,
        enduring_conditions=enduring,
        preconditions=preconditions,
        postconditions=postconditions,
        asserted_properties=asserted_properties,
    )


_GO_AGENT = SituationObject(THING, properties=[SELF_MOVING])
_GO_GOAL = SituationObject(THING)


def _make_go_description() -> Iterable[Tuple[OntologyNode, ActionDescription]]:
    # bare go
    postconditions = [Relation(IN_REGION, _GO_AGENT, _GO_GOAL)]
    during: DuringAction[SituationObject] = DuringAction(
        objects_to_paths=[(_GO_AGENT, SpatialPath(TO, _GO_GOAL))]
    )
    asserted_properties = [(_GO_AGENT, VOLITIONALLY_INVOLVED), (_GO_AGENT, MOVES)]
    yield GO, ActionDescription(
        frame=ActionDescriptionFrame({AGENT: _GO_AGENT}),
        during=during,
        postconditions=postconditions,
        asserted_properties=asserted_properties,
    )

    # goes to goal
    yield GO, ActionDescription(
        frame=ActionDescriptionFrame({AGENT: _GO_AGENT, GOAL: _GO_GOAL}),
        during=during,
        postconditions=postconditions,
        asserted_properties=asserted_properties,
    )


_COME_AGENT = SituationObject(THING, properties=[ANIMATE])
_COME_GOAL = SituationObject(THING)

_COME_ACTION_DESCRIPTION = ActionDescription(
    frame=ActionDescriptionFrame(
        # AGENT comes to DESTINATION
        {AGENT: _COME_AGENT, GOAL: _COME_GOAL}
    ),
    preconditions=[
        Relation(
            IN_REGION, _COME_AGENT, Region(_COME_GOAL, distance=DISTAL)
        )
    ],
    during=DuringAction(objects_to_paths=[(_COME_AGENT, SpatialPath(TO, _COME_GOAL))]),
    postconditions=[
        Relation(IN_REGION, _COME_AGENT, Region(_COME_GOAL, distance=PROXIMAL))
    ],
    asserted_properties=[(_COME_AGENT, VOLITIONALLY_INVOLVED), (_COME_AGENT, MOVES)],
)

_TAKE_AGENT = SituationObject(THING, properties=[ANIMATE])
_TAKE_THEME = SituationObject(THING)
_TAKE_GOAL = SituationObject(THING)
_TAKE_MANIPULATOR = SituationObject(THING, properties=[CAN_MANIPULATE_OBJECTS])

_TAKE_ACTION_DESCRIPTION = ActionDescription(
    frame=ActionDescriptionFrame({AGENT: _TAKE_AGENT, THEME: _TAKE_THEME}),
    enduring_conditions=[
        bigger_than(_TAKE_AGENT, _TAKE_THEME),
        partOf(_TAKE_MANIPULATOR, _TAKE_AGENT),
    ],
    preconditions=[negate(has(_TAKE_AGENT, _TAKE_THEME))],
    postconditions=[
        has(_TAKE_AGENT, _TAKE_THEME),
        Relation(
            IN_REGION,
            _TAKE_THEME,
            Region(_TAKE_MANIPULATOR, distance=EXTERIOR_BUT_IN_CONTACT),
        ),
    ],
    asserted_properties=[
        (_TAKE_AGENT, VOLITIONALLY_INVOLVED),
        (_TAKE_AGENT, CAUSES_CHANGE),
        (_TAKE_THEME, UNDERGOES_CHANGE),
    ],
)

_EAT_AGENT = SituationObject(THING, properties=[ANIMATE])
_EAT_PATIENT = SituationObject(INANIMATE_OBJECT, properties=[EDIBLE])

_EAT_ACTION_DESCRIPTION = ActionDescription(
    frame=ActionDescriptionFrame({AGENT: _EAT_AGENT, PATIENT: _EAT_PATIENT}),
    enduring_conditions=[bigger_than(_EAT_AGENT, _EAT_PATIENT)],
    postconditions=[inside(_EAT_PATIENT, _EAT_AGENT)],
    # TODO: express role of mouth
    asserted_properties=[
        (_EAT_AGENT, VOLITIONALLY_INVOLVED),
        (_EAT_AGENT, CAUSES_CHANGE),
        (_EAT_PATIENT, UNDERGOES_CHANGE),
    ],
)

_GIVE_AGENT = SituationObject(THING, properties=[ANIMATE])
_GIVE_THEME = SituationObject(INANIMATE_OBJECT)
_GIVE_GOAL = SituationObject(THING, properties=[ANIMATE])
_GIVE_AGENT_MANIPULATOR = SituationObject(THING, properties=[CAN_MANIPULATE_OBJECTS])
_GIVE_GOAL_MANIPULATOR = SituationObject(THING, properties=[CAN_MANIPULATE_OBJECTS])

_GIVE_ACTION_DESCRIPTION = ActionDescription(
    frame=ActionDescriptionFrame(
        {AGENT: _GIVE_AGENT, THEME: _GIVE_THEME, GOAL: _GIVE_GOAL}
    ),
    enduring_conditions=[
        bigger_than(_GIVE_AGENT, _GIVE_THEME),
        bigger_than(_GIVE_GOAL, _GIVE_THEME),
        partOf(_GIVE_AGENT_MANIPULATOR, _GIVE_AGENT),
        partOf(_GIVE_GOAL_MANIPULATOR, _GIVE_GOAL),
    ],
    preconditions=[
        has(_GIVE_AGENT, _GIVE_THEME),
        negate(has(_GIVE_GOAL, _GIVE_THEME)),
        contacts(_GIVE_AGENT_MANIPULATOR, _GIVE_THEME),
        negate(contacts(_GIVE_GOAL_MANIPULATOR, _GIVE_THEME)),
    ],
    postconditions=[
        negate(has(_GIVE_AGENT, _GIVE_THEME)),
        has(_GIVE_GOAL, _GIVE_THEME),
        negate(contacts(_GIVE_AGENT_MANIPULATOR, _GIVE_THEME)),
        contacts(_GIVE_GOAL_MANIPULATOR, _GIVE_THEME),
    ],
    asserted_properties=[
        (_GIVE_AGENT, VOLITIONALLY_INVOLVED),
        (_GIVE_AGENT, CAUSES_CHANGE),
        (_GIVE_THEME, UNDERGOES_CHANGE),
    ],
)

_SPIN_AGENT = SituationObject(THING, properties=[ANIMATE])
_SPIN_MANIPULATOR = SituationObject(THING, properties=[CAN_MANIPULATE_OBJECTS])


def _make_spin_descriptions() -> Iterable[Tuple[OntologyNode, ActionDescription]]:
    spin_theme = SituationObject(THING)

    # intransitive
    yield SPIN, ActionDescription(
        frame=ActionDescriptionFrame({AGENT: _SPIN_AGENT}),
        during=DuringAction(
            objects_to_paths=[(_SPIN_AGENT, spin_around_primary_axis(_SPIN_AGENT))]
        ),
        asserted_properties=[
            (_SPIN_AGENT, VOLITIONALLY_INVOLVED),
            (_SPIN_AGENT, CAUSES_CHANGE),
            (_SPIN_AGENT, UNDERGOES_CHANGE),
        ],
    )

    # transitive
    yield SPIN, ActionDescription(
        frame=ActionDescriptionFrame({AGENT: _SPIN_AGENT, THEME: spin_theme}),
        during=DuringAction(
            objects_to_paths=[(spin_theme, spin_around_primary_axis(spin_theme))]
        ),
        asserted_properties=[
            (_SPIN_AGENT, VOLITIONALLY_INVOLVED),
            (_SPIN_AGENT, CAUSES_CHANGE),
            (spin_theme, UNDERGOES_CHANGE),
        ],
    )


def spin_around_primary_axis(object_):
    return SpatialPath(
        operator=None,
        reference_object=object_,
        reference_axis=Axis.primary_of(object_),
        orientation_changed=True,
    )


SIT_THING_SAT_ON = SituationObject(THING, debug_handle="thing-sat-on")
SIT_GOAL = SituationObject(THING, debug_handle="sit-goal")  # really a region


def _make_sit_action_descriptions() -> Iterable[Tuple[OntologyNode, ActionDescription]]:
    sit_agent = SituationObject(THING, properties=[ANIMATE], debug_handle="sit-agent")

    post_conditions = [Relation(IN_REGION, sit_agent, SIT_GOAL)]

    yield SIT, ActionDescription(
        frame=ActionDescriptionFrame({AGENT: sit_agent, GOAL: SIT_GOAL}),
        preconditions=[negate(contacts(sit_agent, SIT_THING_SAT_ON))],
        postconditions=post_conditions,
        asserted_properties=[(sit_agent, VOLITIONALLY_INVOLVED), (sit_agent, MOVES)],
    )

    yield SIT, ActionDescription(
        frame=ActionDescriptionFrame({AGENT: sit_agent}),
        preconditions=[negate(contacts(sit_agent, SIT_THING_SAT_ON))],
        postconditions=post_conditions,
        asserted_properties=[(sit_agent, VOLITIONALLY_INVOLVED), (sit_agent, MOVES)],
    )


DRINK_CONTAINER_AUX = SituationObject(THING, properties=[HOLLOW])


def _make_drink_description() -> Iterable[Tuple[OntologyNode, ActionDescription]]:
    drink_agent = SituationObject(THING, properties=[ANIMATE])
    drink_theme = SituationObject(THING, properties=[LIQUID])

    yield (
        DRINK,
        ActionDescription(
            frame=ActionDescriptionFrame({AGENT: drink_agent, THEME: drink_theme}),
            preconditions=[
                inside(drink_theme, DRINK_CONTAINER_AUX),
                bigger_than(drink_agent, DRINK_CONTAINER_AUX),
            ],
            postconditions=[inside(drink_theme, drink_agent)],
            asserted_properties=[
                (drink_agent, VOLITIONALLY_INVOLVED),
                (drink_agent, CAUSES_CHANGE),
                (drink_theme, UNDERGOES_CHANGE),
            ],
        ),
    )


_FALL_THEME = SituationObject(THING)
_FALL_GROUND = SituationObject(GROUND)

_FALL_ACTION_DESCRIPTION = ActionDescription(
    frame=ActionDescriptionFrame({THEME: _FALL_THEME}),
    during=DuringAction(
        objects_to_paths=[
            (_FALL_THEME, SpatialPath(operator=TOWARD, reference_object=_FALL_GROUND))
        ]
    ),
    asserted_properties=[(_FALL_THEME, MOVES)],
)

_THROW_AGENT = SituationObject(THING, properties=[ANIMATE])
_THROW_THEME = SituationObject(INANIMATE_OBJECT)
THROW_GOAL = SituationObject(THING)
_THROW_MANIPULATOR = SituationObject(THING, properties=[CAN_MANIPULATE_OBJECTS])
_THROW_GROUND = SituationObject(GROUND)


def _make_throw_descriptions() -> Iterable[Tuple[OntologyNode, ActionDescription]]:
    during: DuringAction[SituationObject] = DuringAction(
        objects_to_paths=[(_THROW_THEME, SpatialPath(TO, THROW_GOAL))],
        # must be above the ground at some point during the action
        at_some_point=[
            Relation(
                IN_REGION,
                _THROW_THEME,
                Region(
                    reference_object=_THROW_GROUND,
                    distance=DISTAL,
                    direction=Direction(
                        positive=True, relative_to_axis=GRAVITATIONAL_AXIS
                    ),
                ),
            )
        ],
    )
    enduring = [
        partOf(_THROW_MANIPULATOR, _THROW_AGENT),
        bigger_than(_THROW_AGENT, _THROW_THEME),
    ]
    preconditions = [
        has(_THROW_AGENT, _THROW_THEME),
        contacts(_THROW_MANIPULATOR, _THROW_THEME),
    ]
    postconditions = [
        inside(_THROW_THEME, THROW_GOAL),
        negate(contacts(_THROW_MANIPULATOR, _THROW_THEME)),
    ]
    asserted_properties = [
        (_THROW_AGENT, VOLITIONALLY_INVOLVED),
        (_THROW_AGENT, CAUSES_CHANGE),
        (_THROW_THEME, UNDERGOES_CHANGE),
    ]
    # explicit goal
    yield THROW, ActionDescription(
        frame=ActionDescriptionFrame(
            {AGENT: _THROW_AGENT, THEME: _THROW_THEME, GOAL: THROW_GOAL}
        ),
        during=during,
        enduring_conditions=enduring,
        preconditions=preconditions,
        postconditions=postconditions,
        asserted_properties=asserted_properties,
    )
    # implicit goal
    yield THROW, ActionDescription(
        frame=ActionDescriptionFrame({AGENT: _THROW_AGENT, THEME: _THROW_THEME}),
        during=during,
        enduring_conditions=enduring,
        preconditions=preconditions,
        postconditions=postconditions,
        asserted_properties=asserted_properties,
    )


_MOVE_AGENT = SituationObject(THING, properties=[ANIMATE])
_MOVE_THEME = SituationObject(THING)
MOVE_GOAL = SituationObject(THING)
_MOVE_MANIPULATOR = SituationObject(THING, properties=[CAN_MANIPULATE_OBJECTS])


def _make_move_descriptions() -> Iterable[Tuple[OntologyNode, ActionDescription]]:
    # bare move - "X moves (of its own accord)"
    yield MOVE, ActionDescription(
        frame=ActionDescriptionFrame({AGENT: _MOVE_AGENT}),
        postconditions=[Relation(IN_REGION, _MOVE_AGENT, MOVE_GOAL)],
        asserted_properties=[
            (_MOVE_AGENT, VOLITIONALLY_INVOLVED),
            (_MOVE_AGENT, CAUSES_CHANGE),
            (_MOVE_AGENT, UNDERGOES_CHANGE),
        ],
    )

    # X moves Y
    yield MOVE, ActionDescription(
        frame=ActionDescriptionFrame({AGENT: _MOVE_AGENT, THEME: _MOVE_THEME}),
        postconditions=[Relation(IN_REGION, _MOVE_THEME, MOVE_GOAL)],
        asserted_properties=[
            (_MOVE_AGENT, VOLITIONALLY_INVOLVED),
            (_MOVE_AGENT, CAUSES_CHANGE),
            (_MOVE_THEME, UNDERGOES_CHANGE),
        ],
    )

    # X moves Y to Z
    # TODO: manipulator
    yield MOVE, ActionDescription(
        frame=ActionDescriptionFrame(
            {AGENT: _MOVE_AGENT, THEME: _MOVE_THEME, GOAL: MOVE_GOAL}
        ),
        postconditions=[Relation(IN_REGION, _MOVE_THEME, MOVE_GOAL)],
        asserted_properties=[
            (_MOVE_AGENT, VOLITIONALLY_INVOLVED),
            (_MOVE_AGENT, CAUSES_CHANGE),
            (_MOVE_THEME, UNDERGOES_CHANGE),
        ],
    )


JUMP_INITIAL_SUPPORTER_AUX = SituationObject(THING)


def _make_jump_description() -> Iterable[Tuple[OntologyNode, ActionDescription]]:
    jump_agent = SituationObject(THING, properties=[ANIMATE])
    jump_ground = SituationObject(GROUND)

    yield (
        JUMP,
        ActionDescription(
            frame=ActionDescriptionFrame({AGENT: jump_agent}),
            preconditions=[
                Relation(
                    IN_REGION,
                    jump_agent,
                    Region(JUMP_INITIAL_SUPPORTER_AUX, distance=EXTERIOR_BUT_IN_CONTACT),
                )
            ],
            during=DuringAction(
                objects_to_paths=[
                    (jump_agent, SpatialPath(AWAY_FROM, JUMP_INITIAL_SUPPORTER_AUX)),
                    (jump_agent, SpatialPath(AWAY_FROM, jump_ground)),
                ]
            ),
            asserted_properties=[
                (jump_agent, VOLITIONALLY_INVOLVED),
                (jump_agent, MOVES),
            ],
        ),
    )


ROLL_SURFACE_AUXILIARY = SituationObject(
    INANIMATE_OBJECT, [CAN_HAVE_THINGS_RESTING_ON_THEM], debug_handle="roll-surface-aux"
)


def _make_roll_description() -> Iterable[Tuple[OntologyNode, ActionDescription]]:
    roll_agent = SituationObject(THING, properties=[ANIMATE])
    roll_theme = SituationObject(INANIMATE_OBJECT, properties=[ROLLABLE])

    def make_during(rollee: SituationObject) -> DuringAction[SituationObject]:
        return DuringAction(
            continuously=[contacts(rollee, ROLL_SURFACE_AUXILIARY)],
            objects_to_paths=[
                (
                    rollee,
                    SpatialPath(
                        operator=None,
                        reference_object=rollee,
                        reference_axis=Axis(
                            reference_object=None, name="direction of motion"
                        ),
                        orientation_changed=True,
                    ),
                )
            ],
        )

    # transitive roll
    yield (
        ROLL,
        ActionDescription(
            frame=ActionDescriptionFrame({AGENT: roll_agent, THEME: roll_theme}),
            during=make_during(roll_theme),
            asserted_properties=[
                (roll_agent, VOLITIONALLY_INVOLVED),
                (roll_agent, CAUSES_CHANGE),
                (roll_theme, UNDERGOES_CHANGE),
            ],
        ),
    )

    # intransitive roll
    yield (
        ROLL,
        ActionDescription(
            frame=ActionDescriptionFrame({AGENT: roll_agent}),
            during=make_during(roll_agent),
            asserted_properties=[(roll_agent, MOVES)],
        ),
    )


_FLY_AGENT = SituationObject(THING, properties=[ANIMATE])
_FLY_GROUND = SituationObject(GROUND)

_FLY_ACTION_DESCRIPTION = ActionDescription(
    frame=ActionDescriptionFrame({AGENT: _FLY_AGENT}),
    during=DuringAction(
        continuously=[
            Relation(
                IN_REGION,
                _FLY_AGENT,
                Region(
                    reference_object=_FLY_GROUND,
                    distance=DISTAL,
                    direction=Direction(
                        positive=True, relative_to_axis=GRAVITATIONAL_AXIS
                    ),
                ),
            )
        ]
    ),
    asserted_properties=[(_FLY_AGENT, VOLITIONALLY_INVOLVED), (_FLY_AGENT, MOVES)],
)

_ACTIONS_TO_DESCRIPTIONS = [
    (PUT, _PUT_ACTION_DESCRIPTION),
    (COME, _COME_ACTION_DESCRIPTION),
    (GIVE, _GIVE_ACTION_DESCRIPTION),
    (TAKE, _TAKE_ACTION_DESCRIPTION),
    (EAT, _EAT_ACTION_DESCRIPTION),
    (FALL, _FALL_ACTION_DESCRIPTION),
    (FLY, _FLY_ACTION_DESCRIPTION),
]

_ACTIONS_TO_DESCRIPTIONS.extend(_make_roll_description())
_ACTIONS_TO_DESCRIPTIONS.extend(_make_jump_description())
_ACTIONS_TO_DESCRIPTIONS.extend(_make_drink_description())
_ACTIONS_TO_DESCRIPTIONS.extend(_make_sit_action_descriptions())
_ACTIONS_TO_DESCRIPTIONS.extend(_make_move_descriptions())
_ACTIONS_TO_DESCRIPTIONS.extend(_make_spin_descriptions())
_ACTIONS_TO_DESCRIPTIONS.extend(_make_go_description())
_ACTIONS_TO_DESCRIPTIONS.extend(_make_push_descriptions())
_ACTIONS_TO_DESCRIPTIONS.extend(_make_throw_descriptions())

GAILA_PHASE_1_ONTOLOGY = Ontology(
    "gaila-phase-1",
    _ontology_graph,
    structural_schemata=[
        (BALL, _BALL_SCHEMA),
        (CHAIR, _CHAIR_SCHEMA),
        (PERSON, _PERSON_SCHEMA),
        (TABLE, _TABLE_SCHEMA),
        (DOG, _DOG_SCHEMA),
        (BIRD, _BIRD_SCHEMA),
        (BOX, _BOX_SCHEMA),
        (WATER, _WATER_SCHEMA),
        (JUICE, _JUICE_SCHEMA),
        (MILK, _MILK_SCHEMA),
        (DOOR, _DOOR_SCHEMA),
        (HAT, _HAT_SCHEMA),
        (COOKIE, _COOKIE_SCHEMA),
        (HEAD, _HEAD_SCHEMA),
        (CUP, _CUP_SCHEMA),
        (BOX, _BOX_SCHEMA),
        (BOOK, _BOOK_SCHEMA),
        (HOUSE, _HOUSE_SCHEMA),
        (HAND, _HAND_SCHEMA),
        (CAR, _CAR_SCHEMA),
        (TRUCK, _TRUCK_SCHEMA),
        (GROUND, _GROUND_SCHEMA),
        (LEARNER, _LEARNER_SCHEMA),
    ],
    action_to_description=_ACTIONS_TO_DESCRIPTIONS,
    relations=build_size_relationships(
        (
            (HOUSE,),
            (_ROOF, _WALL),
            (CAR, TRUCK),
            (_TRAILER, _FLATBED),
            (_TRUCK_CAB,),
            (TABLE, DOOR),
            (_TABLETOP,),
            (MOM, DAD),
            (DOG, BOX, CHAIR, _TIRE),
            (BABY,),
            (_BODY,),
            (_TORSO, _CHAIR_BACK, _CHAIR_SEAT),
            (_ARM, _LEG),
            (HAND, HEAD, _ARM_SEGMENT),
            (BALL, BIRD, BOOK, COOKIE, CUP, HAT),
            (_TAIL, _WING),
        ),
        relation_type=BIGGER_THAN,
        opposite_type=SMALLER_THAN,
    ),
)


def is_recognized_particular(ontology: Ontology, node: OntologyNode) -> bool:
    return any(
        ontology.is_subtype_of(property_, RECOGNIZED_PARTICULAR_PROPERTY)
        for property_ in ontology.properties_for_node(node)
    )
