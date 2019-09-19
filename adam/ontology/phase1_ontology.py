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
from typing import Optional, Sequence, Tuple

from immutablecollections import (
    ImmutableDict,
    immutabledict,
    immutableset,
    immutablesetmultidict,
)
from more_itertools import flatten

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
from adam.ontology.phase1_size_relationships import build_size_relationships
from adam.ontology.action_description import ActionDescription, ActionDescriptionFrame
from adam.ontology.during import DuringAction
from adam.ontology.ontology import Ontology
from adam.ontology.phase1_spatial_relations import (
    AWAY_FROM,
    Direction,
    EXTERIOR_BUT_IN_CONTACT,
    FROM,
    INTERIOR,
    SpatialPath,
    TO,
    GRAVITATIONAL_AXIS,
    Axis,
    Region,
    TOWARD,
    DISTAL,
    PROXIMAL,
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

COLOR = OntologyNode("color")
subtype(COLOR, PERCEIVABLE_PROPERTY)
RED = OntologyNode("red", [CAN_FILL_TEMPLATE_SLOT])
BLUE = OntologyNode("blue", [CAN_FILL_TEMPLATE_SLOT])
GREEN = OntologyNode("green", [CAN_FILL_TEMPLATE_SLOT])
BLACK = OntologyNode("black", [CAN_FILL_TEMPLATE_SLOT])
WHITE = OntologyNode("white", [CAN_FILL_TEMPLATE_SLOT])
TRANSPARENT = OntologyNode("transparent", [CAN_FILL_TEMPLATE_SLOT])
subtype(RED, COLOR)
subtype(BLUE, COLOR)
subtype(GREEN, COLOR)
subtype(BLACK, COLOR)
subtype(WHITE, COLOR)
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
    "ground", non_inheritable_properties=[IS_GROUND, CAN_HAVE_THINGS_RESTING_ON_THEM]
)
subtype(GROUND, INANIMATE_OBJECT)

TABLE = OntologyNode("table", [CAN_FILL_TEMPLATE_SLOT, CAN_HAVE_THINGS_RESTING_ON_THEM])
subtype(TABLE, INANIMATE_OBJECT)
BALL = OntologyNode("ball", [CAN_FILL_TEMPLATE_SLOT])
subtype(BALL, INANIMATE_OBJECT)
BOOK = OntologyNode("book", [CAN_FILL_TEMPLATE_SLOT, CAN_HAVE_THINGS_RESTING_ON_THEM])
subtype(BOOK, INANIMATE_OBJECT)
HOUSE = OntologyNode("house", [HOLLOW, CAN_FILL_TEMPLATE_SLOT])
subtype(HOUSE, INANIMATE_OBJECT)
CAR = OntologyNode(
    "car", [HOLLOW, CAN_FILL_TEMPLATE_SLOT, SELF_MOVING, CAN_HAVE_THINGS_RESTING_ON_THEM]
)
subtype(CAR, INANIMATE_OBJECT)
WATER = OntologyNode(
    "water", [LIQUID], non_inheritable_properties=[TRANSPARENT, CAN_FILL_TEMPLATE_SLOT]
)
subtype(WATER, INANIMATE_OBJECT)
JUICE = OntologyNode(
    "juice", [LIQUID], non_inheritable_properties=[RED, CAN_FILL_TEMPLATE_SLOT]
)
subtype(JUICE, INANIMATE_OBJECT)
CUP = OntologyNode("cup", [HOLLOW, CAN_FILL_TEMPLATE_SLOT])
subtype(CUP, INANIMATE_OBJECT)
BOX = OntologyNode(
    "box", [HOLLOW, CAN_FILL_TEMPLATE_SLOT, CAN_HAVE_THINGS_RESTING_ON_THEM]
)
subtype(BOX, INANIMATE_OBJECT)
CHAIR = OntologyNode("chair", [CAN_FILL_TEMPLATE_SLOT, CAN_HAVE_THINGS_RESTING_ON_THEM])
subtype(CHAIR, INANIMATE_OBJECT)
# should a HEAD be hollow? We are answering yes for now,
# because food and liquids can enter it,
# but we eventually want something more sophisticated.
HEAD = OntologyNode(
    "head", [HOLLOW, CAN_FILL_TEMPLATE_SLOT, CAN_HAVE_THINGS_RESTING_ON_THEM]
)
subtype(HEAD, INANIMATE_OBJECT)
MILK = OntologyNode(
    "milk", [LIQUID], non_inheritable_properties=[WHITE, CAN_FILL_TEMPLATE_SLOT]
)
subtype(MILK, INANIMATE_OBJECT)
HAND = OntologyNode("hand", [CAN_MANIPULATE_OBJECTS, CAN_FILL_TEMPLATE_SLOT])
subtype(HAND, INANIMATE_OBJECT)
TRUCK = OntologyNode(
    "truck",
    [HOLLOW, CAN_FILL_TEMPLATE_SLOT, SELF_MOVING, CAN_HAVE_THINGS_RESTING_ON_THEM],
)
subtype(TRUCK, INANIMATE_OBJECT)
DOOR = OntologyNode("door", [CAN_FILL_TEMPLATE_SLOT])
subtype(DOOR, INANIMATE_OBJECT)
HAT = OntologyNode("hat", [CAN_FILL_TEMPLATE_SLOT])
subtype(HAT, INANIMATE_OBJECT)
COOKIE = OntologyNode("cookie", [CAN_FILL_TEMPLATE_SLOT])
subtype(COOKIE, INANIMATE_OBJECT)

PERSON = OntologyNode("person", inheritable_properties=[ANIMATE, SELF_MOVING])
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
DOG = OntologyNode("dog", [CAN_FILL_TEMPLATE_SLOT])
subtype(DOG, NONHUMAN_ANIMAL)
BIRD = OntologyNode("bird", [CAN_FILL_TEMPLATE_SLOT])
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
subtype(_ARM, INANIMATE_OBJECT)
_CHAIR_SEAT = OntologyNode("chairseat")
subtype(_ARM, INANIMATE_OBJECT)
_TABLETOP = OntologyNode("tabletop")
subtype(_TABLETOP, INANIMATE_OBJECT)
_TAIL = OntologyNode("tail")
subtype(_TAIL, _BODY_PART)
_WING = OntologyNode("wing")
subtype(_WING, _BODY_PART)
_ARM_SEGMENT = OntologyNode("armsegment")
subtype(_ARM_SEGMENT, _BODY_PART)
_WALL = OntologyNode("wall")
subtype(_ARM, INANIMATE_OBJECT)
_ROOF = OntologyNode("roof")
subtype(_ARM, INANIMATE_OBJECT)
_TIRE = OntologyNode("tire")
subtype(_ARM, INANIMATE_OBJECT)
_TRUCK_CAB = OntologyNode("truckcab")
subtype(_ARM, INANIMATE_OBJECT)
_TRAILER = OntologyNode("trailer")
subtype(_ARM, INANIMATE_OBJECT)
_FLATBED = OntologyNode("flatbed")
subtype(_ARM, INANIMATE_OBJECT)
_BODY = OntologyNode("body")
subtype(_BODY, _BODY_PART)

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
GIVE = OntologyNode("give")
subtype(GIVE, ACTION)
TURN = OntologyNode("turn")
subtype(TURN, ACTION)
SIT = OntologyNode("sit")
subtype(SIT, ACTION)
DRINK = OntologyNode("drink")
subtype(DRINK, CONSUME)
FALL = OntologyNode("fall")
subtype(FALL, ACTION)  # ?
THROW = OntologyNode("throw")
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


_GROUND_SCHEMA = ObjectStructuralSchema(GROUND)

# Structural Objects without Sub-Parts which are part of our Phase 1 Vocabulary
# These may need to evolve to reflect the changes for visualization of phase 1
_DOOR_SCHEMA = ObjectStructuralSchema(DOOR)
_BALL_SCHEMA = ObjectStructuralSchema(BALL)
_BOX_SCHEMA = ObjectStructuralSchema(BOX)
_WATER_SCHEMA = ObjectStructuralSchema(WATER)
_JUICE_SCHEMA = ObjectStructuralSchema(JUICE)
_BOX_SCHEMA = ObjectStructuralSchema(BOX)
_MILK_SCHEMA = ObjectStructuralSchema(MILK)
_HAT_SCHEMA = ObjectStructuralSchema(HAT)
_COOKIE_SCHEMA = ObjectStructuralSchema(COOKIE)
_CUP_SCHEMA = ObjectStructuralSchema(CUP)
_BOOK_SCHEMA = ObjectStructuralSchema(BOOK)
_HAND_SCHEMA = ObjectStructuralSchema(HAND)
_HEAD_SCHEMA = ObjectStructuralSchema(HEAD)

# Hierarchical structure of objects
_TORSO_SCHEMA = ObjectStructuralSchema(_TORSO)
_LEG_SCHEMA = ObjectStructuralSchema(_LEG)
_CHAIRBACK_SCHEMA = ObjectStructuralSchema(_CHAIR_BACK)
_CHAIR_SEAT_SCHEMA = ObjectStructuralSchema(_CHAIR_SEAT)
_TABLETOP_SCHEMA = ObjectStructuralSchema(_TABLETOP)
_TAIL_SCHEMA = ObjectStructuralSchema(_TAIL)
_WING_SCHEMA = ObjectStructuralSchema(_WING)
_ARM_SEGMENT_SCHEMA = ObjectStructuralSchema(_ARM_SEGMENT)
_ROOF_SCHEMA = ObjectStructuralSchema(_ROOF)
_WALL_SCHEMA = ObjectStructuralSchema(_WALL)
_TIRE_SCHEMA = ObjectStructuralSchema(_TIRE)
_FLATBED_SCHEMA = ObjectStructuralSchema(_FLATBED)
_BODY_SCHEMA = ObjectStructuralSchema(_BODY)

# schemata describing the sub-object structural nature of a Human Arm
_ARM_SCHEMA_HAND = SubObject(_HAND_SCHEMA)
_ARM_SCHEMA_UPPER = SubObject(
    _ARM_SEGMENT_SCHEMA
)  # Is that the correct sub-object we want?
_ARM_SCHEMA_LOWER = SubObject(_ARM_SEGMENT_SCHEMA)

_ARM_SCHEMA = ObjectStructuralSchema(
    _ARM,
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
    PERSON,
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
_DOG_SCHEMA_HEAD = SubObject(_HEAD_SCHEMA)
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
_BIRD_SCHEMA_HEAD = SubObject(_HEAD_SCHEMA)
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
    frames=[
        ActionDescriptionFrame({AGENT: _PUT_AGENT, THEME: _PUT_THEME, GOAL: _PUT_GOAL})
    ],
    during=DuringAction(
        paths=[
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
)

_PUSH_AGENT = SituationObject(THING, properties=[ANIMATE])
_PUSH_THEME = SituationObject(INANIMATE_OBJECT)
_PUSH_GOAL = SituationObject(THING, debug_handle="push_goal")
_PUSH_MANIPULATOR = SituationObject(THING, properties=[CAN_MANIPULATE_OBJECTS])


_PUSH_ACTION_DESCRIPTION = ActionDescription(
    frames=[
        ActionDescriptionFrame({AGENT: _PUSH_AGENT, THEME: _PUSH_THEME, GOAL: _PUSH_GOAL})
    ],
    during=DuringAction(
        continuously=flatten_relations([contacts(_PUT_MANIPULATOR, _PUT_THEME)]),
        paths=[(_PUSH_THEME, SpatialPath(TO, _PUT_GOAL))],
    ),
    enduring_conditions=[
        partOf(_PUSH_MANIPULATOR, _PUSH_AGENT),
        bigger_than(_PUSH_AGENT, _PUSH_THEME),
    ],
    preconditions=[
        contacts(_PUSH_MANIPULATOR, _PUSH_THEME),
        Relation(IN_REGION, _PUSH_THEME, _PUSH_GOAL, negated=True),
    ],
    postconditions=[Relation(IN_REGION, _PUSH_THEME, _PUSH_GOAL)],
    # TODO: encode that the THEME's vertical position does not significantly change,
    # unless there is e.g. a ramp
)

_GO_AGENT = SituationObject(THING, properties=[SELF_MOVING])
_GO_GOAL = SituationObject(THING)

_GO_ACTION_DESCRIPTION = ActionDescription(
    frames=[ActionDescriptionFrame({AGENT: _GO_AGENT, GOAL: _GO_GOAL})],
    during=DuringAction(paths=[(_GO_AGENT, SpatialPath(TO, _GO_GOAL))]),
    postconditions=[Relation(IN_REGION, _GO_AGENT, _GO_GOAL)],
)

_COME_AGENT = SituationObject(THING, properties=[ANIMATE])
_COME_GOAL = SituationObject(THING)

_COME_ACTION_DESCRIPTION = ActionDescription(
    frames=[
        ActionDescriptionFrame(
            # AGENT comes to DESTINATION
            {AGENT: _COME_AGENT, GOAL: _COME_GOAL}
        )
    ],
    preconditions=[Relation(IN_REGION, _COME_AGENT, _COME_GOAL, negated=True)],
    during=DuringAction(paths=[(_COME_AGENT, SpatialPath(TO, _COME_GOAL))]),
    postconditions=[Relation(IN_REGION, _COME_AGENT, _COME_GOAL)],
    # TODO: encode that the new location is relatively closer to the
    # learner or speaker than the old location
)

_TAKE_AGENT = SituationObject(THING, properties=[ANIMATE])
_TAKE_THEME = SituationObject(THING)
_TAKE_GOAL = SituationObject(THING)
_TAKE_MANIPULATOR = SituationObject(THING, properties=[CAN_MANIPULATE_OBJECTS])

_TAKE_ACTION_DESCRIPTION = ActionDescription(
    frames=[ActionDescriptionFrame({AGENT: _TAKE_AGENT, THEME: _TAKE_THEME})],
    enduring_conditions=[
        bigger_than(_TAKE_AGENT, _TAKE_THEME),
        partOf(_TAKE_MANIPULATOR, _TAKE_AGENT),
    ],
    preconditions=[negate(has(_TAKE_AGENT, _TAKE_THEME))],
    postconditions=[has(_TAKE_AGENT, _TAKE_THEME)],
)

_EAT_AGENT = SituationObject(THING, properties=[ANIMATE])
_EAT_THEME = SituationObject(INANIMATE_OBJECT, properties=[EDIBLE])

_EAT_ACTION_DESCRIPTION = ActionDescription(
    frames=[ActionDescriptionFrame({AGENT: _EAT_AGENT, THEME: _EAT_THEME})],
    enduring_conditions=[bigger_than(_EAT_AGENT, _EAT_THEME)],
    postconditions=[inside(_EAT_THEME, _EAT_AGENT)],
    # TODO: express role of mouth
)

_GIVE_AGENT = SituationObject(THING, properties=[ANIMATE])
_GIVE_THEME = SituationObject(INANIMATE_OBJECT)
_GIVE_GOAL = SituationObject(THING, properties=[ANIMATE])
_GIVE_AGENT_MANIPULATOR = SituationObject(THING, properties=[CAN_MANIPULATE_OBJECTS])
_GIVE_GOAL_MANIPULATOR = SituationObject(THING, properties=[CAN_MANIPULATE_OBJECTS])

_GIVE_ACTION_DESCRIPTION = ActionDescription(
    frames=[
        ActionDescriptionFrame({AGENT: _GIVE_AGENT, THEME: _GIVE_THEME, GOAL: _GIVE_GOAL})
    ],
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
)

_TURN_AGENT = SituationObject(THING, properties=[ANIMATE])
_TURN_THEME = SituationObject(THING)
_TURN_MANIPULATOR = SituationObject(THING, properties=[CAN_MANIPULATE_OBJECTS])

_TURN_ACTION_DESCRIPTION = ActionDescription(
    frames=[ActionDescriptionFrame({AGENT: _TURN_AGENT, THEME: _TURN_THEME})],
    during=DuringAction(
        paths=[
            (
                _TURN_THEME,
                SpatialPath(
                    operator=None,
                    reference_object=_TURN_THEME,
                    reference_axis=Axis.primary_of(_TURN_THEME),
                    orientation_changed=True,
                ),
            )
        ]
    ),
)

_SIT_AGENT = SituationObject(THING, properties=[ANIMATE])
_SIT_GOAL = SituationObject(THING)

_SIT_ACTION_DESCRIPTION = ActionDescription(
    frames=[ActionDescriptionFrame({AGENT: _SIT_AGENT, GOAL: _SIT_GOAL})],
    preconditions=[negate(contacts(_SIT_AGENT, _SIT_GOAL))],
    postconditions=[contacts(_SIT_AGENT, _SIT_GOAL), above(_SIT_AGENT, _SIT_GOAL)],
)

_DRINK_AGENT = SituationObject(THING, properties=[ANIMATE])
_DRINK_THEME = SituationObject(THING, properties=[LIQUID])
_DRINK_CONTAINER = SituationObject(THING, properties=[HOLLOW])

_DRINK_ACTION_DESCRIPTION = ActionDescription(
    frames=[ActionDescriptionFrame({AGENT: _DRINK_AGENT, THEME: _DRINK_THEME})],
    preconditions=[
        inside(_DRINK_THEME, _DRINK_CONTAINER),
        bigger_than(_DRINK_AGENT, _DRINK_CONTAINER),
    ],
    postconditions=[inside(_DRINK_THEME, _DRINK_AGENT)],
)

_FALL_PATIENT = SituationObject(THING)

_FALL_ACTION_DESCRIPTION = ActionDescription(
    frames=[ActionDescriptionFrame({AGENT: _FALL_PATIENT})],
    during=DuringAction(
        paths=[(_FALL_PATIENT, SpatialPath(operator=TOWARD, reference_object=GROUND))]
    ),
)

_THROW_AGENT = SituationObject(THING, properties=[ANIMATE])
_THROW_THEME = SituationObject(INANIMATE_OBJECT)
_THROW_GOAL = SituationObject(THING)
_THROW_MANIPULATOR = SituationObject(THING, properties=[CAN_MANIPULATE_OBJECTS])

_THROW_ACTION_DESCRIPTION = ActionDescription(
    frames=[
        ActionDescriptionFrame(
            {AGENT: _THROW_AGENT, THEME: _THROW_THEME, GOAL: _THROW_GOAL}
        )
    ],
    enduring_conditions=[
        bigger_than(_THROW_AGENT, _THROW_THEME),
        partOf(_THROW_MANIPULATOR, _THROW_AGENT),
    ],
    preconditions=[
        has(_THROW_AGENT, _THROW_THEME),
        contacts(_THROW_MANIPULATOR, _THROW_THEME),
    ],
    postconditions=[
        Relation(IN_REGION, _THROW_THEME, _THROW_GOAL),
        negate(contacts(_THROW_MANIPULATOR, _THROW_THEME)),
    ],
    during=DuringAction(
        # must be above the ground at some point during the action
        at_some_point=[
            Relation(
                IN_REGION,
                _THROW_THEME,
                Region(
                    reference_object=GROUND,
                    distance=DISTAL,
                    direction=Direction(
                        positive=True, relative_to_axis=GRAVITATIONAL_AXIS
                    ),
                ),
            )
        ]
    ),
)

_MOVE_AGENT = SituationObject(THING, properties=[ANIMATE])
_MOVE_THEME = SituationObject(THING)
_MOVE_GOAL = SituationObject(THING)
_MOVE_MANIPULATOR = SituationObject(THING, properties=[CAN_MANIPULATE_OBJECTS])

# TODO: a proper treatment of move awaits full treatment of multiple sub-categorization frames
_MOVE_ACTION_DESCRIPTION = ActionDescription(
    frames=[
        ActionDescriptionFrame({AGENT: _MOVE_AGENT, THEME: _MOVE_THEME, GOAL: _MOVE_GOAL})
    ],
    preconditions=[],
    postconditions=[],
)

_JUMP_AGENT = SituationObject(THING, properties=[ANIMATE])
_JUMP_INITIAL_SUPPORTER = SituationObject(THING)

_JUMP_ACTION_DESCRIPTION = ActionDescription(
    frames=[ActionDescriptionFrame({AGENT: _JUMP_AGENT})],
    preconditions=[
        Relation(
            IN_REGION,
            _JUMP_AGENT,
            Region(_JUMP_INITIAL_SUPPORTER, distance=EXTERIOR_BUT_IN_CONTACT),
        )
    ],
    during=DuringAction(
        paths=[
            (_JUMP_AGENT, SpatialPath(AWAY_FROM, _JUMP_INITIAL_SUPPORTER)),
            (_JUMP_AGENT, SpatialPath(AWAY_FROM, GROUND)),
        ]
    ),
)

_ROLL_AGENT = SituationObject(THING, properties=[ANIMATE])
_ROLL_THEME = SituationObject(INANIMATE_OBJECT, properties=[ROLLABLE])
_ROLL_GOAL = SituationObject(THING)
_ROLL_SURFACE = SituationObject(INANIMATE_OBJECT)

_ROLL_ACTION_DESCRIPTION = ActionDescription(
    frames=[
        ActionDescriptionFrame({AGENT: _ROLL_AGENT, THEME: _ROLL_THEME, GOAL: _ROLL_GOAL})
    ],
    during=DuringAction(
        continuously=[contacts(_ROLL_THEME, _ROLL_SURFACE)],
        paths=[
            (
                _ROLL_THEME,
                SpatialPath(
                    operator=None,
                    reference_object=_ROLL_THEME,
                    reference_axis=Axis(
                        reference_object=None, name="direction of motion"
                    ),
                    orientation_changed=True,
                ),
            )
        ],
    ),
    postconditions=[Relation(IN_REGION, _ROLL_THEME, _ROLL_GOAL)],
)

_FLY_AGENT = SituationObject(THING, properties=[ANIMATE])

_FLY_ACTION_DESCRIPTION = ActionDescription(
    frames=[ActionDescriptionFrame({AGENT: _FLY_AGENT})],
    during=DuringAction(
        continuously=[
            Relation(
                IN_REGION,
                _FLY_AGENT,
                Region(
                    reference_object=GROUND,
                    distance=DISTAL,
                    direction=Direction(
                        positive=True, relative_to_axis=GRAVITATIONAL_AXIS
                    ),
                ),
            )
        ]
    ),
)

GAILA_PHASE_1_ONTOLOGY = Ontology(
    _ontology_graph,
    structural_schemata=immutablesetmultidict(
        [
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
        ]
    ),
    action_to_description=immutabledict(
        [
            (PUT, _PUT_ACTION_DESCRIPTION),
            (PUSH, _PUSH_ACTION_DESCRIPTION),
            (GO, _GO_ACTION_DESCRIPTION),
            (COME, _COME_ACTION_DESCRIPTION),
            (GIVE, _GIVE_ACTION_DESCRIPTION),
            (TAKE, _TAKE_ACTION_DESCRIPTION),
            (EAT, _EAT_ACTION_DESCRIPTION),
            (TURN, _TURN_ACTION_DESCRIPTION),
            (SIT, _SIT_ACTION_DESCRIPTION),
            (DRINK, _DRINK_ACTION_DESCRIPTION),
            (FALL, _FALL_ACTION_DESCRIPTION),
            (THROW, _THROW_ACTION_DESCRIPTION),
            (MOVE, _MOVE_ACTION_DESCRIPTION),
            (JUMP, _JUMP_ACTION_DESCRIPTION),
            (ROLL, _ROLL_ACTION_DESCRIPTION),
            (FLY, _FLY_ACTION_DESCRIPTION),
        ]
    ),
    node_to_relations=build_size_relationships(
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
