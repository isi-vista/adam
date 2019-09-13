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
from immutablecollections import immutabledict, immutableset, immutablesetmultidict
from more_itertools import flatten

from adam.ontology import (ABSTRACT, ACTION, ObjectStructuralSchema, OntologyNode, PROPERTY,
                           RELATION, SubObject, THING, make_dsl_relation,
                           make_opposite_dsl_relation, make_symetric_dsl_relation,
                           minimal_ontology_graph, sub_object_relations)
from adam.ontology.action_description import ActionDescription, ActionDescriptionFrame
from adam.ontology.ontology import Ontology
from adam.situation import SituationObject, SituationRelation

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
DESTINATION = OntologyNode("destination")
subtype(DESTINATION, SEMANTIC_ROLE)


# these are "properties of properties" (e.g. whether a property is perceivable by the learner)

META_PROPERTY = OntologyNode("meta-property")
PERCEIVABLE = OntologyNode("perceivable")
subtype(PERCEIVABLE, META_PROPERTY)
BINARY = OntologyNode("binary")
subtype(BINARY, META_PROPERTY)

# properties of objects which can be perceived by the learner
PERCEIVABLE_PROPERTY = OntologyNode("perceivable-property", [PERCEIVABLE])
subtype(PERCEIVABLE_PROPERTY, PROPERTY)
ANIMATE = OntologyNode("animate", [BINARY])
subtype(ANIMATE, PERCEIVABLE_PROPERTY)
INANIMATE = OntologyNode("inanimate", [BINARY])
subtype(INANIMATE, PERCEIVABLE_PROPERTY)
SENTIENT = OntologyNode("sentient", [BINARY])
subtype(SENTIENT, PERCEIVABLE_PROPERTY)
LIQUID = OntologyNode("liquid", [BINARY])
subtype(LIQUID, PERCEIVABLE_PROPERTY)

RECOGNIZED_PARTICULAR_PROPERTY = OntologyNode("recognized-particular", [BINARY, ABSTRACT])
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

# Properties not perceived by the learner, but useful for situation generation

CAN_MANIPULATE_OBJECTS = OntologyNode("can-manipulate-objects")
subtype(CAN_MANIPULATE_OBJECTS, PROPERTY)


COLOR = OntologyNode("color", non_inheritable_properties=[ABSTRACT])
subtype(COLOR, PERCEIVABLE_PROPERTY)
RED = OntologyNode("red")
BLUE = OntologyNode("blue")
GREEN = OntologyNode("green")
BLACK = OntologyNode("black")
WHITE = OntologyNode("white")
subtype(RED, COLOR)
subtype(BLUE, COLOR)
subtype(GREEN, COLOR)
subtype(BLACK, COLOR)
subtype(WHITE, COLOR)
COLORS_TO_RGBS = {
    RED: [
        (255, 0, 0),
        (237, 28, 36),
        (196, 2, 51),
        (242, 0, 60),
        (237, 41, 57),
        (238, 32, 77),
    ],
    BLUE: [
        (0, 0, 255),
        (51, 51, 153),
        (0, 135, 189),
        (0, 147, 175),
        (0, 24, 168),
        (31, 117, 254),
    ],
    GREEN: [(0, 255, 0), (75, 111, 68), (86, 130, 3), (34, 139, 34)],
    BLACK: [(0, 0, 0), (12, 2, 15), (53, 56, 57), (52, 52, 52)],
    WHITE: [(255, 255, 255), (248, 248, 255), (245, 245, 245), (254, 254, 250)],
}

# Objects
# Information about the hierarchical structure of objects
# is given at the end of this module because it is so bulky.

INANIMATE_OBJECT = OntologyNode(
    "inanimate-object",
    inheritable_properties=[INANIMATE],
    non_inheritable_properties=[ABSTRACT],
)
subtype(INANIMATE_OBJECT, THING)
TABLE = OntologyNode("table")
subtype(TABLE, INANIMATE_OBJECT)
BALL = OntologyNode("ball")
subtype(BALL, INANIMATE_OBJECT)
BOOK = OntologyNode("book")
subtype(BOOK, INANIMATE_OBJECT)
HOUSE = OntologyNode("house")
subtype(HOUSE, INANIMATE_OBJECT)
CAR = OntologyNode("car")
subtype(CAR, INANIMATE_OBJECT)
WATER = OntologyNode("water", [LIQUID])
subtype(WATER, INANIMATE_OBJECT)
JUICE = OntologyNode("juice", [LIQUID])
subtype(JUICE, INANIMATE_OBJECT)
CUP = OntologyNode("cup")
subtype(CUP, INANIMATE_OBJECT)
BOX = OntologyNode("box")
subtype(BOX, INANIMATE_OBJECT)
CHAIR = OntologyNode("chair")
subtype(CHAIR, INANIMATE_OBJECT)
HEAD = OntologyNode("head")
subtype(HEAD, INANIMATE_OBJECT)
MILK = OntologyNode("milk", [LIQUID])
subtype(MILK, INANIMATE_OBJECT)
HAND = OntologyNode("hand", [CAN_MANIPULATE_OBJECTS])
subtype(HAND, INANIMATE_OBJECT)
TRUCK = OntologyNode("truck")
subtype(TRUCK, INANIMATE_OBJECT)
DOOR = OntologyNode("door")
subtype(DOOR, INANIMATE_OBJECT)
HAT = OntologyNode("hat")
subtype(HAT, INANIMATE_OBJECT)
COOKIE = OntologyNode("cookie")
subtype(COOKIE, INANIMATE_OBJECT)

PERSON = OntologyNode(
    "person", inheritable_properties=[ANIMATE], non_inheritable_properties=[ABSTRACT]
)
subtype(PERSON, THING)
IS_MOM = OntologyNode("is-mom")
subtype(IS_MOM, RECOGNIZED_PARTICULAR_PROPERTY)
MOM = OntologyNode("mom", [IS_MOM])
subtype(MOM, PERSON)

IS_DAD = OntologyNode("is-dad")
subtype(IS_DAD, RECOGNIZED_PARTICULAR_PROPERTY)
DAD = OntologyNode("dad", [IS_DAD])
subtype(DAD, PERSON)

BABY = OntologyNode("baby")
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

NONHUMAN_ANIMAL = OntologyNode(
    "animal", inheritable_properties=[ANIMATE], non_inheritable_properties=[ABSTRACT]
)
subtype(NONHUMAN_ANIMAL, THING)
DOG = OntologyNode("dog")
subtype(DOG, NONHUMAN_ANIMAL)
BIRD = OntologyNode("bird")
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
_ARM = OntologyNode("arm")
_TORSO = OntologyNode("torso")
_LEG = OntologyNode("leg")
_CHAIR_BACK = OntologyNode("chairback")
_CHAIR_SEAT = OntologyNode("chairseat")
_TABLETOP = OntologyNode("tabletop")
_TAIL = OntologyNode("tail")
_WING = OntologyNode("wing")
_ARM_SEGMENT = OntologyNode("armsegment")
_WALL = OntologyNode("wall")
_ROOF = OntologyNode("roof")
_TIRE = OntologyNode("tire")
_TRUCK_CAB = OntologyNode("truckcab")
_TRAILER = OntologyNode("trailer")
_FLATBED = OntologyNode("flatbed")
_BODY = OntologyNode("body")

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
HAVE = OntologyNode("have")
subtype(HAVE, STATE)
ROLL = OntologyNode("roll")
subtype(ROLL, ACTION)
FLY = OntologyNode("fly")
subtype(FLY, ACTION)


# Relations
# These are used both for situations and in the perceptual representation

SPATIAL_RELATION = OntologyNode("spatial-relation")
subtype(RELATION, SPATIAL_RELATION)
# On is an English-specific bundle of semantics, but that's okay, because this is just for
# data generation, and it will get decomposed before being presented as perceptions to the
# learner.
ON = OntologyNode("on")
subtype(ON, SPATIAL_RELATION)
PART_OF = OntologyNode("partOf")
"""
A relation indicating that one object is part of another object.
"""
subtype(PART_OF, RELATION)

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


SUPPORTS = OntologyNode("supports")
"""
A relation indicating that  one object provides the force to counteract gravity and prevent another 
object from falling.

Needs refinement to solve ambiguity: https://github.com/isi-vista/adam/issues/88
"""
subtype(SUPPORTS, SPATIAL_RELATION)

supports = make_dsl_relation(SUPPORTS)  # pylint:disable=invalid-name


CONTACTS = OntologyNode("contacts")
"""
A symmetric relation indicating that one object touches another.
"""
subtype(CONTACTS, SPATIAL_RELATION)


contacts = make_symetric_dsl_relation(CONTACTS)  # pylint:disable=invalid-name


ABOVE = OntologyNode("above")
"""
A relation indicating that (at least part of) one object occupies part of the region above another 
object.
"""
subtype(ABOVE, SPATIAL_RELATION)

BELOW = OntologyNode("below")
"""
A relation indicating that (at least part of) one object occupies part of the region below another 
object.
"""
subtype(BELOW, SPATIAL_RELATION)

above = make_opposite_dsl_relation(  # pylint:disable=invalid-name
    ABOVE, opposite_type=BELOW
)

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
    sub_object_relations=sub_object_relations(
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
    sub_object_relations=sub_object_relations(
        [
            supports(_PERSON_SCHEMA_TORSO, _PERSON_SCHEMA_HEAD),
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
    sub_object_relations=sub_object_relations(
        [
            contacts(_CHAIR_LEGS, _CHAIR_SCHEMA_SEAT),
            supports(_CHAIR_LEGS, _CHAIR_SCHEMA_SEAT),
            above(_CHAIR_SCHEMA_SEAT, _CHAIR_LEGS),
            contacts(_CHAIR_SCHEMA_BACK, _CHAIR_SCHEMA_SEAT),
            supports(_CHAIR_SCHEMA_SEAT, _CHAIR_SCHEMA_BACK),
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
    sub_object_relations=sub_object_relations(
        [
            # Relationship of tabletop to the legs
            contacts(_TABLE_SCHEMA_TABLETOP, _TABLE_LEGS),
            above(_TABLE_SCHEMA_TABLETOP, _TABLE_LEGS),
            supports(_TABLE_LEGS, _TABLE_SCHEMA_TABLETOP),
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
    sub_object_relations=sub_object_relations(
        [
            contacts(_DOG_SCHEMA_TORSO, _DOG_APPENDAGES),
            supports(_DOG_SCHEMA_TORSO, [_DOG_SCHEMA_HEAD, _DOG_SCHEMA_TAIL]),
            supports(_DOG_LEGS, _DOG_SCHEMA_TORSO),
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
    sub_object_relations=sub_object_relations(
        [
            contacts(_BIRD_SCHEMA_TORSO, _BIRD_APPENDAGES),
            above(_BIRD_SCHEMA_HEAD, _BIRD_SCHEMA_TORSO),
            above(_BIRD_SCHEMA_TORSO, _BIRD_LEGS),
            bigger_than(_BIRD_SCHEMA_TORSO, _BIRD_SCHEMA_HEAD),
            bigger_than(_BIRD_SCHEMA_TORSO, _BIRD_LEGS),
            supports(_BIRD_LEGS, _BIRD_SCHEMA_TORSO),
            supports(
                _BIRD_SCHEMA_TORSO,
                [
                    _BIRD_SCHEMA_HEAD,
                    _BIRD_SCHEMA_TAIL,
                    _BIRD_SCHEMA_LEFT_WING,
                    _BIRD_SCHEMA_RIGHT_WING,
                ],
            ),
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
    sub_object_relations=sub_object_relations(
        [
            contacts(_HOUSE_SCHEMA_ROOF, _HOUSE_SCHEMA_GROUND_FLOOR),
            supports(_HOUSE_SCHEMA_GROUND_FLOOR, _HOUSE_SCHEMA_ROOF),
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
    sub_object_relations=sub_object_relations(
        [
            contacts(_CAR_SCHEMA_TIRES, _CAR_SCHEMA_BODY),
            supports(_CAR_SCHEMA_TIRES, _CAR_SCHEMA_BODY),
        ]
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
    sub_object_relations=sub_object_relations(
        [
            above(_TRUCK_CAB_BODY, _TRUCK_CAB_TIRES),
            contacts(_TRUCK_CAB_BODY, _TRUCK_CAB_TIRES),
            supports(_TRUCK_CAB_TIRES, _TRUCK_CAB_BODY),
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
    sub_object_relations=sub_object_relations(
        [
            contacts(_TRUCK_TRAILER_FLATBED, _TRUCK_TRAILER_TIRES),
            supports(_TRUCK_TRAILER_TIRES, _TRUCK_TRAILER_FLATBED),
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
    sub_object_relations=sub_object_relations(
        [
            contacts(_TRUCK_SCHEMA_CAB, _TRUCK_SCHEMA_TRAILER),
            bigger_than(_TRUCK_SCHEMA_TRAILER, _TRUCK_SCHEMA_CAB),
        ]
    ),
)

_PUT_AGENT = SituationObject(THING, properties=[ANIMATE])
_PUT_THEME = SituationObject(THING)
_PUT_GOAL = SituationObject(THING)
_PUT_MANIPULATOR = SituationObject(THING, properties=[CAN_MANIPULATE_OBJECTS])

_PUT_ACTION_DESCRIPTION = ActionDescription(
    frames=[
        ActionDescriptionFrame(
            {AGENT: _PUT_AGENT, THEME: _PUT_THEME, DESTINATION: _PUT_GOAL}
        )
    ],
    preconditions=[
        SituationRelation(SMALLER_THAN, _PUT_THEME, _PUT_AGENT),
        # TODO: that theme is not already located in GOAL
        # SituationRelation(PART_OF, _PUT_MANIPULATOR, _PUT_AGENT),
        # SituationRelation(CONTACTS, _PUT_MANIPULATOR, _PUT_THEME),
        # SituationRelation(SUPPORTS, _PUT_MANIPULATOR, _PUT_THEME),
    ],
    postconditions=[
        # TODO: that theme is located in GOAL
        # SituationRelation(CONTACTS, _PUT_MANIPULATOR, _PUT_THEME, negated=True),
        # SituationRelation(SUPPORTS, _PUT_MANIPULATOR, _PUT_THEME, negated=True),
        SituationRelation(CONTACTS, _PUT_THEME, _PUT_GOAL)
    ],
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
        ]
    ),
    action_to_description=immutabledict([(PUT, _PUT_ACTION_DESCRIPTION)]),
)
