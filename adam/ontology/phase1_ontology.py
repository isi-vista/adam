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
from immutablecollections import immutablesetmultidict
from more_itertools import flatten
from networkx import DiGraph

from adam.ontology import (
    OntologyNode,
    Ontology,
    ObjectStructuralSchema,
    SubObject,
    sub_object_relations,
    make_dsl_relation,
    make_symetric_dsl_relation,
    make_opposite_dsl_relation,
)

_ontology_graph = DiGraph()  # pylint:disable=invalid-name


def subtype(sub: OntologyNode, _super: OntologyNode) -> None:
    _ontology_graph.add_edge(sub, _super)


META_PROPERTY = OntologyNode("meta-property")
PERCEIVABLE = OntologyNode("perceivable")
subtype(PERCEIVABLE, META_PROPERTY)
BINARY = OntologyNode("binary")
subtype(BINARY, META_PROPERTY)

PROPERTY = OntologyNode("property")
ANIMATE = OntologyNode("animate", local_properties=[PERCEIVABLE, BINARY])
subtype(ANIMATE, PROPERTY)
INANIMATE = OntologyNode("inanimate", local_properties=[PERCEIVABLE, BINARY])
subtype(INANIMATE, PROPERTY)
SENTIENT = OntologyNode("sentient", local_properties=[PERCEIVABLE, BINARY])
subtype(SENTIENT, PROPERTY)

RECOGNIZED_PARTICULAR = OntologyNode(
    "recognized-particular", local_properties=[PERCEIVABLE, BINARY]
)
"""
Indicates that a node in the ontology corresponds to a particular (rather than a class)
which is assumed to be known to the `LanguageLearner`. 
The prototypical cases here are *Mom* and *Dad*.
"""
subtype(RECOGNIZED_PARTICULAR, PROPERTY)


COLOR = OntologyNode("color")
RED = OntologyNode("red", local_properties=[COLOR, PERCEIVABLE])
BLUE = OntologyNode("blue", local_properties=[COLOR, PERCEIVABLE])
subtype(RED, COLOR)
subtype(BLUE, COLOR)
RED_RGBS = [
    (255, 0, 0),
    (237, 28, 36),
    (196, 2, 51),
    (242, 0, 60),
    (237, 41, 57),
    (238, 32, 77),
]
BLUE_RGBS = [
    (0, 0, 255),
    (51, 51, 153),
    (0, 135, 189),
    (0, 147, 175),
    (0, 24, 168),
    (31, 117, 254),
]


# Objects
# Information about the hierarchical structure of objects
# is given at the end of this module because it is so bulky.

PHYSICAL_OBJECT = OntologyNode("object")

INANIMATE_OBJECT = OntologyNode("inanimate-object", [INANIMATE])
subtype(INANIMATE_OBJECT, PHYSICAL_OBJECT)
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
WATER = OntologyNode("water")
subtype(WATER, INANIMATE_OBJECT)
JUICE = OntologyNode("juice")
subtype(JUICE, INANIMATE_OBJECT)
CUP = OntologyNode("cup")
subtype(CUP, INANIMATE_OBJECT)
BOX = OntologyNode("box")
subtype(BOX, INANIMATE_OBJECT)
CHAIR = OntologyNode("chair")
subtype(CHAIR, INANIMATE_OBJECT)
HEAD = OntologyNode("head")
subtype(HEAD, INANIMATE_OBJECT)
MILK = OntologyNode("milk")
subtype(MILK, INANIMATE_OBJECT)
HAND = OntologyNode("hand")
subtype(HAND, INANIMATE_OBJECT)
TRUCK = OntologyNode("truck")
subtype(TRUCK, INANIMATE_OBJECT)
DOOR = OntologyNode("door")
subtype(DOOR, INANIMATE_OBJECT)
HAT = OntologyNode("hat")
subtype(HAT, INANIMATE_OBJECT)
COOKIE = OntologyNode("cookie")
subtype(COOKIE, INANIMATE_OBJECT)

PERSON = OntologyNode("person", [ANIMATE])
subtype(PERSON, PHYSICAL_OBJECT)
MOM = OntologyNode("mom", [RECOGNIZED_PARTICULAR])
subtype(MOM, PERSON)
DAD = OntologyNode("dad", [RECOGNIZED_PARTICULAR])
subtype(DAD, PERSON)
BABY = OntologyNode("baby")
subtype(BABY, PERSON)

NONHUMAN_ANIMAL = OntologyNode("animal", [ANIMATE])
subtype(NONHUMAN_ANIMAL, PHYSICAL_OBJECT)
DOG = OntologyNode("dog")
subtype(DOG, NONHUMAN_ANIMAL)
BIRD = OntologyNode("bird")
subtype(BIRD, NONHUMAN_ANIMAL)


# Terms below are internal and can only be accessed as parts of other objects
_HEAD = OntologyNode("head")
_ARM = OntologyNode("arm")
_TORSO = OntologyNode("torso")
_LEG = OntologyNode("leg")
_CHAIR_BACK = OntologyNode("chairback")
_CHAIR_SEAT = OntologyNode("chairseat")
_TABLETOP = OntologyNode("tabletop")
_TAIL = OntologyNode("tail")
_WING = OntologyNode("wing")

# Verbs

ACTION = OntologyNode("action")
PUT = OntologyNode("put")
PUSH = OntologyNode("push")
subtype(PUT, ACTION)
subtype(PUSH, ACTION)

# Relations
# These are used both for situations and in the perceptual representation

RELATION = OntologyNode("relation")
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

# Hierarchical structure of objects
_HEAD_SCHEMA = ObjectStructuralSchema(_HEAD)
_TORSO_SCHEMA = ObjectStructuralSchema(_TORSO)
_ARM_SCHEMA = ObjectStructuralSchema(_ARM)
_LEG_SCHEMA = ObjectStructuralSchema(_LEG)
_CHAIRBACK_SCHEMA = ObjectStructuralSchema(_CHAIR_BACK)
_CHAIR_SEAT_SCHEMA = ObjectStructuralSchema(_CHAIR_SEAT)
_TABLETOP_SCHEMA = ObjectStructuralSchema(_TABLETOP)
_TAIL_SCHEMA = ObjectStructuralSchema(_TAIL)
_WING_SCHEMA = ObjectStructuralSchema(_WING)

# schemata describing the hierarchical physical structure of objects
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

_CHAIR_SCHMEA_BACK = SubObject(_CHAIRBACK_SCHEMA)
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
        _CHAIR_SCHMEA_BACK,
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
            contacts(_CHAIR_SCHMEA_BACK, _CHAIR_SCHEMA_SEAT),
            supports(_CHAIR_SCHEMA_SEAT, _CHAIR_SCHMEA_BACK),
            above(_CHAIR_SCHMEA_BACK, _CHAIR_SCHEMA_SEAT),
        ]
    ),
)

# schemata describing the hierarchical physical structure of objects
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

# schemata describing the hierarchical physical structure of objects
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
            supports(_DOG_SCHEMA_TORSO, _DOG_SCHEMA_HEAD),
            supports(_DOG_SCHEMA_TORSO, _DOG_SCHEMA_TAIL),
            supports(_DOG_LEGS, _DOG_SCHEMA_TORSO),
            above(_DOG_SCHEMA_HEAD, _DOG_SCHEMA_TORSO),
            above(_DOG_SCHEMA_TORSO, _DOG_LEGS),
            bigger_than(_DOG_SCHEMA_TORSO, _DOG_SCHEMA_TAIL),
        ]
    ),
)

# schemata describing the hierarchical physical structure of objects
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
            supports(_BIRD_SCHEMA_TORSO, _BIRD_SCHEMA_HEAD),
            supports(_BIRD_SCHEMA_TORSO, _BIRD_SCHEMA_TAIL),
            supports(_BIRD_SCHEMA_TORSO, _BIRD_WINGS),
        ]
    ),
)

_BALL_SCHEMA = ObjectStructuralSchema(BALL)
_BOX_SCHEMA = ObjectStructuralSchema(BOX)
_WATER_SCHEMA = ObjectStructuralSchema(WATER)
_JUICE_SCHEMA = ObjectStructuralSchema(JUICE)
_BOX_SCHEMA = ObjectStructuralSchema(BOX)
_MILK_SCHEMA = ObjectStructuralSchema(MILK)
_DOOR_SCHEMA = ObjectStructuralSchema(DOOR)
_HAT_SCHEMA = ObjectStructuralSchema(HAT)
_COOKIE_SCHEMA = ObjectStructuralSchema(COOKIE)

GAILA_PHASE_1_ONTOLOGY = Ontology.from_directed_graph(
    _ontology_graph,
    immutablesetmultidict(
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
        ]
    ),
)
