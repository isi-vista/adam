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
from typing import Tuple

from networkx import DiGraph

from adam.ontology import (
    OntologyProperty,
    OntologyNode,
    Ontology,
    HierarchicalObjectSchema,
    SubObject,
    SubObjectRelation,
    sub_object_relations,
)

ANIMATE = OntologyProperty("animate")
INANIMATE = OntologyProperty("inanimate")
SENTIENT = OntologyProperty("sentient")

RECOGNIZED_PARTICULAR = OntologyProperty("recognized-particular")
"""
Indicates that a node in the ontology corresponds to a particular (rather than a class)
which is assumed to be known to the `LanguageLearner`. 
The prototypical cases here are *Mom* and *Dad*.
"""

_ontology_graph = DiGraph()  # pylint:disable=invalid-name


def subtype(sub: OntologyNode, _super: OntologyNode) -> None:
    _ontology_graph.add_edge(sub, _super)


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

PERSON = OntologyNode("person", [ANIMATE])
subtype(PERSON, PHYSICAL_OBJECT)
MOM = OntologyNode("mom", [RECOGNIZED_PARTICULAR])
subtype(MOM, PERSON)
DAD = OntologyNode("dad", [RECOGNIZED_PARTICULAR])
subtype(DAD, PERSON)

# head, arm, torso, and leg are not part of the phase 1
# vocabulary but are useful for describing the structure
# of PERSON
HEAD = OntologyNode("head")
ARM = OntologyNode("arm")
TORSO = OntologyNode("torso")
LEG = OntologyNode("leg")

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


def bigger_than(obj1: SubObject, obj2: SubObject) -> Tuple[SubObjectRelation, ...]:
    return (
        SubObjectRelation(BIGGER_THAN, obj1, obj2),
        SubObjectRelation(SMALLER_THAN, obj2, obj1),
    )


SUPPORTS = OntologyNode("supports")
"""
A relation indicating that  one object provides the force to counteract gravity and prevent another 
object from falling.
"""
subtype(SUPPORTS, SPATIAL_RELATION)


def supports(obj1: SubObject, obj2: SubObject) -> Tuple[SubObjectRelation, ...]:
    """
    Convenience method for indicating that one `SubObject` in a `HierarchicalObjectSchema` has a `SUPPORTS` relation with another.

    For us with `SubObjectRelation`.

    Args:
        *obj1*: The `SubObject` which supports obj2
        *obj2*: The `SubObject` being supported
    Returns:
        Tuple[`SubObjectRelation`,...] see `SubObjectRelation` for more information
    """
    return (SubObjectRelation(SUPPORTS, obj1, obj2),)


CONTACTS = OntologyNode("contacts")
"""
A symmetric relation indicating that one object touches another.
"""
subtype(CONTACTS, SPATIAL_RELATION)


def contacts(obj1: SubObject, obj2: SubObject) -> Tuple[SubObjectRelation, ...]:
    """
    Convenience methord for indicating that one `SubObject` in a `HierarchicalObjectSchema` has a `CONTACTS` relation with another.

    For us with `SubObjectRelation`.

    Args:
        *obj1*: The `SubObject` which has a reciprocal contacts relationship with
        *obj2*: The `SubObject` which contacts obj1

    Returns:
        Tuple[`SubObjectRelation`,...] see `SubObjectRelation` for more information
    """
    return (
        SubObjectRelation(CONTACTS, obj1, obj2),
        SubObjectRelation(CONTACTS, obj2, obj1),
    )


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


def above(obj1: SubObject, obj2: SubObject) -> Tuple[SubObjectRelation, ...]:
    """
    Convenience methord for indicating that one `SubObject` in a `HierarchicalObjectSchema` has a `ABOVE` relation with another.

    When one entity is above another, the inverse is also true. This function provides the implicit
    inverse assertion for hierarchical objects.

    For us with `SubObjectRelation`.

    Args:
        obj1: The `SubObject` which is above obj2
        obj2: The `SubObject` which is below obj1

    Returns:
        Tuple[`SubObjectRelation`,...] see `SubObjectRelation` for more information
    """
    return (SubObjectRelation(ABOVE, obj1, obj2), SubObjectRelation(BELOW, obj2, obj1))


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

HEAD_SCHEMA = HierarchicalObjectSchema(HEAD)
TORSO_SCHEMA = HierarchicalObjectSchema(TORSO)
ARM_SCHEMA = HierarchicalObjectSchema(ARM)
LEG_SCHEMA = HierarchicalObjectSchema(LEG)

# schemata describing the hierarchical physical structure of objects
_PERSON_SCHEMA_HEAD = SubObject(HEAD_SCHEMA)
_PERSON_SCHEMA_TORSO = SubObject(TORSO_SCHEMA)
_PERSON_SCHEMA_LEFT_ARM = SubObject(ARM_SCHEMA)
_PERSON_SCHEMA_RIGHT_ARM = SubObject(ARM_SCHEMA)
_PERSON_SCHEMA_LEFT_LEG = SubObject(LEG_SCHEMA)
_PERSON_SCHEMA_RIGHT_LEG = SubObject(LEG_SCHEMA)


PERSON_SCHEMA = HierarchicalObjectSchema(
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
            # relation of head to torso
            contacts(_PERSON_SCHEMA_HEAD, _PERSON_SCHEMA_TORSO),
            supports(_PERSON_SCHEMA_TORSO, _PERSON_SCHEMA_HEAD),
            above(_PERSON_SCHEMA_HEAD, _PERSON_SCHEMA_TORSO),
            bigger_than(_PERSON_SCHEMA_TORSO, _PERSON_SCHEMA_HEAD),
            # relation of limbs to torso
            contacts(_PERSON_SCHEMA_TORSO, _PERSON_SCHEMA_LEFT_ARM),
            contacts(_PERSON_SCHEMA_TORSO, _PERSON_SCHEMA_LEFT_ARM),
            contacts(_PERSON_SCHEMA_TORSO, _PERSON_SCHEMA_LEFT_LEG),
            contacts(_PERSON_SCHEMA_TORSO, _PERSON_SCHEMA_RIGHT_LEG),
        ]
    ),
)

GAILA_PHASE_1_ONTOLOGY = Ontology.from_directed_graph(_ontology_graph)
