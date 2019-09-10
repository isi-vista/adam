r"""
Representations for simple ontologies.

These ontologies are intended to be used when describing `Situation`\ s and writing `SituationTemplate`\ s.
"""
from typing import Callable, Iterable, Tuple, Union

from attr import attrib, attrs
from attr.validators import instance_of
from immutablecollections import (
    ImmutableSet,
    immutableset,
)
from immutablecollections.converter_utils import (
    _to_immutableset,
)
from more_itertools import flatten


@attrs(frozen=True, slots=True, repr=False)
class OntologyNode:
    r"""
    A node in an ontology representing some type of object, action, or relation, such as
    "animate object" or "transfer action."

    An `OntologyNode` has a *handle*, which is a user-facing description used for debugging
    and testing only.

    It may also have a set of *local_properties* which are inherited by all child nodes.
    """

    handle: str = attrib(validator=instance_of(str))
    """
    A simple human-readable description of this node,
    used for debugging and testing only.
    """
    inheritable_properties: ImmutableSet["OntologyNode"] = attrib(
        converter=_to_immutableset, default=immutableset()
    )
    r"""
    Properties of the `OntologyNode`, as a set of `OntologyNode`\ s
    which should be inherited by its children.
    """
    non_inheritable_properties: ImmutableSet["OntologyNode"] = attrib(
        converter=_to_immutableset, default=immutableset(), kw_only=True
    )
    r"""
    Properties of the `OntologyNode`, as a set of `OntologyNode`\ s
    which should not be inherited by its children.
    """

    def __repr__(self) -> str:
        if self.inheritable_properties:
            local_properties = ",".join(
                str(local_property) for local_property in self.inheritable_properties
            )
            properties_string = f"[{local_properties}]"
        else:
            properties_string = ""
        return f"{self.handle}{properties_string}"


# by convention, the following should appear in all Ontologies
ABSTRACT = OntologyNode("abstract")
r"""
A property indicating that a node can't be instantiated in a scene.
"""

THING = OntologyNode("thing")
r"""
Ancestor of all objects in an `Ontology`.

By convention this should appear in all `Ontology`\ s.
"""
RELATION = OntologyNode("relation")
r"""
Ancestor of all relations in an `Ontology`.

By convention this should appear in all `Ontology`\ s.
"""
ACTION = OntologyNode("action")
r"""
Ancestor of all actions in an `Ontology`.

By convention this should appear in all `Ontology`\ s.
"""
PROPERTY = OntologyNode("property")
r"""
Ancestor of all properties in an `Ontology`.

By convention this should appear in all `Ontology`\ s.
"""

META_PROPERTY = OntologyNode("meta-property")
r"""
A property of a property.

For example, whether it is perceivable or binary.

By convention this should appear in all `Ontology`\ s.
"""

REQUIRED_ONTOLOGY_NODES = immutableset([THING, RELATION, ACTION, PROPERTY, META_PROPERTY,
                                        ABSTRACT])


@attrs(frozen=True, slots=True, repr=False)
class ObjectStructuralSchema:
    r"""
    A hierarchical representation of the internal structure of some type of object.

    An `ObjectStructuralSchema` represents the general pattern of the structure of an object,
    rather than the structure of any particular object
    (e.g. people in general, rather than a particular person).

    For example a person's body is made up of a head, torso, left arm, right arm, left leg, and
    right leg. These sub-objects have various relations to one another
    (e.g. the head is above and supported by the torso).

    Declaring an `ObjectStructuralSchema` can be verbose;
     see `SubObjectRelation`\ s for additional tips on how to make this more compact.
    """

    parent_object: OntologyNode = attrib(validator=instance_of(OntologyNode))
    """
    The `OntologyNode` this `ObjectStructuralSchema` represents the structure of.
    """
    sub_objects: ImmutableSet["SubObject"] = attrib(
        converter=_to_immutableset, default=immutableset()
    )
    r"""
    The component parts which make up an object of the type *parent_object*.
    
    These `SubObject`\ s themselves wrap `ObjectStructuralSchema`\ s 
    and can therefore themselves have complex internal structure.
    """
    sub_object_relations: ImmutableSet["SubObjectRelation"] = attrib(
        converter=_to_immutableset, default=immutableset()
    )
    r"""
    A set of `SubObjectRelation` which define how the `SubObject`\ s relate to one another. 
    """


# need cmp=False to keep otherwise identical sub-components distinct
# (e.g. left arm, right arm)
@attrs(frozen=True, slots=True, repr=False, cmp=False)
class SubObject:
    r"""
    A sub-component of a generic type of object.

    This is for use only in constructing `ObjectStructuralSchema`\ ta.
    """

    schema: ObjectStructuralSchema = attrib()
    """
    The `ObjectStructuralSchema` describing the internal structure of this sub-component.
    
    For example, an ARM is a sub-component of a PERSON, but ARM itself has a complex structure
    (e.g. it includes a hand)
    """


@attrs(frozen=True, slots=True, repr=False)
class SubObjectRelation:
    """
    This class defines the relationships between `SubObject` of a `ObjectStructuralSchema`
    """

    relation_type: OntologyNode = attrib(validator=instance_of(OntologyNode))
    """
    An `OntologyNode` which gives the relationship type between the args
    """
    arg1: SubObject = attrib()
    """
    A `SubObject` which is the first argument of the relation_type
    """
    arg2: SubObject = attrib()
    """
    A `SubObject` which is the second argument of the relation_type
    """


# DSL to make writing object hierarchies easier
def sub_object_relations(
    relation_collections: Iterable[Iterable[SubObjectRelation]]
) -> ImmutableSet[SubObjectRelation]:
    """
    Convenience method to enable writing sub-object relations
    in an `ObjectStructuralSchema` more easily.

    This method simply flattens collections of items in the input iterable.

    This is useful because it allows you to write methods for your relations which produce
    collections of relations as their output. This allows you to use such DSL-like methods to
    enforce constraints between the relations.

    Please see adam.ontology.phase1_ontology.PERSON_SCHEMA for an example of how this is useful.
    """
    return immutableset(flatten(relation_collections))


_OneOrMoreSubObjects = Union[SubObject, Iterable[SubObject]]


def make_dsl_relation(
    relation_type: OntologyNode
) -> Callable[
    [_OneOrMoreSubObjects, _OneOrMoreSubObjects], Tuple[SubObjectRelation, ...]
]:
    r"""
    Make a function which, when given either single or groups
    of sub-object arguments for two slots of a relation,
    generates `SubObjectRelation`\ s of type *relation_type*
    for the cross-product of the arguments.

    See `adam.ontology.phase1_ontology` for many examples.
    """

    def dsl_relation_function(
        arg1s: _OneOrMoreSubObjects, arg2s: _OneOrMoreSubObjects
    ) -> Tuple[SubObjectRelation, ...]:
        if isinstance(arg1s, SubObject):
            arg1s = (arg1s,)
        if isinstance(arg2s, SubObject):
            arg2s = (arg2s,)
        return tuple(
            SubObjectRelation(relation_type, arg1, arg2)
            for arg1 in arg1s
            for arg2 in arg2s
        )

    return dsl_relation_function


def make_symmetric_dsl_relation(
    relation_type: OntologyNode
) -> Callable[
    [_OneOrMoreSubObjects, _OneOrMoreSubObjects], Tuple[SubObjectRelation, ...]
]:
    r"""
    Make a function which, when given either single or groups
    of sub-object arguments for two slots of a relation,
    generates a symmetric `SubObjectRelation`\ s of type *relation_type*
    for the cross-product of the arguments.

    See `adam.ontology.phase1_ontology` for many examples.
    """

    def dsl_symetric_function(
        arg1s: _OneOrMoreSubObjects, arg2s: _OneOrMoreSubObjects
    ) -> Tuple[SubObjectRelation, ...]:
        if isinstance(arg1s, SubObject):
            arg1s = (arg1s,)
        if isinstance(arg2s, SubObject):
            arg2s = (arg2s,)
        return flatten(
            [
                tuple(
                    SubObjectRelation(relation_type, arg1, arg2)
                    for arg1 in arg1s
                    for arg2 in arg2s
                ),
                tuple(
                    SubObjectRelation(relation_type, arg2, arg1)
                    for arg2 in arg2s
                    for arg1 in arg1s
                ),
            ]
        )

    return dsl_symetric_function


def make_opposite_dsl_relation(
    relation_type: OntologyNode, *, opposite_type: OntologyNode
) -> Callable[
    [_OneOrMoreSubObjects, _OneOrMoreSubObjects], Tuple[SubObjectRelation, ...]
]:
    r"""
    Make a function which, when given either single or groups
    of sub-object arguments for two slots of a relation,
    generates a  `SubObjectRelation`\ s of type *relation_type*
    and an inverse of type *opposite_type* for the for the
    reversed cross-product of the arguments

    See `adam.ontology.phase1_ontology` for many examples.
    """

    def dsl_opposite_function(
        arg1s: _OneOrMoreSubObjects, arg2s: _OneOrMoreSubObjects
    ) -> Tuple[SubObjectRelation, ...]:
        if isinstance(arg1s, SubObject):
            arg1s = (arg1s,)
        if isinstance(arg2s, SubObject):
            arg2s = (arg2s,)
        return flatten(
            [
                tuple(
                    SubObjectRelation(relation_type, arg1, arg2)
                    for arg1 in arg1s
                    for arg2 in arg2s
                ),
                tuple(
                    SubObjectRelation(opposite_type, arg2, arg1)
                    for arg1 in arg1s
                    for arg2 in arg2s
                ),
            ]
        )

    return dsl_opposite_function
