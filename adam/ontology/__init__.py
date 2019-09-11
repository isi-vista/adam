r"""
Representations for simple ontologies.

These ontologies are intended to be used when describing `Situation`\ s and writing `SituationTemplate`\ s.
"""
from typing import Callable, Iterable, List, Tuple, Union, AbstractSet

from attr import attrib, attrs
from attr.validators import instance_of
from immutablecollections import (
    ImmutableSet,
    ImmutableSetMultiDict,
    immutableset,
    immutablesetmultidict,
)
from immutablecollections.converter_utils import (
    _to_immutableset,
    _to_immutablesetmultidict,
)
from more_itertools import flatten
from networkx import DiGraph, dfs_preorder_nodes, has_path, simple_cycles
from vistautils.preconditions import check_arg


# convenience method for use in Ontology
def _copy_digraph(digraph: DiGraph) -> DiGraph:
    return digraph.copy()


@attrs(frozen=True, slots=True)
class Ontology:
    r"""
    A hierarchical collection of types for objects, actions, etc.

    Types are represented by `OntologyNode`\ s with parent-child relationships.

    Every `OntologyNode` may have a set of properties which are inherited by all child nodes.

    Every `Ontology` must contain the special nodes `THING`, `RELATION`, `ACTION`,
    `PROPERTY`, `META_PROPERTY`, and `ABSTRACT`.
    To assist in creating legal `Ontology`\ s, we provide `minimal_ontology_graph`.
    """

    _graph: DiGraph = attrib(validator=instance_of(DiGraph), converter=_copy_digraph)
    structural_schemata: ImmutableSetMultiDict[
        "OntologyNode", "ObjectStructuralSchema"
    ] = attrib(converter=_to_immutablesetmultidict, default=immutablesetmultidict())

    def __attrs_post_init__(self) -> None:
        for cycle in simple_cycles(self._graph):
            raise ValueError(f"The ontology graph may not have cycles but got {cycle}")
        for required_node in REQUIRED_ONTOLOGY_NODES:
            check_arg(
                required_node in self,
                f"Ontology lacks required {required_node.handle} node",
            )

    @staticmethod
    def from_directed_graph(
        graph: DiGraph,
        structural_schemata: ImmutableSetMultiDict[
            "OntologyNode", "ObjectStructuralSchema"
        ] = immutablesetmultidict(),
    ) -> "Ontology":
        r"""
        Create an `Ontology` from an acyclic NetworkX ::class`DiGraph`.

        Args:
            graph: a NetworkX graph representing the ontology.  Sub-class
                `OntologyNode`\ s should have edges pointing to super-class `OntologyNode`\ s.
                If the graph is not acyclic, the result is undefined.

        Returns:
            The `Ontology` encoding the relationships in the `networkx.DiGraph`.
        """
        return Ontology(graph, structural_schemata)

    def is_subtype_of(
        self, node: "OntologyNode", query_supertype: "OntologyNode"
    ) -> bool:
        """
        Determines whether *node* is a sub-type of *query_supertype*.
        """
        # graph edges run from sub-types to super-types
        return has_path(self._graph, node, query_supertype)

    def nodes_with_properties(
        self,
        root_node: "OntologyNode",
        required_properties: Iterable["OntologyNode"],
        *,
        banned_properties: AbstractSet["OntologyNode"] = immutableset(),
    ) -> ImmutableSet["OntologyNode"]:
        r"""
        Get all `OntologyNode`\ s which are a dominated by *root_node* (or are *root_node*
        itself) which possess all the *required_properties* and none of the *banned_properties*,
        either directly or by inheritance from a dominating node.

        Args:
            root_node: the node to search the ontology tree at and under
            required_properties: the properties (as `OntologyNode`\ s) every returned node must have
            banned_properties: the properties (as `OntologyNode`\ s) which no returned node may
                               have

        Returns:
             All `OntologyNode`\ s which are a dominated by *root_node* (or are *root_node*
             itself) which possess all the *required_properties* and none of the
             *banned_properties*, either directly or by inheritance from a dominating node.
        """

        if root_node not in self._graph:
            raise RuntimeError(
                f"Cannot get object with type {root_node} because it does not "
                f"appear in the ontology {self}"
            )

        return immutableset(
            node
            for node in dfs_preorder_nodes(self._graph.reverse(copy=False), root_node)
            if self.has_all_properties(
                node, required_properties, banned_properties=banned_properties
            )
        )

    def has_all_properties(
        self,
        node: "OntologyNode",
        required_properties: Iterable["OntologyNode"],
        *,
        banned_properties: AbstractSet["OntologyNode"] = immutableset(),
    ) -> bool:
        r"""
        Checks an `OntologyNode` for a collection of `OntologyNode`\ s.

        Args:
            node: the `OntologyNode` being inquired about
            required_properties: the `OntologyNode`\ s being inquired about
            banned_properties: this function will return false if *node* contains any of these
                               properties. Defaults to the empty set.

        Returns:
            Whether *node* possesses all of *required_properties* and none of *banned_properties*,
            either directly or via inheritance from a dominating node.
        """
        if not required_properties and not banned_properties:
            return True

        node_properties = self.properties_for_node(node)
        return all(
            property_ in node_properties for property_ in required_properties
        ) and not any(property_ in banned_properties for property_ in node_properties)

    def properties_for_node(self, node: "OntologyNode") -> ImmutableSet["OntologyNode"]:
        r"""
        Get all properties a `OntologyNode` possesses.

        Args:
            node: the `OntologyNode` whose properties you want.

        Returns:
            All properties `OntologyNode` possesses, whether directly or by inheritance from a
            dominating node.
        """
        node_properties: List[OntologyNode] = list(node.non_inheritable_properties)

        cur_node = node
        while cur_node:
            node_properties.extend(cur_node.inheritable_properties)
            # need to make a tuple because we can't len() the returned iterator
            parents = tuple(self._graph.successors(cur_node))
            if len(parents) == 1:
                cur_node = parents[0]
            elif parents:
                raise RuntimeError(
                    f"Found multiple parents for ontology node {node}, which is "
                    f"not yet supported"
                )
            else:
                # we have reached a root
                break
        return immutableset(node_properties)

    def __contains__(self, item: "OntologyNode") -> bool:
        return item in self._graph.nodes


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

REQUIRED_ONTOLOGY_NODES = immutableset(
    [THING, RELATION, ACTION, PROPERTY, META_PROPERTY, ABSTRACT]
)


def minimal_ontology_graph():
    """
    Get the NetworkX DiGraph corresponding to the minimum legal ontology,
    containing all required nodes.

    This is useful as a convenient foundation for building your own ontologies.
    """
    ret = DiGraph()
    for node in REQUIRED_ONTOLOGY_NODES:
        ret.add_node(node)
    return ret


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


def make_symetric_dsl_relation(
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
