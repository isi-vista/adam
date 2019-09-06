r"""
Representations for simple ontologies.

These ontologies are intended to be used when describing `Situation`\ s and writing `SituationTemplate`\ s.
"""
from typing import Iterable, List

from attr import attrib, attrs
from attr.validators import instance_of
from immutablecollections import (
    ImmutableSet,
    immutableset,
    ImmutableSetMultiDict,
    immutablesetmultidict,
)
from immutablecollections.converter_utils import (
    _to_immutableset,
    _to_immutablesetmultidict,
)
from more_itertools import flatten
from networkx import DiGraph, dfs_preorder_nodes, simple_cycles


@attrs(frozen=True, slots=True)
class Ontology:
    r"""
    A hierarchical collection of types for objects, actions, etc.

    Types are represented by `OntologyNode`\ s with parent-child relationships.

    Every `OntologyNode` may have a set of properties which are inherited by all child nodes.
    """

    _graph: DiGraph = attrib(validator=instance_of(DiGraph))
    hierarchical_object_schemata: ImmutableSetMultiDict[
        "OntologyNode", "HierarchicalObjectSchema"
    ] = attrib(converter=_to_immutablesetmultidict, default=immutablesetmultidict())

    def __attrs_post_init__(self) -> None:
        for cycle in simple_cycles(self._graph):
            raise ValueError(f"The ontology graph may not have cycles but got {cycle}")

    @staticmethod
    def from_directed_graph(graph: DiGraph) -> "Ontology":
        r"""
        Create an `Ontology` from an acyclic NetworkX ::class`DiGraph`.

        Args:
            graph: a NetworkX graph representing the ontology.  Sub-class
                `OntologyNode`\ s should have edges pointing to super-class `OntologyNode`\ s.
                If the graph is not acyclic, the result is undefined.

        Returns:
            The `Ontology` encoding the relationships in the `networkx.DiGraph`.
        """
        return Ontology(graph)

    def nodes_with_properties(
        self, root_node: "OntologyNode", required_properties: Iterable["OntologyProperty"]
    ) -> ImmutableSet["OntologyNode"]:
        r"""
        Get all `OntologyNode`\ s which are a dominated by *root_node* (or are *root_node*
        itself) which possess all the *required_properties*, either directly or by inheritance
        from a dominating node.

        Args:
            root_node: the node to search the ontology tree at and under
            required_properties: the `OntologyProperty`\ s every returned node must have

        Returns:
             All `OntologyNode`\ s which are a dominated by *root_node* (or are *root_node*
             itself) which possess all the *required_properties*, either directly or by inheritance
             from a dominating node.
        """

        if root_node not in self._graph:
            raise RuntimeError(
                f"Cannot get object with type {root_node} because it does not "
                f"appear in the ontology {self}"
            )

        return immutableset(
            node
            for node in dfs_preorder_nodes(self._graph, root_node)
            if self.has_all_properties(node, required_properties)
        )

    def has_all_properties(
        self, node: "OntologyNode", required_properties: Iterable["OntologyProperty"]
    ) -> bool:
        r"""
        Checks an `OntologyNode` for a collection of `OntologyProperty`\ s.

        Args:
            node: the `OntologyNode` being inquired about
            required_properties: the `OntologyProperty`\ s being inquired about

        Returns:
            Whether *node* possesses all of *required_properties*, either directly or via
            inheritance from a dominating node.
        """
        node_properties = self.properties_for_node(node)
        return all(property_ in node_properties for property_ in required_properties)

    def properties_for_node(
        self, node: "OntologyNode"
    ) -> ImmutableSet["OntologyProperty"]:
        r"""
        Get all properties a `OntologyNode` possesses.

        Args:
            node: the `OntologyNode` whose properties you want.

        Returns:
            All properties `OntologyNode` possesses, whether directly or by inheritance from a
            dominating node.
        """
        node_properties: List[OntologyProperty] = []

        cur_node = node
        while cur_node:
            # noinspection PyProtectedMember
            for (
                property_
            ) in cur_node._local_properties:  # pylint:disable=protected-access
                node_properties.append(property_)
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
    _local_properties: ImmutableSet["OntologyProperty"] = attrib(
        converter=_to_immutableset, default=immutableset()
    )
    r"""
    `OntologyProperty`\ s of this `OntologyNode`.
    These will be inherited by its children.
    """

    def __repr__(self) -> str:
        if self._local_properties:
            local_properties = ",".join(
                str(local_property) for local_property in self._local_properties
            )
            properties_string = f"[{local_properties}]"
        else:
            properties_string = ""
        return f"{self.handle}{properties_string}"


@attrs(frozen=True, slots=True, repr=False)
class OntologyProperty:
    r"""
    A property which a node in an `Ontology` may bear, such as "animate".
    """

    _handle: str = attrib(validator=instance_of(str))
    """
    A simple human-readable description of this property,
    used for debugging and testing only.
    """
    perceivable: bool = attrib(default=False, kw_only=True)
    """
    A boolean indicator whether an ontological property is perceivable for the learner.
    E.g. sentience and animacy, which are innately known. 
    """

    def __repr__(self) -> str:
        return f"+{self._handle},{self.perceivable}"


@attrs(frozen=True, slots=True, repr=False)
class HierarchicalObjectSchema:
    r"""
    A hierarchical representation of the internal structure of some type of object.

    A `HierarchicalObjectSchema` represents the general pattern of the structure of an object,
    rather than the structure of any particular object
    (e.g. people in general, rather than a particular person).

    For example a person's body is made up of a head, torso, left arm, right arm, left leg, and
    right leg. These sub-objects have various relations to one another
    (e.g. the head is above and supported by the torso).

    `HierarchicalObjectSchema` can be verbose see `SubObjectRelation`\ s for additional details.
    """

    parent_object: OntologyNode = attrib(validator=instance_of(OntologyNode))
    """
    The `OntologyNode` this `HierarchicalObjectSchema` represents the structure of.
    """
    sub_objects: ImmutableSet["SubObject"] = attrib(
        converter=_to_immutableset, default=immutableset()
    )
    r"""
    The component parts which make up an object of the type *parent_object*.
    
    These `SubObject`\ s themselves wrap `HierarchicalObjectSchema`\ s 
    and can therefore themselves have complex internal structure.
    """
    sub_object_relations: ImmutableSet["SubObjectRelation"] = attrib(
        converter=_to_immutableset, default=immutableset()
    )
    r"""
    A set of `SubObjectRelation` which define how the `SubObject`\ s relate to one another. 
    """


@attrs(frozen=True, slots=True, repr=False)
class SubObject:
    r"""
    A sub-component of a generic type of object.

    This is for use only in constructing `HierarchicalObjectSchema`\ ta.
    """

    schema: HierarchicalObjectSchema = attrib()
    """
    The `HierarchicalObjectSchema` describing the internal structure of this sub-component.
    
    For example, an ARM is a sub-component of a PERSON, but ARM itself has a complex structure
    (e.g. it includes a hand)
    """


@attrs(frozen=True, slots=True, repr=False)
class SubObjectRelation:
    """
    This class defines the relationships between `SubObject` of a `HierarchicalObjectSchema`
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
    Convenience method to enable writing sub-object relations in a `HierarchicalObjectSchema` more easily.

    This method simply flattens collections of items in the input iterable.

    This is useful because it allows you to write methods for your relations which produce
    collections of relations as their output. This allows you to use such DSL-like methods to
    enforce constraints between the relations.

    Please see adam.ontology.phase1_ontology.PERSON_SCHEMA for an example of how this is useful.
    """
    return immutableset(flatten(relation_collections))
