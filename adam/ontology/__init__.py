r"""
Representations for simple ontologies.

These ontologies are intended to be used when describing `Situation`\ s and writing
`SituationTemplate`\ s.
"""
import itertools
from typing import Iterable, List

from attr import attrib, attrs
from attr.validators import instance_of
from immutablecollections import ImmutableSet, immutableset
from immutablecollections.converter_utils import _to_immutableset
from networkx import DiGraph, dfs_preorder_nodes, simple_cycles


@attrs(frozen=True, slots=True)
class Ontology:
    r"""
    A hierarchical collection of types for objects, actions, etc.

    Types are represented by `OntologyNode`\ s with parent-child relationships.

    Every `OntologyNode` may have a set of properties which are inherited by all child nodes.
    """

    _graph: DiGraph = attrib(validator=instance_of(DiGraph))

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
        return all(
            property_ in self.nodes_with_properties(node, required_properties)
            for property_ in required_properties
        )

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

    def __repr__(self) -> str:
        return f"+{self._handle}"
