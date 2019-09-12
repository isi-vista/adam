from typing import Iterable, List

from attr import attrib, attrs
from attr.validators import instance_of
from immutablecollections import (
    ImmutableSet,
    ImmutableSetMultiDict,
    immutableset,
    immutablesetmultidict,
)
from immutablecollections.converter_utils import _to_immutablesetmultidict
from networkx import DiGraph, dfs_preorder_nodes, simple_cycles

from adam.ontology import OntologyNode, ObjectStructuralSchema


@attrs(frozen=True, slots=True)
class Ontology:
    r"""
    A hierarchical collection of types for objects, actions, etc.
    Types are represented by `OntologyNode`\ s with parent-child relationships.
    Every `OntologyNode` may have a set of properties which are inherited by all child nodes.
    """

    _graph: DiGraph = attrib(validator=instance_of(DiGraph))
    structural_schemata: ImmutableSetMultiDict[
        OntologyNode, ObjectStructuralSchema
    ] = attrib(
        converter=_to_immutablesetmultidict, default=immutablesetmultidict(), kw_only=True
    )

    def __attrs_post_init__(self) -> None:
        for cycle in simple_cycles(self._graph):
            raise ValueError(f"The ontology graph may not have cycles but got {cycle}")

    def nodes_with_properties(
        self, root_node: "OntologyNode", required_properties: Iterable["OntologyNode"]
    ) -> ImmutableSet["OntologyNode"]:
        r"""
        Get all `OntologyNode`\ s which are a dominated by *root_node* (or are *root_node*
        itself) which possess all the *required_properties*, either directly or by inheritance
        from a dominating node.
        Args:
            root_node: the node to search the ontology tree at and under
            required_properties: the `OntologyNode`\ s every returned node must have
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
        self, node: "OntologyNode", required_properties: Iterable["OntologyNode"]
    ) -> bool:
        r"""
        Checks an `OntologyNode` for a collection of `OntologyNode`\ s.
        Args:
            node: the `OntologyNode` being inquired about
            required_properties: the `OntologyNode`\ s being inquired about
        Returns:
            Whether *node* possesses all of *required_properties*, either directly or via
            inheritance from a dominating node.
        """
        node_properties = self.properties_for_node(node)
        return all(property_ in node_properties for property_ in required_properties)

    def properties_for_node(self, node: "OntologyNode") -> ImmutableSet["OntologyNode"]:
        r"""
        Get all properties a `OntologyNode` possesses.
        Args:
            node: the `OntologyNode` whose properties you want.
        Returns:
            All properties `OntologyNode` possesses, whether directly or by inheritance from a
            dominating node.
        """
        node_properties: List[OntologyNode] = []

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
