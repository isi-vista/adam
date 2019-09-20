from itertools import chain
from typing import AbstractSet, Iterable, List

from attr import attrib, attrs
from attr.validators import instance_of
from immutablecollections import (
    ImmutableDict,
    ImmutableSet,
    ImmutableSetMultiDict,
    immutabledict,
    immutableset,
    immutablesetmultidict,
)
from immutablecollections.converter_utils import (
    _to_immutabledict,
    _to_immutablesetmultidict,
    _to_immutableset,
)
from networkx import DiGraph, ancestors, dfs_preorder_nodes, has_path, simple_cycles
from vistautils.preconditions import check_arg

from adam.ontology import (
    CAN_FILL_TEMPLATE_SLOT,
    OntologyNode,
    REQUIRED_ONTOLOGY_NODES,
    THING,
)
from adam.ontology.structural_schema import ObjectStructuralSchema

# convenience method for use in Ontology
from adam.ontology.action_description import ActionDescription
from adam.relation import Relation


def _copy_digraph(digraph: DiGraph) -> DiGraph:
    return digraph.copy()


@attrs(frozen=True, slots=True)
class Ontology:
    r"""
    A hierarchical collection of types for objects, actions, etc.

    Types are represented by `OntologyNode`\ s with parent-child relationships.

    Every `OntologyNode` may have a set of properties which are inherited by all child nodes.

    Every `Ontology` must contain the special nodes `THING`, `RELATION`, `ACTION`,
    `PROPERTY`, `META_PROPERTY`, and `CAN_FILL_TEMPLATE_SLOT`.

    An `Ontology` must have an `ObjectStructuralSchema` associated with
    each `CAN_FILL_TEMPLATE_SLOT` `THING`.  `ObjectStructuralSchema`\ ta are inherited,
    but any which are explicitly-specified will cause any inherited schemata
    to be ignored.

    To assist in creating legal `Ontology`\ s, we provide `minimal_ontology_graph`.
    """

    _graph: DiGraph = attrib(validator=instance_of(DiGraph), converter=_copy_digraph)
    _structural_schemata: ImmutableSetMultiDict[
        "OntologyNode", "ObjectStructuralSchema"
    ] = attrib(converter=_to_immutablesetmultidict, default=immutablesetmultidict())
    action_to_description: ImmutableDict[OntologyNode, ActionDescription] = attrib(
        converter=_to_immutabledict, default=immutabledict(), kw_only=True
    )
    relations: ImmutableSet[Relation[OntologyNode]] = attrib(
        converter=_to_immutableset, default=immutableset(), kw_only=True
    )
    subjects_to_relations: ImmutableSetMultiDict[
        OntologyNode, Relation[OntologyNode]
    ] = attrib(init=False)

    def __attrs_post_init__(self) -> None:
        for cycle in simple_cycles(self._graph):
            raise ValueError(f"The ontology graph may not have cycles but got {cycle}")

        for required_node in REQUIRED_ONTOLOGY_NODES:
            check_arg(
                required_node in self,
                f"Ontology lacks required {required_node.handle} node",
            )

        # every sub-type of THING must either have a structural schema
        # or be a sub-type of something with a structural schema
        for thing_node in dfs_preorder_nodes(self._graph.reverse(copy=False), THING):
            if not any(
                node in self._structural_schemata for node in self.ancestors(thing_node)
            ):
                if self.has_all_properties(thing_node, [CAN_FILL_TEMPLATE_SLOT]):
                    raise RuntimeError(
                        f"No structural schema is available for {thing_node}"
                    )

    def ancestors(self, node: OntologyNode) -> Iterable[OntologyNode]:
        return chain([node], ancestors(self._graph.reverse(copy=False), node))

    def structural_schemata(
        self, node: OntologyNode
    ) -> AbstractSet[ObjectStructuralSchema]:
        for node in self.ancestors(node):
            if node in self._structural_schemata:
                return self._structural_schemata[node]
        return immutableset()

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

    @subjects_to_relations.default
    def _subjects_to_relations(
        self
    ) -> ImmutableSetMultiDict[OntologyNode, Relation[OntologyNode]]:
        return immutablesetmultidict(
            (relation.first_slot, relation) for relation in self.relations
        )
