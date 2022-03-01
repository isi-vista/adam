from typing import Any, Callable

from attr import attrs, attrib
from immutablecollections import ImmutableSet
from immutablecollections.converter_utils import _to_immutableset
from networkx import DiGraph


def digraph_with_nodes_sorted_by(
    graph: DiGraph, sort_key: Callable[[Any], Any]
) -> DiGraph:
    """
    Get a `DiGraph` identical to `graph` except that the iteration order of its nodes
    is according to *sort_key*.

    *sort_key* should expect to receive a 2-tuple of the node itself
    and its NetworkX node attribute dictionary
    and should return a value to sort by,
    just like a regular Python sort key function.
    """
    new_graph = DiGraph()

    # add edges from the old graph to the new graph,
    # in the order specified by sort_key
    for (node, node_data) in sorted(graph.nodes(data=True), key=sort_key):
        new_graph.add_node(node, **node_data)

    # copy all edges from the original graph
    for (source, dest, data) in graph.edges(data=True):
        new_graph.add_edge(source, dest, **data)

    return new_graph


@attrs
class ShowNodes:
    """
    See `subgraph` for more information
    """

    nodes: ImmutableSet[Any] = attrib(converter=_to_immutableset)

    def __call__(self, node):
        return node in self.nodes


def copy_digraph(digraph: DiGraph) -> DiGraph:
    """
    We need this for use as an attrs converter.
    """
    return digraph.copy()
