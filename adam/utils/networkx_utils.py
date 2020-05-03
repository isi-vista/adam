from typing import Any, Callable, Iterable, Optional

from attr import attrs, attrib
from immutablecollections import immutableset, ImmutableSet
from immutablecollections.converter_utils import _to_immutableset
from networkx import MultiDiGraph, nx


def digraph_with_nodes_sorted_by(
    graph: MultiDiGraph, sort_key: Callable[[Any], Any]
) -> MultiDiGraph:
    """
    Get a `DiGraph` identical to `graph` except that the iteration order of its nodes
    is according to *sort_key*.

    *sort_key* should expect to receive a 2-tuple of the node itself
    and its NetworkX node attribute dictionary
    and should return a value to sort by,
    just like a regular Python sort key function.
    """
    new_graph = MultiDiGraph()

    # add edges from the old graph to the new graph,
    # in the order specified by sort_key
    for (node, node_data) in sorted(graph.nodes(data=True), key=sort_key):
        new_graph.add_node(node, **node_data)

    # copy all edges from the original graph
    for (source, dest, data) in graph.edges(data=True):
        new_graph.add_edge(source, dest, **data)

    return new_graph


def subgraph(graph: MultiDiGraph, nodes: Iterable[Any]) -> MultiDiGraph:
    """
    Get a Subgraph view of a a Digraph with node iteration order in a deterministic fashion

    This code originally comes from
    https://github.com/networkx/networkx/blob/master/networkx/classes/graph.py#L1614
    We made our own version as the method used in the original implementation got
    induced_nodes as a set, which loses the determinism in iteration order over the set.
    To fix this issue, we use an immutableset to cast the nodes into. This was necessary
    as the lack of determinism was causing inconsistent failure/success states in our
    pattern matching code.
    """
    induced_nodes = ShowNodes(nodes=immutableset(graph.nbunch_iter(nodes)))
    # if already a subgraph, don't make a chain
    subgraph_builder = nx.graphviews.subgraph_view
    if hasattr(graph, "_NODE_OK"):
        return subgraph_builder(
            graph, induced_nodes, graph._EDGE_OK  # pylint:disable=protected-access
        )
    return subgraph_builder(graph, induced_nodes)


@attrs
class ShowNodes:
    """
    See `subgraph` for more information
    """

    nodes: ImmutableSet[Any] = attrib(converter=_to_immutableset)

    def __call__(self, node):
        return node in self.nodes


def copy_digraph(digraph: MultiDiGraph) -> MultiDiGraph:
    """
    We need this for use as an attrs converter.
    """
    return digraph.copy()


@attrs(frozen=True, slots=True)
class EdgeWrapper:
    """
    Used as a helper class to wrap the return of a complete edge marker for
    a multidigraph edge
    """

    source: Any = attrib()
    target: Any = attrib()
    key: Any = attrib()
    data: Any = attrib()


def get_edges(
    graph: MultiDiGraph,
    nodes: Optional[Iterable[Any]] = None,
    label: Optional[Any] = True,
) -> ImmutableSet[EdgeWrapper]:
    """
    Helper method to return an iterable of edges in a multidigraph

    See `EdgeWrapper`
    """
    return immutableset(
        EdgeWrapper(source=s, target=t, key=k, data=d)
        for (s, t, k, d) in graph.edges(nodes, keys=True, data=label)
    )
