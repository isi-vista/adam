from more_itertools import first
from networkx import MultiDiGraph

from adam.perception._edge_isomorphism import _EdgeInducedIsomorphismMatching


def simple_match(a, b):
    return a == b


def test_single_node():
    pattern = MultiDiGraph()
    graph = MultiDiGraph()

    pattern.add_node("a")
    graph.add_node("a")

    assert (
        first(
            _EdgeInducedIsomorphismMatching(
                graph=graph,
                pattern=pattern,
                node_semantic_matcher=simple_match,
                edge_semantic_matcher=simple_match,
                graph_edge_label_key="doesn'tmatter",
                pattern_edge_label_key="doesn'tmatter",
            ).matches()
        )
        != None
    )
