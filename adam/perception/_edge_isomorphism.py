from collections import deque
from typing import Callable, Deque, Dict, Generic, Iterable, Optional, Tuple, TypeVar

from attr.validators import instance_of
from networkx import MultiDiGraph, weakly_connected_components

from attr import Factory, attrib, attrs


class EdgeInducedIsoMorphismMatcher:
    pass


_GraphNode = TypeVar("_GraphNode")
_PatternNode = TypeVar("_PatternNode")
_GraphEdge = TypeVar("_GraphEdge")
_PatternEdge = TypeVar("_PatternEdge")

DEFAULT_GRAPH_KEY = "graph"
DEFAULT_PREDICATE_KEY = "predicate"


def default_pattern_vs_pattern_node_matcher(graph_node, pattern_node):
    return pattern_node.matches_predicate(graph_node)


def default_pattern_vs_graph_node_matcher(graph_node, pattern_node):
    return pattern_node(graph_node)


def default_pattern_vs_patten_edge_matcher(graph_edge, pattern_edge):
    return pattern_edge.matches_predicate(graph_edge)


def default_pattern_vs_graph_edge_matcher(graph_edge, pattern_edge):
    return pattern_edge(graph_edge)


@attrs
class _EdgeInducedIsomorphismMatching(
    Generic[_GraphNode, _PatternNode, _GraphEdge, _PatternEdge]
):
    graph: MultiDiGraph = attrib(validator=instance_of(MultiDiGraph), kw_only=True)
    pattern: MultiDiGraph = attrib(validator=instance_of(MultiDiGraph), kw_only=True)

    node_semantic_matcher: Callable[[_GraphNode, _PatternNode], bool] = attrib(
        kw_only=True
    )
    edge_semantic_matcher: Callable[[_GraphEdge, _PatternEdge], bool] = attrib(
        kw_only=True
    )
    graph_edge_label_key: str = attrib(kw_only=True, validator=instance_of(str))
    pattern_edge_label_key: str = attrib(kw_only=True, validator=instance_of(str))

    pattern_node_to_graph_node: Dict[_PatternNode, _GraphNode] = attrib(
        init=False, default=Factory(dict)
    )
    graph_node_to_pattern_node: Dict[_GraphNode, _PatternNode] = attrib(
        init=False, default=Factory(dict)
    )

    mapping_stack_pattern_graph: Deque[Tuple[_PatternNode, _GraphNode]] = attrib(
        init=False, default=Factory(deque)
    )
    largest_pattern_node_to_graph_node: Dict[_PatternNode, _GraphNode] = attrib(
        init=False, default=Factory(dict)
    )
    failing_node_for_deepest_pattern_match: Optional[_PatternNode] = attrib(
        init=False, default=None
    )

    def matches(self) -> Iterable[Dict[_PatternNode, _GraphNode]]:
        """
        Iterates over all legal alignments of graph nodes to pattern nodes.
        """
        # We track the largest match we find during the alignment search process
        # both for debugging and also for use by heuristic graph intersection algorithms.
        at_largest_match_so_far = len(self.pattern_node_to_graph_node) > len(
            self.largest_pattern_node_to_graph_node
        )
        if at_largest_match_so_far:
            self.largest_pattern_node_to_graph_node = (
                self.pattern_node_to_graph_node.copy()
            )

        if len(self.pattern_node_to_graph_node) == len(self.pattern):
            # We have a complete alignment of the pattern nodes...
            if self._complete_match_is_legal():
                # so if it is a legal alignment, we're done!
                yield self.graph_node_to_pattern_node
            else:
                # if not, give up on this search and backtrack.
                return

        (
            pattern_node_to_match,
            graph_nodes_to_match_against,
        ) = self._next_match_candidates()
        if at_largest_match_so_far:
            # If our attempt to match this pattern node fails below,
            # we know this pattern node is responsible for
            # our failure to extend our largest match so far.
            # We record what pattern node we were trying to match when this extension failed.
            # This can be useful for debugging and for pattern pruning and refinement.
            self.failing_node_for_deepest_pattern_match = pattern_node_to_match
        for graph_node_to_match_against in graph_nodes_to_match_against:
            if self._semantic_feasibility(
                pattern_node_to_match, graph_node_to_match_against
            ):
                if self._syntactic_feasibility(
                    pattern_node_to_match, graph_node_to_match_against
                ):
                    # Commit to this mapping and seek to extend it.
                    self.mapping_stack_pattern_graph.append(
                        (pattern_node_to_match, graph_node_to_match_against)
                    )
                    self.pattern_node_to_graph_node[
                        pattern_node_to_match
                    ] = graph_node_to_match_against
                    self.graph_node_to_pattern_node[
                        graph_node_to_match_against
                    ] = pattern_node_to_match

                    for mapping in self.matches():
                        yield mapping

                    # we've finished exploring possible extensions of
                    # aligning this pattern node to this graph node
                    # (in the context of our previous alignment commitments).
                    # It's time to backtrack.
                    self.mapping_stack_pattern_graph.pop()
                    del self.pattern_node_to_graph_node[pattern_node_to_match]
                    del self.graph_node_to_pattern_node[graph_node_to_match_against]

    def _semantic_feasibility(
        self, pattern_node: _PatternNode, graph_node: _GraphNode
    ) -> bool:
        # First check the aligned nodes are acceptable.
        if not self.node_semantic_matcher(graph_node, pattern_node):
            return False

        pattern_edge_key = self.pattern_edge_label_key
        graph_edge_key = self.graph_edge_label_key

        # Now comes the trickier bit of testing edge predicates.
        # First observe that we only need to test edge predicates against edges
        # where both endpoints are mapped in the current mapping.
        # If there is an edge with an unmapped endpoint,
        # it will get tested when that endpoint node is checked for semantic feasibility.
        for pattern_predecessor in self.pattern.pred[pattern_node]:
            predecessor_mapped_node_in_graph = self.pattern_node_to_graph_node.get(
                pattern_predecessor
            )
            if predecessor_mapped_node_in_graph:
                # We have an edge pattern_predecessor ---> G2_node
                # which is mapped to the edge predecessor_mapped_node_in_graph ---> G1_node.
                # Is this a legal mapping?
                # Note that can be multiple relations(=edges) between nodes, so we need to ensure
                # that each relation in the pattern has *at least one* matching relation
                # in the graph
                for (_, _, pattern_predicate) in self.pattern.edges(
                    [pattern_predecessor, pattern_node], data=pattern_edge_key
                ):
                    # Every pattern edge must align to some graph edge.
                    # Currently, it is okay if multiple pattern edges align to the same one.
                    has_matching_graph_edge = False
                    for (_, _, graph_edge_label) in self.graph.edges(
                        [predecessor_mapped_node_in_graph, graph_node],
                        data=graph_edge_key,
                    ):
                        if self.edge_semantic_matcher(
                            graph_edge_label, pattern_predicate
                        ):
                            has_matching_graph_edge = True
                            break
                    if not has_matching_graph_edge:
                        return False

        for pattern_successor in self.pattern[pattern_node]:
            successor_mapped_node_in_graph = self.pattern_node_to_graph_node.get(
                pattern_successor
            )
            if successor_mapped_node_in_graph:
                # We have an edge G2_node --> pattern_successor
                # which is mapped to the edge G1_node --> successor_mapped_node_in_graph.
                # Is this a legal mapping?
                # Note that can be multiple relations(=edges) between nodes, so we need to ensure
                # that each relation in the pattern has *at least one* matching relation
                # in the graph
                for (_, _, pattern_predicate) in self.pattern.edges(
                    [pattern_node, pattern_successor], data=pattern_edge_key
                ):
                    # Every pattern edge must align to some graph edge.
                    # Currently, it is okay if multiple pattern edges align to the same one.
                    has_matching_graph_edge = False
                    for (_, _, graph_edge_label) in self.graph.edges(
                        [graph_node, successor_mapped_node_in_graph], data=graph_edge_key
                    ):
                        if self.edge_semantic_matcher(
                            graph_edge_label, pattern_predicate
                        ):
                            has_matching_graph_edge = True
                            break
                    if not has_matching_graph_edge:
                        return False

        return True

    def _syntactic_feasibility(
        self, pattern_node: _PatternNode, graph_node: _GraphNode
    ) -> bool:
        # Note we ban self-loops in attrs_post_init

        # Below, we ensure that if a node and its neighbor are in our candidate partial alignment
        # on either the graph or pattern sides,
        # the aligned nodes on the other sides are also neighbors.
        for graph_predecessor in self.graph.pred[graph_node]:
            pattern_predecessor = self.graph_node_to_pattern_node.get(graph_predecessor)
            if pattern_predecessor:
                if self.graph.number_of_edges(
                    graph_predecessor, graph_node
                ) != self.pattern.number_of_edges(pattern_predecessor, pattern_node):
                    return False

        for pattern_predecessor in self.pattern.pred[pattern_node]:
            graph_predecessor = self.pattern_node_to_graph_node.get(pattern_predecessor)
            if graph_predecessor:
                if self.pattern.number_of_edges(
                    pattern_predecessor, pattern_node
                ) != self.graph.number_of_edges(graph_predecessor, graph_node):
                    return False

        for graph_successor in self.graph.succ[graph_node]:
            pattern_successor = self.graph_node_to_pattern_node.get(graph_successor)
            if pattern_successor:
                if self.graph.number_of_edges(
                    graph_node, graph_successor
                ) != self.pattern.number_of_edges(pattern_node, pattern_successor):
                    return False

        for pattern_successor in self.pattern.succ[pattern_node]:
            graph_successor = self.pattern_node_to_graph_node.get(pattern_successor)
            if graph_successor:
                if self.pattern.number_of_edges(
                    pattern_node, pattern_successor
                ) != self.graph.number_of_edges(graph_node, graph_successor):
                    return False

        return True

    def _complete_match_is_legal(self) -> bool:
        return True

    def _next_match_candidates(self) -> bool:
        raise NotImplementedError()

    def __attrs_post_init__(self) -> None:
        for node in self.graph:
            if self.graph.number_of_edges(node, node) > 1:
                raise RuntimeError(
                    f"Cannot match against graphs with self-loops, " f"but {node} has one"
                )
        for node in self.pattern:
            if self.graph.number_of_edges(node, node) > 1:
                raise RuntimeError(
                    f"Cannot match a pattern with self-loops, " f"but {node} has one"
                )

        num_graph_components = len(tuple(weakly_connected_components(self.graph)))
        if num_graph_components > 1:
            raise RuntimeError(
                f"Currently we only allow the graph to be matched against to have "
                f"one connected component, but got {num_graph_components}"
            )
        num_pattern_components = len(tuple(weakly_connected_components(self.graph)))
        if num_pattern_components > 1:
            raise RuntimeError(
                f"Currently we only allow the pattern being to have "
                f"one connected component, but got {num_pattern_components}"
            )
