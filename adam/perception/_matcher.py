# -*- coding: utf-8 -*-
# This code originated with NetworkX and we don't want to fix all their Pylint problems.
# pylint: skip-file
"""
This is derived from the VF2 graph isomorphism implementation in NetworkX.
That implementation is copyrighted by the NetworkX maintainers
and licensed under the 3-clause BSD license.
That implementation was originally coded by Christopher Ellison
as part of the Computational Mechanics Python (CMPy) project.
James P. Crutchfield, principal investigator.
Complexity Sciences Center and Physics Department, UC Davis.

We have made our own version because we want to track things such as
where matches most frequently fail in order to assist with hypothesis refinement.

This code should not be used by anything except the perception_graph module.
"""

import sys
from collections import defaultdict
from itertools import chain
from typing import Mapping, Any, Dict, Callable, Optional

from immutablecollections import immutableset, ImmutableSet
from more_itertools import flatten
from networkx import DiGraph


class GraphMatching:
    """Implementation of VF2 algorithm for matching undirected graphs.

    Suitable for Graph and MultiGraph instances.
    """

    def __init__(
        self, graph: DiGraph, pattern: DiGraph, use_lookahead_pruning: bool = True
    ) -> None:
        self.graph = graph
        self.pattern = pattern
        # we specify disable_order_check here because we know the DiGraph provides
        # the nodes in a deterministic order, but immutableset() can't tell.
        self.graph_nodes: ImmutableSet[Any] = immutableset(
            graph.nodes(), disable_order_check=True
        )
        self.pattern_nodes: ImmutableSet[Any] = immutableset(
            pattern.nodes(), disable_order_check=True
        )
        self.pattern_node_order = {n: i for i, n in enumerate(pattern)}

        # Set recursion limit.
        self.old_recursion_limit = sys.getrecursionlimit()
        expected_max_recursion_level = len(self.pattern)
        if self.old_recursion_limit < 1.5 * expected_max_recursion_level:
            # Give some breathing room.
            sys.setrecursionlimit(int(1.5 * expected_max_recursion_level))

        # Declare that we will be searching for a graph-graph isomorphism.
        self.test = "graph"

        # in debug mode, we keep track of the largest (by # of nodes) incomplete match
        # we can find
        self.debug_largest_match: Mapping[Any, Any] = {}
        self.failing_pattern_node_for_deepest_match = None

        self.use_lookahead_pruning = use_lookahead_pruning

        self._reset_debugging_maps()

        # Initialize state
        self.initialize()

    def reset_recursion_limit(self):
        """Restores the recursion limit."""
        # TODO:
        # Currently, we use recursion and set the recursion level higher.
        # It would be nice to restore the level, but because the
        # (Di)GraphMatcher classes make use of cyclic references, garbage
        # collection will never happen when we define __del__() to
        # restore the recursion level. The result is a memory leak.
        # So for now, we do not automatically restore the recursion level,
        # and instead provide a method to do this manually. Eventually,
        # we should turn this into a non-recursive implementation.
        sys.setrecursionlimit(self.old_recursion_limit)

    def next_match_candidates(self):
        """
        Get a tuple of the next pattern node to match and an Iterable of graph nodes
        to try to match it against.
        """

        # All computations are done using the current state!

        graph_nodes = self.graph_nodes
        pattern_nodes = self.pattern_nodes
        # by default, this respects the order of the nodes in the DiGraph,
        # which appears to be by insertion order (at least on recent Pythons)
        min_key = self.pattern_node_order.__getitem__

        # First we compute the "forward frontier" sets.
        # These are the nodes which are reachable in one hop from the currently matched nodes
        # by following edges in the forwards direction.
        graph_match_forward_frontier = [
            node
            for node in self.graph_nodes_in_or_succeeding_match
            if node not in self.graph_node_to_pattern_node
        ]
        pattern_match_forward_frontier = [
            node
            for node in self.pattern_nodes_in_or_succeeding_match
            if node not in self.pattern_node_to_graph_node
        ]

        # if there are candidate node alignments moving forwards along the edges we attempt
        # those first
        # RMG: why doesn't this fail when the pattern has a forward frontier but the graph does not?
        # Shouldn't that indicate a failed match?
        if graph_match_forward_frontier and pattern_match_forward_frontier:
            pattern_node = min(pattern_match_forward_frontier, key=min_key)
            return (pattern_node, graph_match_forward_frontier)
        else:
            # Compute the "backward frontier" sets.
            # These are the nodes which are reachable in one hop from the currently matched nodes
            # by following edges in the *backwards* direction direction.
            graph_match_backwards_frontier = [
                node
                for node in self.graph_nodes_in_or_preceding_match
                if node not in self.graph_node_to_pattern_node
            ]
            pattern_match_backwards_frontier = [
                node
                for node in self.pattern_nodes_in_or_preceding_match
                if node not in self.pattern_node_to_graph_node
            ]

            # if there are candidate node alignments moving backwards along the edges we attempt
            # those next.
            # RMG: why doesn't this fail when the pattern has a backward frontier
            # but the graph does not? Shouldn't that indicate a failed match?
            if graph_match_backwards_frontier and pattern_match_backwards_frontier:
                pattern_node = min(pattern_match_backwards_frontier, key=min_key)
                return (pattern_node, graph_match_backwards_frontier)
            else:
                # otherwise we just take the first unmatched pattern node
                pattern_node = min(
                    pattern_nodes - set(self.pattern_node_to_graph_node), key=min_key
                )
                return (
                    pattern_node,
                    [
                        graph_node
                        for graph_node in graph_nodes
                        if graph_node not in self.graph_node_to_pattern_node
                    ],
                )

        # For all other cases, we don't have any candidate pairs.

    def initialize(self):
        """Reinitializes the state of the algorithm.

        This method should be redefined if using something other than DiGMState.
        If only subclassing GraphMatcher, a redefinition is not necessary.
        """

        # the alignment of nodes between pattern and graph for the match so far
        self.graph_node_to_pattern_node = {}
        self.pattern_node_to_graph_node = {}

        # See the paper for definitions of M_x and T_x^{y}

        # the maps below track which nodes are on the "frontier" of the matched region
        # of the pattern and graph matches, etc. - that is, which nodes precede or succeed them.
        # We match nodes themselves are included in both sets for algorithmic convenience.
        # For efficiency during search, these are dicts mapping each node
        # to the depth of the search tree when the node was first encountered
        # as a neighbor to the match.
        self.graph_nodes_in_or_preceding_match = {}
        self.pattern_nodes_in_or_preceding_match = {}
        self.graph_nodes_in_or_succeeding_match = {}
        self.pattern_nodes_in_or_succeeding_match = {}

        self.state = GraphMatchingState(self)

        # Provide a convenient way to access the isomorphism mapping.
        self.mapping = self.graph_node_to_pattern_node.copy()

    def debug_diagnostics(self) -> Mapping[Any, Any]:
        # for nodes, we want to partition them into three groups
        pattern_nodes_which_were_compared_and_matched_at_least_once = immutableset(
            pattern_node
            for (
                pattern_node,
                attempts,
            ) in self.pattern_node_to_num_predicate_attempts.items()
            if self.pattern_node_to_num_predicate_failures[pattern_node] < attempts
        )
        pattern_nodes_which_were_compared_but_never_found_a_match = immutableset(
            pattern_node
            for (
                pattern_node,
                attempts,
            ) in self.pattern_node_to_num_predicate_attempts.items()
            if self.pattern_node_to_num_predicate_failures[pattern_node] == attempts
        )

        failed_edges = immutableset(
            pattern_edge
            for (pattern_edge, attempts) in self.pattern_edge_to_num_match_attempts
            if self.pattern_edge_to_num_predicate_failures[pattern_edge]
            + self.pattern_edge_to_num_presence_failures[pattern_edge]
            == attempts
        )

        syntax_node_failures = immutableset(
            pattern_node
            for (pattern_node, attempts) in self.node_to_syntax_attempts.items()
            if self.node_to_syntax_failures[pattern_node] == attempts
        )

        return {
            "nodes-matched-at-least-once": pattern_nodes_which_were_compared_and_matched_at_least_once,
            "nodes-never-matched": pattern_nodes_which_were_compared_but_never_found_a_match,
            "failed_edges": failed_edges,
            "syntax_failures": syntax_node_failures,
        }

    def is_isomorphic(self):
        """Returns True if G1 and G2 are isomorphic graphs."""

        # Let's do two very quick checks!
        # QUESTION: Should we call faster_graph_could_be_isomorphic(G1,G2)?
        # For now, I just copy the code.

        # Check global properties
        if self.graph.order() != self.pattern.order():
            return False

        # Check local properties
        graph_node_degree = sorted(d for n, d in self.graph.degree())
        pattern_node_degree = sorted(d for n, d in self.pattern.degree())
        if graph_node_degree != pattern_node_degree:
            return False

        try:
            next(self.isomorphisms_iter())
            return True
        except StopIteration:
            return False

    def isomorphisms_iter(self):
        """Generator over isomorphisms between G1 and G2."""
        # Declare that we are looking for a graph-graph isomorphism.
        self.test = "graph"
        self.initialize()
        for mapping in self.match():
            yield mapping

    def match(
        self,
        *,
        collect_debug_statistics: bool = False,
        debug_callback: Optional[Callable[[Any, Any], None]] = None,
        matching_pattern: bool = False
    ):
        """Extends the isomorphism mapping.

        This function is called recursively to determine if a complete
        isomorphism can be found between G1 and G2.  It cleans up the class
        variables after each recursive call. If an isomorphism is found,
        we yield the mapping.

        *matching_pattern* should be indicated as true if the two graphs
        which are being matched are both made up of `NodePredicate` objects.

        """
        at_largest_match_so_far = len(self.pattern_node_to_graph_node) >= len(
            self.debug_largest_match
        )
        if at_largest_match_so_far:
            self.debug_largest_match = self.pattern_node_to_graph_node.copy()
            # Check rendering debug flag to see if we should render the graph
            if debug_callback:
                debug_callback(self.graph, self.graph_node_to_pattern_node)
        if len(self.graph_node_to_pattern_node) == len(self.pattern):
            # Save the final mapping, otherwise garbage collection deletes it.
            self.mapping = self.graph_node_to_pattern_node.copy()
            # The mapping is complete.
            yield self.mapping
        else:
            (
                next_pattern_node_to_match,
                graph_nodes_to_match_against,
            ) = self.next_match_candidates()
            for graph_node in graph_nodes_to_match_against:
                if self.semantic_feasibility(
                    graph_node,
                    next_pattern_node_to_match,
                    collect_debug_statistics=collect_debug_statistics,
                    matching_pattern=matching_pattern,
                ):
                    if collect_debug_statistics:
                        self.node_to_syntax_attempts[next_pattern_node_to_match] += 1
                    if self.syntactic_feasibility(graph_node, next_pattern_node_to_match):
                        # Recursive call, adding the feasible state.
                        newstate = self.state.__class__(
                            self, graph_node, next_pattern_node_to_match
                        )
                        for mapping in self.match(
                            collect_debug_statistics=collect_debug_statistics,
                            debug_callback=debug_callback,
                        ):
                            yield mapping

                        # restore data structures
                        newstate.restore()
                    else:
                        if collect_debug_statistics:
                            self.node_to_syntax_failures[next_pattern_node_to_match] += 1

            if at_largest_match_so_far:
                # We failed to extend our largest match so far.
                # We record what pattern node we were trying to match when this extension failed.
                # This can be useful for debugging and for pattern pruning and refinement.
                self.failing_pattern_node_for_deepest_match = next_pattern_node_to_match

    def semantic_feasibility(
        self,
        graph_node,
        pattern_node,
        collect_debug_statistics=False,
        matching_pattern=False,
    ):
        """Returns True if adding (G1_node, G2_node) is symantically feasible.

        The semantic feasibility function should return True if it is
        acceptable to add the candidate pair (G1_node, G2_node) to the current
        partial isomorphism mapping.   The logic should focus on semantic
        information contained in the edge data or a formalized node class.

        By acceptable, we mean that the subsequent mapping can still become a
        complete isomorphism mapping.  Thus, if adding the candidate pair
        definitely makes it so that the subsequent mapping cannot become a
        complete isomorphism mapping, then this function must return False.

        The default semantic feasibility function always returns True. The
        effect is that semantics are not considered in the matching of G1
        and G2.

        The semantic checks might differ based on the what type of test is
        being performed.  A keyword description of the test is stored in
        self.test.  Here is a quick description of the currently implemented
        tests::

          test='graph'
            Indicates that the graph matcher is looking for a graph-graph
            isomorphism.

          test='subgraph'
            Indicates that the graph matcher is looking for a subgraph-graph
            isomorphism such that a subgraph of G1 is isomorphic to G2.

          test='mono'
            Indicates that the graph matcher is looking for a subgraph-graph
            monomorphism such that a subgraph of G1 is monomorphic to G2.

        Any subclass which redefines semantic_feasibility() must maintain
        the above form to keep the match() method functional. Implementations
        should consider multigraphs.
        """
        if collect_debug_statistics:
            self.pattern_node_to_num_predicate_attempts[pattern_node] += 1

        # We assume the nodes of G2 are node predicates which must hold true for the
        # corresponding G1 graph node for there to be a match.
        # IF we are matching two pattern nodes together we can't just use the
        # __call__ function on the Predicates, we need to call the .matches_predicate
        # instead. We use a boolean rather than checking at runtime to speed up this
        # process
        if matching_pattern:
            if not pattern_node.matches_predicate(graph_node):
                if collect_debug_statistics:
                    self.pattern_node_to_num_predicate_failures[pattern_node] += 1
                return False
        else:
            if not pattern_node(graph_node):
                if collect_debug_statistics:
                    self.pattern_node_to_num_predicate_failures[pattern_node] += 1
                return False

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
                # TODO: the current implementation does not handle multi-graphs
                pattern_edge = self.pattern.edges[pattern_predecessor, pattern_node]
                pattern_predicate = pattern_edge["predicate"]
                graph_edge = self.graph.get_edge_data(
                    predecessor_mapped_node_in_graph, graph_node
                )

                if collect_debug_statistics:
                    self.pattern_edge_to_num_match_attempts[
                        (pattern_predecessor, pattern_node)
                    ] += 1

                if not graph_edge:
                    if collect_debug_statistics:
                        self.pattern_edge_to_num_presence_failures[
                            (pattern_predecessor, pattern_node)
                        ] += 1
                    return False
                if not pattern_predicate(
                    predecessor_mapped_node_in_graph, graph_edge["label"], graph_node
                ):
                    if collect_debug_statistics:
                        self.pattern_edge_to_num_predicate_failures[
                            (pattern_predecessor, pattern_node)
                        ] += 1
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
                # TODO: the current implementation does not handle multi-graphs
                pattern_edge = self.pattern.edges[pattern_node, pattern_successor]
                pattern_predicate = pattern_edge["predicate"]
                graph_edge = self.graph.get_edge_data(
                    graph_node, successor_mapped_node_in_graph
                )

                if collect_debug_statistics:
                    self.pattern_edge_to_num_match_attempts[
                        (pattern_node, pattern_successor)
                    ] += 1

                if not graph_edge:
                    if collect_debug_statistics:
                        self.pattern_edge_to_num_presence_failures[
                            (pattern_node, pattern_successor)
                        ] += 1
                    return False

                if not pattern_predicate(
                    graph_node, graph_edge["label"], successor_mapped_node_in_graph
                ):
                    if collect_debug_statistics:
                        self.pattern_edge_to_num_predicate_failures[
                            (pattern_node, pattern_successor)
                        ] += 1
                    return False
        return True

    def subgraph_is_isomorphic(self):
        """Returns True if a subgraph of G1 is isomorphic to G2."""
        try:
            next(self.subgraph_isomorphisms_iter())
            return True
        except StopIteration:
            return False

    def subgraph_is_monomorphic(self):
        """Returns True if a subgraph of G1 is monomorphic to G2."""
        try:
            next(self.subgraph_monomorphisms_iter())
            return True
        except StopIteration:
            return False

    #    subgraph_is_isomorphic.__doc__ += "\n" + subgraph.replace('\n','\n'+indent)

    def subgraph_isomorphisms_iter(
        self,
        *,
        collect_debug_statistics: bool = False,
        debug_callback: Optional[Callable[[Any, Any], None]] = None,
        matching_pattern: bool = False
    ):
        """Generator over isomorphisms between a subgraph of G1 and G2."""
        # Declare that we are looking for graph-subgraph isomorphism.
        self.test = "subgraph"
        self.initialize()
        self.debug_largest_match = {}
        self.failing_pattern_node_for_deepest_match = None
        self._reset_debugging_maps()
        for mapping in self.match(
            collect_debug_statistics=collect_debug_statistics,
            debug_callback=debug_callback,
            matching_pattern=matching_pattern,
        ):
            yield mapping

    def subgraph_monomorphisms_iter(self):
        """Generator over monomorphisms between a subgraph of G1 and G2."""
        # Declare that we are looking for graph-subgraph monomorphism.
        self.test = "mono"
        self.initialize()
        for mapping in self.match():
            yield mapping

    #    subgraph_isomorphisms_iter.__doc__ += "\n" + subgraph.replace('\n','\n'+indent)

    def syntactic_feasibility(self, graph_node, pattern_node):
        """Returns True if adding (G1_node, G2_node) is syntactically feasible.

        This function returns True if it is adding the candidate pair
        to the current partial isomorphism/monomorphism mapping is allowable.
        The addition is allowable if the inclusion of the candidate pair does
        not make it impossible for an isomorphism/monomorphism to be found.
        """

        # The VF2 algorithm was designed to work with graphs having, at most,
        # one edge connecting any two nodes.  This is not the case when
        # dealing with an MultiGraphs.
        #
        # Basically, when we test the look-ahead rules R_pred and R_succ, we
        # will make sure that the number of edges are checked.  We also add
        # a R_self check to verify that the number of selfloops is acceptable.

        # Users might be comparing DiGraph instances with MultiDiGraph
        # instances. So the generic DiGraphMatcher class must work with
        # MultiDiGraphs. Care must be taken since the value in the innermost
        # dictionary is a singlet for DiGraph instances.  For MultiDiGraphs,
        # the value in the innermost dictionary is a list.

        ###
        # Test at each step to get a return value as soon as possible.
        ###

        # Look ahead 0

        # R_self

        # The number of selfloops for G1_node must equal the number of
        # self-loops for G2_node. Without this check, we would fail on R_pred
        # at the next recursion level. This should prune the tree even further.
        if self.test == "mono":
            if self.graph.number_of_edges(
                graph_node, graph_node
            ) < self.pattern.number_of_edges(pattern_node, pattern_node):
                return False
        else:
            if self.graph.number_of_edges(
                graph_node, graph_node
            ) != self.pattern.number_of_edges(pattern_node, pattern_node):
                return False

        # R_pred

        # For each predecessor n' of n in the partial mapping, the
        # corresponding node m' is a predecessor of m, and vice versa. Also,
        # the number of edges must be equal
        if self.test != "mono":
            for predecessor in self.graph.pred[graph_node]:
                if predecessor in self.graph_node_to_pattern_node:
                    if not (
                        self.graph_node_to_pattern_node[predecessor]
                        in self.pattern.pred[pattern_node]
                    ):
                        return False
                    elif self.graph.number_of_edges(
                        predecessor, graph_node
                    ) != self.pattern.number_of_edges(
                        self.graph_node_to_pattern_node[predecessor], pattern_node
                    ):
                        return False

        for predecessor in self.pattern.pred[pattern_node]:
            if predecessor in self.pattern_node_to_graph_node:
                if not (
                    self.pattern_node_to_graph_node[predecessor]
                    in self.graph.pred[graph_node]
                ):
                    return False
                elif self.test == "mono":
                    if self.graph.number_of_edges(
                        self.pattern_node_to_graph_node[predecessor], graph_node
                    ) < self.pattern.number_of_edges(predecessor, pattern_node):
                        return False
                else:
                    if self.graph.number_of_edges(
                        self.pattern_node_to_graph_node[predecessor], graph_node
                    ) != self.pattern.number_of_edges(predecessor, pattern_node):
                        return False

        # R_succ

        # For each successor n' of n in the partial mapping, the corresponding
        # node m' is a successor of m, and vice versa. Also, the number of
        # edges must be equal.
        if self.test != "mono":
            for successor in self.graph[graph_node]:
                if successor in self.graph_node_to_pattern_node:
                    if not (
                        self.graph_node_to_pattern_node[successor]
                        in self.pattern[pattern_node]
                    ):
                        return False
                    elif self.graph.number_of_edges(
                        graph_node, successor
                    ) != self.pattern.number_of_edges(
                        pattern_node, self.graph_node_to_pattern_node[successor]
                    ):
                        return False

        for successor in self.pattern[pattern_node]:
            if successor in self.pattern_node_to_graph_node:
                if not (
                    self.pattern_node_to_graph_node[successor] in self.graph[graph_node]
                ):
                    return False
                elif self.test == "mono":
                    if self.graph.number_of_edges(
                        graph_node, self.pattern_node_to_graph_node[successor]
                    ) < self.pattern.number_of_edges(pattern_node, successor):
                        return False
                else:
                    if self.graph.number_of_edges(
                        graph_node, self.pattern_node_to_graph_node[successor]
                    ) != self.pattern.number_of_edges(pattern_node, successor):
                        return False

        if self.use_lookahead_pruning:
            if self.test != "mono":

                # Look ahead 1

                # R_termin
                # The number of predecessors of n that are in T_1^{in} is equal to the
                # number of predecessors of m that are in T_2^{in}.
                num1 = 0
                for predecessor in self.graph.pred[graph_node]:
                    if (predecessor in self.graph_nodes_in_or_preceding_match) and (
                        predecessor not in self.graph_node_to_pattern_node
                    ):
                        num1 += 1
                num2 = 0
                for predecessor in self.pattern.pred[pattern_node]:
                    if (predecessor in self.pattern_nodes_in_or_preceding_match) and (
                        predecessor not in self.pattern_node_to_graph_node
                    ):
                        num2 += 1
                if self.test == "graph":
                    if not (num1 == num2):
                        return False
                else:  # self.test == 'subgraph'
                    if not (num1 >= num2):
                        return False

                # The number of successors of n that are in T_1^{in} is equal to the
                # number of successors of m that are in T_2^{in}.
                num1 = 0
                for successor in self.graph[graph_node]:
                    if (successor in self.graph_nodes_in_or_preceding_match) and (
                        successor not in self.graph_node_to_pattern_node
                    ):
                        num1 += 1
                num2 = 0
                for successor in self.pattern[pattern_node]:
                    if (successor in self.pattern_nodes_in_or_preceding_match) and (
                        successor not in self.pattern_node_to_graph_node
                    ):
                        num2 += 1
                if self.test == "graph":
                    if not (num1 == num2):
                        return False
                else:  # self.test == 'subgraph'
                    if not (num1 >= num2):
                        return False

                # R_termout

                # The number of predecessors of n that are in T_1^{out} is equal to the
                # number of predecessors of m that are in T_2^{out}.
                num1 = 0
                for predecessor in self.graph.pred[graph_node]:
                    if (predecessor in self.graph_nodes_in_or_succeeding_match) and (
                        predecessor not in self.graph_node_to_pattern_node
                    ):
                        num1 += 1
                num2 = 0
                for predecessor in self.pattern.pred[pattern_node]:
                    if (predecessor in self.pattern_nodes_in_or_succeeding_match) and (
                        predecessor not in self.pattern_node_to_graph_node
                    ):
                        num2 += 1
                if self.test == "graph":
                    if not (num1 == num2):
                        return False
                else:  # self.test == 'subgraph'
                    if not (num1 >= num2):
                        return False

                # The number of successors of n that are in T_1^{out} is equal to the
                # number of successors of m that are in T_2^{out}.
                num1 = 0
                for successor in self.graph[graph_node]:
                    if (successor in self.graph_nodes_in_or_succeeding_match) and (
                        successor not in self.graph_node_to_pattern_node
                    ):
                        num1 += 1
                num2 = 0
                for successor in self.pattern[pattern_node]:
                    if (successor in self.pattern_nodes_in_or_succeeding_match) and (
                        successor not in self.pattern_node_to_graph_node
                    ):
                        num2 += 1
                if self.test == "graph":
                    if not (num1 == num2):
                        return False
                else:  # self.test == 'subgraph'
                    if not (num1 >= num2):
                        return False

                # Look ahead 2

                # R_new

                # The number of predecessors of n that are neither in the core_1 nor
                # T_1^{in} nor T_1^{out} is equal to the number of predecessors of m
                # that are neither in core_2 nor T_2^{in} nor T_2^{out}.
                num1 = 0
                for predecessor in self.graph.pred[graph_node]:
                    if (predecessor not in self.graph_nodes_in_or_preceding_match) and (
                        predecessor not in self.graph_nodes_in_or_succeeding_match
                    ):
                        num1 += 1
                num2 = 0
                for predecessor in self.pattern.pred[pattern_node]:
                    if (predecessor not in self.pattern_nodes_in_or_preceding_match) and (
                        predecessor not in self.pattern_nodes_in_or_succeeding_match
                    ):
                        num2 += 1
                if self.test == "graph":
                    if not (num1 == num2):
                        return False
                else:  # self.test == 'subgraph'
                    if not (num1 >= num2):
                        return False

                # The number of successors of n that are neither in the core_1 nor
                # T_1^{in} nor T_1^{out} is equal to the number of successors of m
                # that are neither in core_2 nor T_2^{in} nor T_2^{out}.
                num1 = 0
                for successor in self.graph[graph_node]:
                    if (successor not in self.graph_nodes_in_or_preceding_match) and (
                        successor not in self.graph_nodes_in_or_succeeding_match
                    ):
                        num1 += 1
                num2 = 0
                for successor in self.pattern[pattern_node]:
                    if (successor not in self.pattern_nodes_in_or_preceding_match) and (
                        successor not in self.pattern_nodes_in_or_succeeding_match
                    ):
                        num2 += 1
                if self.test == "graph":
                    if not (num1 == num2):
                        return False
                else:  # self.test == 'subgraph'
                    if not (num1 >= num2):
                        return False

        # Otherwise, this node pair is syntactically feasible!
        return True

    def _reset_debugging_maps(self) -> None:
        # how often does each node fail to match due to predicates?
        self.pattern_node_to_num_predicate_failures: Dict[Any, int] = defaultdict(int)
        self.pattern_node_to_num_predicate_attempts: Dict[Any, int] = defaultdict(int)

        # how often does each edge fail to match due to the edge predicate returning false?
        self.pattern_edge_to_num_predicate_failures: Dict[Any, int] = defaultdict(int)
        # and how often because no corresponding edge exists to apply the predicate to?
        self.pattern_edge_to_num_presence_failures: Dict[Any, int] = defaultdict(int)
        self.pattern_edge_to_num_match_attempts: Dict[Any, int] = defaultdict(int)

        self.node_to_syntax_failures: Dict[Any, int] = defaultdict(int)
        self.node_to_syntax_attempts: Dict[Any, int] = defaultdict(int)


class GraphMatchingState(object):
    """Internal representation of state for the DiGraphMatcher class.

    This class is used internally by the DiGraphMatcher class.  It is used
    only to store state specific data. There will be at most G2.order() of
    these objects in memory at a time, due to the depth-first search
    strategy employed by the VF2 algorithm.

    """

    def __init__(self, GM: GraphMatching, graph_node=None, pattern_node=None) -> None:
        """Initializes DiGMState object.

        Pass in the DiGraphMatcher to which this DiGMState belongs and the
        new node pair that will be added to the GraphMatcher's current
        isomorphism mapping.
        """
        self.GM = GM

        # Initialize the last stored node pair.
        self.graph_node = None
        self.pattern_node = None
        self.depth = len(GM.graph_node_to_pattern_node)

        if graph_node is None or pattern_node is None:
            # Then we reset the class variables
            GM.graph_node_to_pattern_node = {}
            GM.pattern_node_to_graph_node = {}
            GM.graph_nodes_in_or_preceding_match = {}
            GM.pattern_nodes_in_or_preceding_match = {}
            GM.graph_nodes_in_or_succeeding_match = {}
            GM.pattern_nodes_in_or_succeeding_match = {}

        # Watch out! G1_node == 0 should evaluate to True.
        if graph_node is not None and pattern_node is not None:
            # Add the node pair to the isomorphism mapping.
            GM.graph_node_to_pattern_node[graph_node] = pattern_node
            GM.pattern_node_to_graph_node[pattern_node] = graph_node

            # Store the node that was added last.
            self.graph_node = graph_node
            self.pattern_node = pattern_node

            # Now we must update the other four vectors.
            # We will add only if it is not in there already!
            self.depth = len(GM.graph_node_to_pattern_node)

            # First we add the new nodes...
            for vector in (
                GM.graph_nodes_in_or_preceding_match,
                GM.graph_nodes_in_or_succeeding_match,
            ):
                if graph_node not in vector:
                    vector[graph_node] = self.depth
            for vector in (
                GM.pattern_nodes_in_or_preceding_match,
                GM.pattern_nodes_in_or_succeeding_match,
            ):
                if pattern_node not in vector:
                    vector[pattern_node] = self.depth

            # Now we add every other node...

            # Updates for T_1^{in}
            # we use immutableset to guarantee deterministic iteration
            new_nodes_0: ImmutableSet[Any] = immutableset(
                flatten(
                    [
                        predecessor
                        for predecessor in GM.graph.predecessors(node)
                        if predecessor not in GM.graph_node_to_pattern_node
                    ]
                    for node in GM.graph_node_to_pattern_node
                )
            )
            for node in new_nodes_0:
                if node not in GM.graph_nodes_in_or_preceding_match:
                    GM.graph_nodes_in_or_preceding_match[node] = self.depth

            # Updates for T_2^{in}
            new_nodes_1: ImmutableSet[Any] = immutableset(
                flatten(
                    [
                        predecessor
                        for predecessor in GM.pattern.predecessors(node)
                        if predecessor not in GM.pattern_node_to_graph_node
                    ]
                    for node in GM.pattern_node_to_graph_node
                )
            )
            for node in new_nodes_1:
                if node not in GM.pattern_nodes_in_or_preceding_match:
                    GM.pattern_nodes_in_or_preceding_match[node] = self.depth

            # Updates for T_1^{out}
            new_nodes_2: ImmutableSet[Any] = immutableset(
                flatten(
                    [
                        successor
                        for successor in GM.graph.successors(node)
                        if successor not in GM.graph_node_to_pattern_node
                    ]
                    for node in GM.graph_node_to_pattern_node
                )
            )

            for node in new_nodes_2:
                if node not in GM.graph_nodes_in_or_succeeding_match:
                    GM.graph_nodes_in_or_succeeding_match[node] = self.depth

            # Updates for T_2^{out}
            new_nodes_3: ImmutableSet[Any] = immutableset(
                flatten(
                    [
                        successor
                        for successor in GM.pattern.successors(node)
                        if successor not in GM.pattern_node_to_graph_node
                    ]
                    for node in GM.pattern_node_to_graph_node
                )
            )

            for node in new_nodes_3:
                if node not in GM.pattern_nodes_in_or_succeeding_match:
                    GM.pattern_nodes_in_or_succeeding_match[node] = self.depth

    def restore(self):
        """Deletes the DiGMState object and restores the class variables."""

        # First we remove the node that was added from the core vectors.
        # Watch out! G1_node == 0 should evaluate to True.
        if self.graph_node is not None and self.pattern_node is not None:
            del self.GM.graph_node_to_pattern_node[self.graph_node]
            del self.GM.pattern_node_to_graph_node[self.pattern_node]

        # Now we revert the other four vectors.
        # Thus, we delete all entries which have this depth level.
        for vector in (
            self.GM.graph_nodes_in_or_preceding_match,
            self.GM.pattern_nodes_in_or_preceding_match,
            self.GM.graph_nodes_in_or_succeeding_match,
            self.GM.pattern_nodes_in_or_succeeding_match,
        ):
            for node in list(vector.keys()):
                if vector[node] == self.depth:
                    del vector[node]
