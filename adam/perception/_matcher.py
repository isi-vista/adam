# -*- coding: utf-8 -*-
# This code originated with NetworkX and we don't want to fix all their Pylint problems.
# pylint: skip-file
"""
This is derived from the VF2 graph isomorphism implementation in NetworkX.
That implementation is copyrighted by the NetworkX maintainers
and licensed under the 3-clause BSD license.
That implenentation was originally coded by Christopher Ellison
as part of the Computational Mechanics Python (CMPy) project.
James P. Crutchfield, principal investigator.
Complexity Sciences Center and Physics Department, UC Davis.

We have made our own version because we want to track things such as
where matches most frequently fail in order to assist with hypothesis refinement.
"""

import sys
from collections import defaultdict
from typing import Mapping, Any, Dict

from immutablecollections import immutableset
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
        self.graph_nodes = set(graph.nodes())
        self.pattern_nodes = set(pattern.nodes())
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

    def candidate_pairs_iter(self):
        """Iterator over candidate pairs of nodes in G1 and G2."""

        # All computations are done using the current state!

        graph_nodes = self.graph_nodes
        pattern_nodes = self.pattern_nodes
        min_key = self.pattern_node_order.__getitem__

        # First we compute the out-terminal sets.
        T1_out = [
            node
            for node in self.graph_nodes_in_or_succeeding_match
            if node not in self.graph_node_to_pattern_node
        ]
        T2_out = [
            node
            for node in self.pattern_nodes_in_or_succeeding_match
            if node not in self.pattern_node_to_graph_node
        ]

        # If T1_out and T2_out are both nonempty.
        # P(s) = T1_out x {min T2_out}
        if T1_out and T2_out:
            node_2 = min(T2_out, key=min_key)
            for node_1 in T1_out:
                yield node_1, node_2

        # If T1_out and T2_out were both empty....
        # We compute the in-terminal sets.

        # elif not (T1_out or T2_out):   # as suggested by [2], incorrect
        else:  # as suggested by [1], correct
            T1_in = [
                node
                for node in self.graph_nodes_in_or_preceding_match
                if node not in self.graph_node_to_pattern_node
            ]
            T2_in = [
                node
                for node in self.pattern_nodes_in_or_preceding_match
                if node not in self.pattern_node_to_graph_node
            ]

            # If T1_in and T2_in are both nonempty.
            # P(s) = T1_out x {min T2_out}
            if T1_in and T2_in:
                node_2 = min(T2_in, key=min_key)
                for node_1 in T1_in:
                    yield node_1, node_2

            # If all terminal sets are empty...
            # P(s) = (N_1 - M_1) x {min (N_2 - M_2)}

            # elif not (T1_in or T2_in):   # as suggested by  [2], incorrect
            else:  # as inferred from [1], correct
                node_2 = min(
                    pattern_nodes - set(self.pattern_node_to_graph_node), key=min_key
                )
                for node_1 in graph_nodes:
                    if node_1 not in self.graph_node_to_pattern_node:
                        yield node_1, node_2

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

    def match(self, *, debug: bool = False):
        """Extends the isomorphism mapping.

        This function is called recursively to determine if a complete
        isomorphism can be found between G1 and G2.  It cleans up the class
        variables after each recursive call. If an isomorphism is found,
        we yield the mapping.

        """
        if debug and len(self.pattern_node_to_graph_node) >= len(
            self.debug_largest_match
        ):
            self.debug_largest_match = self.pattern_node_to_graph_node.copy()

        if len(self.graph_node_to_pattern_node) == len(self.pattern):
            # Save the final mapping, otherwise garbage collection deletes it.
            self.mapping = self.graph_node_to_pattern_node.copy()
            # The mapping is complete.
            yield self.mapping
        else:
            for graph_node, pattern_node in self.candidate_pairs_iter():
                if self.semantic_feasibility(graph_node, pattern_node, debug=debug):
                    if debug:
                        self.node_to_syntax_attempts[pattern_node] += 1
                    if self.syntactic_feasibility(graph_node, pattern_node):
                        # Recursive call, adding the feasible state.
                        newstate = self.state.__class__(self, graph_node, pattern_node)
                        for mapping in self.match(debug=debug):
                            yield mapping

                        # restore data structures
                        newstate.restore()
                    else:
                        if debug:
                            self.node_to_syntax_failures[pattern_node] += 1

    def semantic_feasibility(self, graph_node, pattern_node, debug=False):
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

        if debug:
            self.pattern_node_to_num_predicate_attempts[pattern_node] += 1

        # We assume the nodes of G2 are node predicates which must hold true for the
        # corresponding G1 graph node for there to be a match.
        if not pattern_node(graph_node):
            if debug:
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

                if debug:
                    self.pattern_edge_to_num_match_attempts[
                        (pattern_predecessor, pattern_node)
                    ] += 1

                if not graph_edge:
                    if debug:
                        self.pattern_edge_to_num_presence_failures[
                            (pattern_predecessor, pattern_node)
                        ] += 1
                    return False
                if not pattern_predicate(
                    predecessor_mapped_node_in_graph, graph_edge["label"], graph_node
                ):
                    if debug:
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

                if debug:
                    self.pattern_edge_to_num_match_attempts[
                        (pattern_node, pattern_successor)
                    ] += 1

                if not graph_edge:
                    if debug:
                        self.pattern_edge_to_num_presence_failures[
                            (pattern_node, pattern_successor)
                        ] += 1
                    return False

                if not pattern_predicate(
                    graph_node, graph_edge["label"], successor_mapped_node_in_graph
                ):
                    if debug:
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

    def subgraph_isomorphisms_iter(self, *, debug: bool = False):
        """Generator over isomorphisms between a subgraph of G1 and G2."""
        # Declare that we are looking for graph-subgraph isomorphism.
        self.test = "subgraph"
        self.initialize()
        self.debug_largest_match = {}
        self._reset_debugging_maps()
        for mapping in self.match(debug=debug):
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
            new_nodes = set([])
            for node in GM.graph_node_to_pattern_node:
                new_nodes.update(
                    [
                        predecessor
                        for predecessor in GM.graph.predecessors(node)
                        if predecessor not in GM.graph_node_to_pattern_node
                    ]
                )
            for node in new_nodes:
                if node not in GM.graph_nodes_in_or_preceding_match:
                    GM.graph_nodes_in_or_preceding_match[node] = self.depth

            # Updates for T_2^{in}
            new_nodes = set([])
            for node in GM.pattern_node_to_graph_node:
                new_nodes.update(
                    [
                        predecessor
                        for predecessor in GM.pattern.predecessors(node)
                        if predecessor not in GM.pattern_node_to_graph_node
                    ]
                )
            for node in new_nodes:
                if node not in GM.pattern_nodes_in_or_preceding_match:
                    GM.pattern_nodes_in_or_preceding_match[node] = self.depth

            # Updates for T_1^{out}
            new_nodes = set([])
            for node in GM.graph_node_to_pattern_node:
                new_nodes.update(
                    [
                        successor
                        for successor in GM.graph.successors(node)
                        if successor not in GM.graph_node_to_pattern_node
                    ]
                )
            for node in new_nodes:
                if node not in GM.graph_nodes_in_or_succeeding_match:
                    GM.graph_nodes_in_or_succeeding_match[node] = self.depth

            # Updates for T_2^{out}
            new_nodes = set([])
            for node in GM.pattern_node_to_graph_node:
                new_nodes.update(
                    [
                        successor
                        for successor in GM.pattern.successors(node)
                        if successor not in GM.pattern_node_to_graph_node
                    ]
                )
            for node in new_nodes:
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
