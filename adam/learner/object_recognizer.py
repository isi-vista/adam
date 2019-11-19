from typing import Iterable, List, Tuple

from immutablecollections import ImmutableDict, immutabledict, immutableset
from more_itertools import first
from networkx import DiGraph

from adam.ontology import OntologyNode
from adam.ontology.phase1_ontology import (
    GAILA_PHASE_1_ONTOLOGY,
    PHASE_1_CURRICULUM_OBJECTS,
    TRUCK,
)
from adam.perception.perception_graph import (
    MatchedObjectNode,
    PerceptionGraph,
    PerceptionGraphPattern,
    PerceptionGraphPatternMatch,
)
from attr import attrs

_LIST_OF_PERCEIVED_PATTERNS = immutableset(
    (
        node.handle,
        PerceptionGraphPattern.from_schema(
            first(GAILA_PHASE_1_ONTOLOGY.structural_schemata(node))
        ),
    )
    for node in PHASE_1_CURRICULUM_OBJECTS
    if node
    in GAILA_PHASE_1_ONTOLOGY._structural_schemata.keys()  # pylint:disable=protected-access
    and node not in [TRUCK]
    # Currently can't be matched is what the list at the end is for
)

MATCHED_OBJECT_PATTERN_LABEL = OntologyNode("has-matched-object-pattern")


@attrs(frozen=True)
class ObjectRecognizer:
    """
    The ObjectRecognizer finds object matches in the scene pattern and adds a `MatchedObjectPerceptionPredicate`
    which can be used to learn additional semantics which relate objects to other objects
    """

    def match_objects(
        self,
        perception_graph: PerceptionGraph,
        possible_perceived_objects: Iterable[
            Tuple[str, PerceptionGraphPattern]
        ] = _LIST_OF_PERCEIVED_PATTERNS,
    ) -> Tuple[PerceptionGraph, ImmutableDict[str, MatchedObjectNode]]:
        """
        Match object patterns to objects in the scenes, then add a node for the matched object and copy relationships
        to it. These new patterns can be used to determine static prepositional relationships.
        """
        matched_object_nodes: List[Tuple[str, MatchedObjectNode]] = []
        graph_to_modify = perception_graph.copy_as_digraph()
        for (description, pattern) in possible_perceived_objects:
            matcher = pattern.matcher(perception_graph)
            pattern_matches = list(matcher.matches(use_lookahead_pruning=False))
            for pattern_match in pattern_matches:
                self._replace_match_with_object_graph_node(
                    graph_to_modify, pattern_match, matched_object_nodes, description
                )
        return PerceptionGraph(graph=graph_to_modify), immutabledict(matched_object_nodes)

    def _replace_match_with_object_graph_node(
        self,
        networkx_graph_to_modify_in_place: DiGraph,
        pattern_match: PerceptionGraphPatternMatch,
        matched_object_nodes: List[Tuple[str, MatchedObjectNode]],
        description: str,
    ):
        """
        Internal function to copy existing relationships from the matched object pattern onto a
        `MatchedObjectPerceptionPredicate`
        """
        node = MatchedObjectNode(name=(description,))

        matched_object_nodes.append((description, node))
        networkx_graph_to_modify_in_place.add_node(node)

        for (
            graph_node
        ) in (
            pattern_match.matched_sub_graph._graph.nodes  # pylint:disable=protected-access
        ):
            for neighbor_node in networkx_graph_to_modify_in_place.neighbors(graph_node):
                if networkx_graph_to_modify_in_place.has_edge(graph_node, neighbor_node):
                    edge_data = networkx_graph_to_modify_in_place.get_edge_data(
                        graph_node, neighbor_node
                    )
                    networkx_graph_to_modify_in_place.add_edge(
                        node, neighbor_node, **edge_data
                    )
            networkx_graph_to_modify_in_place.add_edge(
                graph_node, node, label=MATCHED_OBJECT_PATTERN_LABEL
            )
