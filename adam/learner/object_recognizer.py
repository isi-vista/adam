from typing import Iterable, List

from attr import attrs
from immutablecollections import immutableset, ImmutableSet
from networkx import DiGraph

from adam.ontology import OntologyNode
from adam.ontology.phase1_ontology import (
    GAILA_PHASE_1_ONTOLOGY,
    CAR,
    CHAIR,
    TRUCK,
    PHASE_1_CURRICULUM_OBJECTS,
)
from adam.perception.perception_graph import (
    PerceptionGraph,
    PerceptionGraphPatternMatch,
    PerceptionGraphPattern,
    MatchedObjectPerceptionPredicate)

_LIST_OF_PERCEIVED_PATTERNS = immutableset(
    PerceptionGraphPattern.from_schema(GAILA_PHASE_1_ONTOLOGY.structural_schemata(node)[0])
    for node in PHASE_1_CURRICULUM_OBJECTS
    if node not in [CAR, CHAIR, TRUCK]
)

MATCHED_OBJECT_PATTERN_LABEL = OntologyNode("has-matched-object-pattern")


@attrs
class ObjectRecognizer:
    """
    The ObjectRecognizer finds object matches in the scene pattern and adds a `MatchedObjectPerceptionPredicate`
    which can be used to learn additional semantics which relate objects to other objects
    """

    def match_objects(self, perception_graph: PerceptionGraph, possible_perceived_objects: Iterable[PerceptionGraphPattern] = _LIST_OF_PERCEIVED_PATTERNS) -> (PerceptionGraph, ImmutableSet[MatchedObjectPerceptionPredicate]):
        """
        Match object patterns to objects in the scenes, then add a node for the matched object and copy relationships
        to it. These new patterns can be used to determine static prepositional relationships.
        """
        matched_object_nodes = []
        for pattern in possible_perceived_objects:
            matcher = pattern.matcher(perception_graph)
            pattern_matches = matcher.matches(use_lookahead_pruning=False)
            for pattern_match in pattern_matches:
                self._replace_match_with_object_graph_node(
                    perception_graph._graph, pattern_match, matched_object_nodes,
                )
        return perception_graph, immutableset(matched_object_nodes)

    def _replace_match_with_object_graph_node(
        self,
        networkx_graph_to_modify_in_place: DiGraph,
        pattern_match: PerceptionGraphPatternMatch,
        matched_object_nodes: List[MatchedObjectPerceptionPredicate],
    ):
        """
        Internal function to copy existing relationships from the matched object pattern onto a
        `MatchedObjectPerceptionPredicate`
        """
        node = MatchedObjectPerceptionPredicate()

        matched_object_nodes.append(node)
        networkx_graph_to_modify_in_place.add_node(node)

        for graph_node in pattern_match.matched_sub_graph._graph.nodes:
            for neighbor_node in networkx_graph_to_modify_in_place.neighbors(graph_node):
                if networkx_graph_to_modify_in_place.has_edge(graph_node, neighbor_node):
                    edge_data = networkx_graph_to_modify_in_place.get_edge_data(graph_node, neighbor_node)
                    networkx_graph_to_modify_in_place.add_edge(node, neighbor_node, **edge_data)
            networkx_graph_to_modify_in_place.add_edge(graph_node, node, label=MATCHED_OBJECT_PATTERN_LABEL)
