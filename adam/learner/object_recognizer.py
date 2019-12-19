import logging
from typing import Iterable, List, Tuple

from immutablecollections import ImmutableDict, immutabledict, immutableset, ImmutableSet
from immutablecollections.converter_utils import _to_immutabledict
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
    PerceptionGraphNode,
)
from attr import attrs, attrib

OBJECTS_WHICH_FAIL_TO_MATCH_DUE_TO_KNOWN_BUG = [TRUCK]

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
    and node not in OBJECTS_WHICH_FAIL_TO_MATCH_DUE_TO_KNOWN_BUG
)

MATCHED_OBJECT_PATTERN_LABEL = OntologyNode("has-matched-object-pattern")


@attrs(frozen=True, slots=True, auto_attribs=True)
class PerceptionGraphFromObjectRecognizer:
    """
    See `ObjectRecognizer.match_objects`
    """

    perception_graph: PerceptionGraph
    description_to_matched_object_node: ImmutableDict[str, MatchedObjectNode] = attrib(
        converter=_to_immutabledict
    )


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
    ) -> PerceptionGraphFromObjectRecognizer:
        """
        Match object patterns to objects in the scenes, then add a node for the matched object and copy relationships
        to it. These new patterns can be used to determine static prepositional relationships.
        """
        matched_object_nodes: List[Tuple[str, MatchedObjectNode]] = []
        graph_to_modify = perception_graph.copy_as_digraph()
        for (description, pattern) in possible_perceived_objects:
            matcher = pattern.matcher(perception_graph)
            pattern_matches = list(matcher.matches(use_lookahead_pruning=True))
            for pattern_match in pattern_matches:
                self._replace_match_with_object_graph_node(
                    graph_to_modify, pattern_match, matched_object_nodes, description
                )
        if matched_object_nodes:
            logging.info(
                "Object recognizer recognized: %s", [x[0] for x in matched_object_nodes]
            )
        return PerceptionGraphFromObjectRecognizer(
            perception_graph=PerceptionGraph(graph=graph_to_modify),
            description_to_matched_object_node=immutabledict(matched_object_nodes),
        )

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
        matched_object_node = MatchedObjectNode(name=(description,))

        matched_object_nodes.append((description, matched_object_node))
        networkx_graph_to_modify_in_place.add_node(matched_object_node)

        matched_subgraph_nodes: ImmutableSet[PerceptionGraphNode] = immutableset(
            pattern_match.matched_sub_graph._graph.nodes,  # pylint:disable=protected-access
            disable_order_check=True,
        )

        for matched_subgraph_node in matched_subgraph_nodes:
            # If there is an edge from the matched sub-graph to a node outside it,
            # also add an edge from the object match node to that node.
            for (
                matched_subgraph_node_successor
            ) in networkx_graph_to_modify_in_place.successors(matched_subgraph_node):
                # don't want to add edges which are internal to the matched sub-graph
                if matched_subgraph_node_successor not in matched_subgraph_nodes:
                    edge_data = networkx_graph_to_modify_in_place.get_edge_data(
                        matched_subgraph_node, matched_subgraph_node_successor
                    )
                    networkx_graph_to_modify_in_place.add_edge(
                        matched_object_node, matched_subgraph_node_successor, **edge_data
                    )

            # If there is an edge to the matched sub-graph from a node outside it,
            # also add an edge to the object match node from that node.
            for (
                matched_subgraph_node_predecessor
            ) in networkx_graph_to_modify_in_place.predecessors(matched_subgraph_node):
                # don't want to add edges which are internal to the matched sub-graph
                if matched_subgraph_node_predecessor not in matched_subgraph_nodes:
                    edge_data = networkx_graph_to_modify_in_place.get_edge_data(
                        matched_subgraph_node_predecessor, matched_subgraph_node
                    )
                    networkx_graph_to_modify_in_place.add_edge(
                        matched_subgraph_node_predecessor,
                        matched_object_node,
                        **edge_data
                    )

            # we also link every node in the matched sub-graph to the newly introduced node
            # representing the object match.
            networkx_graph_to_modify_in_place.add_edge(
                matched_subgraph_node,
                matched_object_node,
                label=MATCHED_OBJECT_PATTERN_LABEL,
            )
