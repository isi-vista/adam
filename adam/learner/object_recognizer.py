import logging
from itertools import chain
from pathlib import Path
from typing import Iterable, List, Tuple

from immutablecollections import ImmutableDict, ImmutableSet, immutabledict, immutableset
from immutablecollections.converter_utils import _to_immutabledict
from more_itertools import first
from networkx import DiGraph

from adam.axes import GRAVITATIONAL_DOWN_TO_UP_AXIS, LEARNER_AXES, WORLD_AXES
from adam.ontology import OntologyNode
from adam.ontology.phase1_ontology import (
    GAILA_PHASE_1_ONTOLOGY,
    PHASE_1_CURRICULUM_OBJECTS,
)
from adam.perception.perception_graph import (
    GraphLogger,
    MatchedObjectNode,
    PerceptionGraph,
    PerceptionGraphNode,
    PerceptionGraphPattern,
    PerceptionGraphPatternMatch,
    RelationTypeIsPredicate,
)
from attr import attrib, attrs

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


# these are shared aspects of the world which, although they might be referenced by
# object recognition patterns, should not be deleted when those patterns are match.
# For example, a geon axis local to an object is no longer needed when the object
# has been recognized, but we still need the gravitational axes
SHARED_WORLD_ITEMS = set(
    chain([GRAVITATIONAL_DOWN_TO_UP_AXIS], WORLD_AXES.all_axes, LEARNER_AXES.all_axes)
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
        logger = GraphLogger(Path("/Users/gabbard/tmp"), enable_graph_rendering=True)
        matched_object_nodes: List[Tuple[str, MatchedObjectNode]] = []
        graph_to_return = perception_graph.copy_as_digraph()
        for (description, pattern) in possible_perceived_objects:
            logger.log_graph(pattern, logging.INFO, description)
            matcher = pattern.matcher(PerceptionGraph(graph_to_return))
            pattern_match = first(matcher.matches(use_lookahead_pruning=True), None)
            # It's important not to simply iterate over pattern matches
            # because they might overlap, or be variants of the same match
            # (e.g. permutations of how table legs match)
            while pattern_match:
                self._replace_match_with_object_graph_node(
                    graph_to_return, pattern_match, matched_object_nodes, description
                )
                matcher = pattern.matcher(PerceptionGraph(graph_to_return))
                pattern_match = first(matcher.matches(use_lookahead_pruning=True), None)
        if matched_object_nodes:
            logging.info(
                "Object recognizer recognized: %s", [x[0] for x in matched_object_nodes]
            )
        return PerceptionGraphFromObjectRecognizer(
            perception_graph=PerceptionGraph(graph=graph_to_return),
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
            if isinstance(matched_subgraph_node, MatchedObjectNode):
                raise RuntimeError(
                    f"We do not currently allow object recognitions to themselves "
                    f"operate over other object recognitions, but got match "
                    f"{pattern_match.matched_sub_graph}"
                )

            if matched_subgraph_node in SHARED_WORLD_ITEMS:
                continue

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
                    label = edge_data["label"]
                    if (
                        isinstance(label, RelationTypeIsPredicate)
                        and label.dot_label == "rel(" "has-matched-object-pattern)"
                    ):
                        raise RuntimeError(
                            f"Overlapping nodes in object recognition: "
                            f"{matched_subgraph_node}, "
                            f"{matched_subgraph_node_successor}"
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
                    label = edge_data["label"]
                    if (
                        isinstance(label, RelationTypeIsPredicate)
                        and label.dot_label == "rel(" "has-matched-object-pattern)"
                    ):
                        raise RuntimeError(
                            f"Overlapping nodes in object recognition: "
                            f"{matched_subgraph_node}, "
                            f"{matched_subgraph_node_predecessor}"
                        )

                    networkx_graph_to_modify_in_place.add_edge(
                        matched_subgraph_node_predecessor,
                        matched_object_node,
                        **edge_data,
                    )

            # we also link every node in the matched sub-graph to the newly introduced node
            # representing the object match.
            # networkx_graph_to_modify_in_place.add_edge(
            #     matched_subgraph_node,
            #     matched_object_node,
            #     label=MATCHED_OBJECT_PATTERN_LABEL,
            # )
        networkx_graph_to_modify_in_place.remove_nodes_from(
            matched_node
            for matched_node in matched_subgraph_nodes
            if matched_node not in SHARED_WORLD_ITEMS
        )
