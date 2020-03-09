from typing import Mapping

from more_itertools import flatten
from networkx import all_shortest_paths, subgraph

from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.surface_templates import SurfaceTemplateVariable, SLOT1, SLOT2
from adam.perception import ObjectPerception
from adam.perception.perception_graph import (
    MatchedObjectNode,
    LanguageAlignedPerception,
    PerceptionGraphNode,
    _graph_node_order,
)
from adam.utils.networkx_utils import digraph_with_nodes_sorted_by
from immutablecollections import ImmutableSet, immutableset


def preposition_hypothesis_from_perception(
    scene_aligned_perception: LanguageAlignedPerception,
    template_variables_to_object_match_nodes: Mapping[
        SurfaceTemplateVariable, MatchedObjectNode
    ],
) -> PerceptionGraphTemplate:
    """
        Create a hypothesis for the semantics of a preposition based on the observed scene.

        Our current implementation is to just include the content
        on the path between the recognized object nodes
        and one hop away from that path.
        """

    # The directions of edges in the perception graph are not necessarily meaningful
    # from the point-of-view of hypothesis generation, so we need an undirected copy
    # of the graph.
    perception_digraph = scene_aligned_perception.perception_graph.copy_as_digraph()
    perception_graph_undirected = perception_digraph.to_undirected(
        # as_view=True loses determinism
        as_view=False
    )

    if {SLOT1, SLOT2} != set(template_variables_to_object_match_nodes.keys()):
        raise RuntimeError(
            "Can only make a preposition hypothesis if the recognized "
            "objects are aligned to SurfaceTemplateVariables SLOT1 and SLOT2"
        )

    slot1_object = template_variables_to_object_match_nodes[SLOT1]
    slot2_object = template_variables_to_object_match_nodes[SLOT2]

    # The core of our hypothesis for the semantics of a preposition is all nodes
    # along the shortest path between the two objects involved in the perception graph.
    hypothesis_spine_nodes: ImmutableSet[PerceptionGraphNode] = immutableset(
        flatten(
            # if there are multiple paths between the object match nodes,
            # we aren't sure which are relevant, so we include them all in our hypothesis
            # and figure we can trim out irrelevant stuff as we make more observations.
            all_shortest_paths(perception_graph_undirected, slot2_object, slot1_object)
        )
    )

    # Along the core of our hypothesis we also want to collect the predecessors and successors
    hypothesis_nodes_mutable = []
    for node in hypothesis_spine_nodes:
        if node not in {slot1_object, slot2_object}:
            for successor in perception_digraph.successors(node):
                if not isinstance(successor, ObjectPerception):
                    hypothesis_nodes_mutable.append(successor)
            for predecessor in perception_digraph.predecessors(node):
                if not isinstance(predecessor, ObjectPerception):
                    hypothesis_nodes_mutable.append(predecessor)

    hypothesis_nodes_mutable.extend(hypothesis_spine_nodes)

    # We wrap the nodes in an immutable set to remove duplicates
    # while preserving iteration determinism.
    hypothesis_nodes = immutableset(hypothesis_nodes_mutable)

    preposition_pattern_graph = digraph_with_nodes_sorted_by(
        subgraph(perception_digraph, hypothesis_nodes), _graph_node_order
    )
    return PerceptionGraphTemplate.from_graph(
        preposition_pattern_graph, template_variables_to_object_match_nodes.items()
    )
