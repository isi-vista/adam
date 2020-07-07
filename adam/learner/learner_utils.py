import logging
from typing import Mapping, Tuple, Union, cast, List

from networkx import (
    number_weakly_connected_components,
    DiGraph,
    weakly_connected_components,
)

from adam.language import LinguisticDescription, TokenSequenceLinguisticDescription
from adam.learner import LearningExample
from adam.learner.alignments import LanguageConceptAlignment
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.surface_templates import (
    SurfaceTemplate,
    SurfaceTemplateBoundToSemanticNodes,
)
from adam.perception import PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception.perception_graph import (
    ObjectSemanticNodePerceptionPredicate,
    PerceptionGraphPatternMatch,
    PerceptionGraphPattern,
    IsPathPredicate,
    RegionPredicate,
    REFERENCE_OBJECT_LABEL,
    NodePredicate,
    RelationTypeIsPredicate,
)
from adam.semantics import (
    Concept,
    ObjectSemanticNode,
    SemanticNode,
    SyntaxSemanticsVariable,
)
from immutablecollections import immutabledict, immutableset, ImmutableSet

from adam.utils.networkx_utils import subgraph


def pattern_match_to_description(
    *,
    surface_template: SurfaceTemplate,
    pattern: PerceptionGraphTemplate,
    match: PerceptionGraphPatternMatch,
    matched_objects_to_names: Mapping[ObjectSemanticNode, Tuple[str, ...]],
) -> TokenSequenceLinguisticDescription:
    """
    Given a `SurfaceTemplate`, will fill it in using a *match* for a *pattern*.
    This requires a mapping from matched object nodes in the perception
    to the strings which should be used to name them.
    """
    matched_object_nodes = immutableset(
        perception_node
        for perception_node in match.pattern_node_to_matched_graph_node.values()
        if isinstance(perception_node, ObjectSemanticNode)
    )
    matched_object_nodes_without_names = matched_object_nodes - immutableset(
        matched_objects_to_names.keys()
    )
    if matched_object_nodes_without_names:
        raise RuntimeError(
            f"The following matched object nodes lack descriptions: "
            f"{matched_object_nodes_without_names}"
        )

    try:
        return surface_template.instantiate(
            template_variable_to_filler=immutabledict(
                (
                    pattern.pattern_node_to_template_variable[pattern_node],
                    matched_objects_to_names[
                        # We know, but the type system does not,
                        # that if a MatchedObjectPerceptionPredicate matched,
                        # the graph node must be a MatchedObjectNode
                        cast(ObjectSemanticNode, matched_graph_node)
                    ],
                )
                for (
                    pattern_node,
                    matched_graph_node,
                ) in match.pattern_node_to_matched_graph_node.items()
                if isinstance(pattern_node, ObjectSemanticNodePerceptionPredicate)
                # There can sometimes be relevant matched object nodes which are not themselves
                # slots, like the addressed possessor for "your X".
                and pattern_node in pattern.pattern_node_to_template_variable
            )
        )
    except KeyError:
        print("foo")
        raise


def pattern_match_to_semantic_node(
    *,
    concept: Concept,
    pattern: PerceptionGraphTemplate,
    match: PerceptionGraphPatternMatch,
) -> SemanticNode:

    template_variable_to_filler: Mapping[
        SyntaxSemanticsVariable, ObjectSemanticNode
    ] = immutabledict(
        (
            pattern.pattern_node_to_template_variable[pattern_node],
            # We know, but the type system does not,
            # that if a ObjectSemanticNodePerceptionPredicate matched,
            # the graph node must be a MatchedObjectNode
            cast(ObjectSemanticNode, matched_graph_node),
        )
        for (
            pattern_node,
            matched_graph_node,
        ) in match.pattern_node_to_matched_graph_node.items()
        if isinstance(pattern_node, ObjectSemanticNodePerceptionPredicate)
        # There can sometimes be relevant matched object nodes which are not themselves
        # slots, like the addressed possessor for "your X".
        and pattern_node in pattern.pattern_node_to_template_variable
    )

    return SemanticNode.for_concepts_and_arguments(
        concept, slots_to_fillers=template_variable_to_filler
    )


def assert_static_situation(
    to_check: Union[
        LearningExample[DevelopmentalPrimitivePerceptionFrame, LinguisticDescription],
        PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame],
    ]
):
    if isinstance(to_check, LearningExample):
        perception = to_check.perception
    else:
        perception = to_check

    if len(perception.frames) != 1:
        raise RuntimeError("Pursuit learner can only handle single frames for now")
    if not isinstance(perception.frames[0], DevelopmentalPrimitivePerceptionFrame):
        raise RuntimeError(f"Cannot process frame type: {type(perception.frames[0])}")


def pattern_remove_incomplete_region_or_spatial_path(
    perception_graph: PerceptionGraphPattern
) -> PerceptionGraphPattern:
    """
    Helper function to return a `PerceptionGraphPattern` verifying
    that region and spatial path perceptions contain a reference object.
    """
    graph = perception_graph.copy_as_digraph()
    region_and_path_nodes: ImmutableSet[NodePredicate] = immutableset(
        node
        for node in graph.nodes
        if isinstance(node, IsPathPredicate) or isinstance(node, RegionPredicate)
    )
    nodes_without_reference: List[NodePredicate] = []
    for node in region_and_path_nodes:
        has_reference_edge: bool = False
        for successor in graph.successors(node):
            predicate = graph.edges[node, successor]["predicate"]
            if isinstance(predicate, RelationTypeIsPredicate):
                if predicate.relation_type == REFERENCE_OBJECT_LABEL:
                    has_reference_edge = True
                    break
        if not has_reference_edge:
            nodes_without_reference.append(node)

    logging.info(
        f"Removing incomplete regions and paths. "
        f"Removing nodes: {nodes_without_reference}"
    )
    graph.remove_nodes_from(nodes_without_reference)

    def sort_by_num_nodes(g: DiGraph) -> int:
        return len(g.nodes)

    # We should maybe consider doing this a different way
    # As this approach just brute force solves the problem rather than being methodical about it
    if number_weakly_connected_components(graph) > 1:
        components = [
            component
            for component in [
                subgraph(graph, comp) for comp in weakly_connected_components(graph)
            ]
        ]
        components.sort(key=sort_by_num_nodes, reverse=True)
        computed_graph = subgraph(graph, components[0].nodes)
        removed_nodes: List[NodePredicate] = []
        for i in range(1, len(components)):
            removed_nodes.extend(components[i].nodes)
        logging.info(f"Cleanup disconnected elements. Removing: {removed_nodes}")
    else:
        computed_graph = graph

    return PerceptionGraphPattern(computed_graph, dynamic=perception_graph.dynamic)


def covers_entire_utterance(
    bound_surface_template: SurfaceTemplateBoundToSemanticNodes,
    language_concept_alignment: LanguageConceptAlignment,
    *,
    token_sequence_count: int,
) -> bool:
    num_covered_tokens = 0
    for element in bound_surface_template.surface_template.elements:
        if isinstance(element, str):
            num_covered_tokens += 1
        else:
            num_covered_tokens += len(
                language_concept_alignment.node_to_language_span[
                    bound_surface_template.slot_to_semantic_node[element]
                ]
            )
    # This assumes the slots and the non-slot elements are non-overlapping,
    # which is true for how we construct them.
    return num_covered_tokens == token_sequence_count
