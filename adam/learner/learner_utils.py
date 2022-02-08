import itertools
import logging
from enum import Enum, auto
from typing import (
    Mapping,
    Tuple,
    Union,
    cast,
    List,
    AbstractSet,
    Dict,
    Iterable,
    Optional,
    Callable,
    Sequence,
)
from adam.language_specific.english import DETERMINERS

from attr import attrib, attrs
from attr.validators import instance_of, optional
from immutablecollections import immutabledict, immutableset, ImmutableSet
from networkx import (
    number_weakly_connected_components,
    DiGraph,
    weakly_connected_components,
)
from vistautils.span import Span

from adam.language import LinguisticDescription, TokenSequenceLinguisticDescription
from adam.learner import LearningExample, get_largest_matching_pattern
from adam.learner.alignments import LanguageConceptAlignment
from adam.learner.alignments import LanguagePerceptionSemanticAlignment
from adam.learner.language_mode import LanguageMode
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.surface_templates import (
    STANDARD_SLOT_VARIABLES,
    SurfaceTemplate,
    SurfaceTemplateBoundToSemanticNodes,
)
from adam.ontology.ontology import Ontology
from adam.ontology.phase1_spatial_relations import Region
from adam.perception import PerceptualRepresentation, MatchMode, ObjectPerception
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
    RgbColorPerception,
)
from adam.perception.perception_graph import (
    ObjectSemanticNodePerceptionPredicate,
    PerceptionGraphPatternMatch,
    PerceptionGraphPattern,
    IsPathPredicate,
    RegionPredicate,
    REFERENCE_OBJECT_LABEL,
    REFERENCE_OBJECT_DESTINATION_LABEL,
    REFERENCE_OBJECT_SOURCE_LABEL,
    NodePredicate,
    RelationTypeIsPredicate,
    PerceptionGraph,
    DebugCallableType,
    GraphLogger,
)
from adam.semantics import (
    Concept,
    ObjectSemanticNode,
    SemanticNode,
    SyntaxSemanticsVariable,
)
from adam.utils import networkx_utils


def get_classifier_for_string(input_string: str) -> Optional[str]:
    if input_string in ["chwang2", "jr3", "jwo1 dz"]:
        return "yi1_jang1"
    elif input_string in ["shu1"]:
        return "yi1_ben3"
    elif input_string in ["wu1"]:
        return "yi1_jyan1"
    elif input_string in ["chi4 che1", "ka3 che1"]:
        return "yi1_lyang4"
    elif input_string in ["yi3 dz"]:
        return "yi1_ba3"
    elif input_string in ["shou3", "gou3", "mau1", "nyau3", "syung2"]:
        return "yi1_jr1"
    elif input_string in ["men2"]:
        return "yi1_shan4"
    elif input_string in ["mau4 dz"]:
        return "yi1_ding3"
    elif input_string in ["chyu1 chi2 bing3", "niu2 rou1"]:
        return "yi1_kwai4"
    elif input_string in ["niu2"]:
        return "yi1_tiao2"
    elif input_string in ["ji1"]:
        return "yi1_zhi1"
    # eliminate mass and proper nouns and use the default classifier if another one hasn't already been used
    elif input_string not in [
        "ba4 ba4",
        "ma1 ma1",
        "shwei3",
        "gwo3 jr1",
        "nyou2 nai3",
        "di4 myan4",
    ]:
        return "yi1_ge4"
    return None


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
    except KeyError as e:
        logging.warning(str(e))
        raise


def pattern_match_to_semantic_node(
    *,
    concept: Concept,
    pattern: PerceptionGraphTemplate,
    match: PerceptionGraphPatternMatch,
    confidence: float,
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
        concept, slots_to_fillers=template_variable_to_filler, confidence=confidence
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
    perception_graph: PerceptionGraphPattern,
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
                if predicate.relation_type in [
                    REFERENCE_OBJECT_LABEL,
                    REFERENCE_OBJECT_DESTINATION_LABEL,
                    REFERENCE_OBJECT_SOURCE_LABEL,
                ]:
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
                graph.subgraph(comp) for comp in weakly_connected_components(graph)
            ]
        ]
        components.sort(key=sort_by_num_nodes, reverse=True)
        computed_graph = graph.subgraph(components[0].nodes)
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
    ignore_determiners: bool = False,
) -> bool:
    num_covered_tokens = 0
    for element in bound_surface_template.surface_template.elements:
        if isinstance(element, str):
            # this is a hack for the classifier learning experiment (the classifier will be ignored with sized so we must ignore it here too)
            if element[:3] != "yi1":
                num_covered_tokens += 1
        else:
            slot_for_element = language_concept_alignment.node_to_language_span[
                bound_surface_template.slot_to_semantic_node[element]
            ]
            aligned_strings_for_slot = language_concept_alignment.language[
                slot_for_element.start : slot_for_element.end
            ]
            # we need to check here that the determiners aren't getting aligned; otherwise it can mess up our count
            if ignore_determiners:
                num_covered_tokens += len(
                    [x for x in aligned_strings_for_slot if x not in DETERMINERS]
                )
            else:
                num_covered_tokens += len(aligned_strings_for_slot)

    # We may need to ignore counting english determiners in our comparison
    # to the template as the way we treat english determiners is currently
    # a hack. See: https://github.com/isi-vista/adam/issues/498
    sized_tokens = (
        len([token for token in language_concept_alignment.language.as_token_sequence()])
        if not ignore_determiners
        else len(
            [
                token
                for token in language_concept_alignment.language.as_token_sequence()
                if token not in DETERMINERS
            ]
        )
    )
    # This assumes the slots and the non-slot elements are non-overlapping,
    # which is true for how we construct them.
    return num_covered_tokens == sized_tokens


class AlignmentSlots(Enum):
    """An argument is a slot for an object, and a fixed string is something we wish to learn"""

    ARGUMENT = auto()
    FIXEDSTRING = auto()


@attrs(frozen=True, slots=True)
class SemanticNodeWithSpan:
    """This is a tuple class currently used by our verb and relation learners"""

    node: SemanticNode = attrib(validator=instance_of(SemanticNode))
    span: Span = attrib(validator=instance_of(Span))


def candidate_templates(
    language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment,
    max_length: int,
    language_mode: LanguageMode,
    candidate_templates_function: Callable[[], Iterable[Tuple[AlignmentSlots, ...]]],
) -> AbstractSet[SurfaceTemplateBoundToSemanticNodes]:
    ret = []
    language_concept_alignment = (
        language_perception_semantic_alignment.language_concept_alignment
    )
    # We make an assumption that the order of nouns in our sentence will be
    # in the same order as they appear in the sentence, left to right,
    # To avoid calculating this condition multiple times we do so once
    # For the number of arguments (nouns) our candidate template
    # desires and store the resulting possible aligments in this dict
    # for easy look up.
    num_arguments_to_alignments_sets: Dict[
        int, ImmutableSet[Tuple[SemanticNodeWithSpan, ...]]
    ] = {}
    sentence_tokens = language_concept_alignment.language.as_token_sequence()

    # Any recognized object is a potential verb argument.
    # This method does not properly handle arguments which themselves have complex structure.
    # Arguments are restricted to ONLY be ObjectSemanticNodes
    # See https://github.com/isi-vista/adam/issues/785

    # We currently do not handle verb arguments
    # which are dropped and/or expressed only via morphology:
    # https://github.com/isi-vista/adam/issues/786

    def in_left_to_right_order(semantic_nodes: Tuple[SemanticNodeWithSpan, ...]) -> bool:
        previous_node = semantic_nodes[0]
        for i in range(1, len(semantic_nodes)):
            if not previous_node.span.precedes(semantic_nodes[i].span):
                return False
            previous_node = semantic_nodes[i]
        return True

    # A sample case for invalid tokens spans is if the language of a situation is
    # "a dog pushes a box to a car" but we are given a candidate template
    # of AFA then we could generate "slot1 pushes a box to slot2"
    # which is an undesired output. So if we provide the spans where
    # aligned objects are we can invalidate templates like the problem one above
    def is_legal_template_span(
        candidate_token_span: Span, *, invalid_token_spans: ImmutableSet[Span]
    ) -> bool:
        # A template token span can't exceed the bounds of the utterance
        if candidate_token_span.start < 0:
            return False
        if candidate_token_span.end > len(sentence_tokens):
            return False
        # or be bigger than our maximum template size...
        if len(candidate_token_span) > max_length:
            return False

        for span in invalid_token_spans:
            if candidate_token_span.contains_span(span):
                return False

        # or we have already aligned any of the tokens in between the objects
        # to some other meaning.
        for token_index in range(candidate_token_span.start, candidate_token_span.end):
            if language_concept_alignment.token_index_is_aligned(token_index):
                return False

        return True

    def aligned_object_nodes(
        num_arguments: int,
        num_arguments_to_alignments_sets: Dict[
            int, ImmutableSet[Tuple[SemanticNodeWithSpan, ...]]
        ],
        language_concept_alignment: LanguageConceptAlignment,
    ) -> ImmutableSet[Tuple[SemanticNodeWithSpan, ...]]:
        if num_arguments not in num_arguments_to_alignments_sets.keys():
            # we haven't seen a request for this number of arguments before so we need to generate all the valid options
            semantic_nodes_with_spans = immutableset(
                SemanticNodeWithSpan(node=node, span=span)
                for (
                    node,
                    span,
                ) in language_concept_alignment.node_to_language_span.items()
                if isinstance(node, ObjectSemanticNode)
            )
            num_arguments_to_alignments_sets[num_arguments] = immutableset(
                ordered_semantic_nodes
                for ordered_semantic_nodes in itertools.product(
                    semantic_nodes_with_spans, repeat=num_arguments
                )
                if in_left_to_right_order(ordered_semantic_nodes)
            )
        return num_arguments_to_alignments_sets[num_arguments]

    def process_aligned_objects_with_template(
        candidate_template: Tuple[AlignmentSlots, ...],
        aligned_nodes: Tuple[SemanticNodeWithSpan, ...],
        *,
        invalid_token_spans: ImmutableSet[Span],
    ) -> Iterable[Optional[SurfaceTemplateBoundToSemanticNodes]]:

        aligned_node_index = 0
        template_elements: List[Union[str, SyntaxSemanticsVariable]] = []
        slot_to_semantic_node: List[Tuple[SyntaxSemanticsVariable, SemanticNode]] = []

        # We need to handle fixed strings that are pre or post fix to the rest of the
        # Sentence differently as they don't have a fixed length so we could generate
        # multiple options.
        prefix_string_end = None
        postfix_string_start = None
        # In the event we generate a candidate template like:
        # A, F, F, A then we want to compute this like A, F, A
        # So we keep track if the previous token was a FIXEDSTRING indicator
        previous_node_was_string = False

        for token in candidate_template:
            # If the token in our template is an argument we need to assign it a
            # unique SyntaxSemanticsVariable, and map it to the SemanticNode
            if token == AlignmentSlots.ARGUMENT:
                slot_semantic_variable = STANDARD_SLOT_VARIABLES[aligned_node_index]
                template_elements.append(slot_semantic_variable)
                aligned_node = aligned_nodes[aligned_node_index].node
                if not isinstance(aligned_node, ObjectSemanticNode):
                    logging.debug(
                        f"Attempted to make template where an Argument is not an ObjectSemanticNode."
                        f"Invalid node: {aligned_node}"
                    )
                    # Log this failure and then ignore this attempt
                    yield None
                slot_to_semantic_node.append((slot_semantic_variable, aligned_node))
                aligned_node_index += 1
                previous_node_was_string = False
            else:
                # We ignore this case to process A, F, F, A like A, F, A
                if previous_node_was_string:
                    continue
                # We make a note of where the end of our prefix string can be
                # Then continue as we'll handle this case afterwards
                elif aligned_node_index == 0:
                    prefix_string_end = aligned_nodes[aligned_node_index].span.start
                # Similiarly to above, we instead mark the start of the postfix string
                elif aligned_node_index == len(aligned_nodes):
                    postfix_string_start = aligned_nodes[aligned_node_index - 1].span.end
                else:
                    # If our FIXEDSTRING is flanked by two Arguments we just want to acquire all the tokens
                    # between them
                    if (
                        aligned_nodes[aligned_node_index - 1].span.end
                        != aligned_nodes[aligned_node_index].span.start
                    ):
                        candidate_token_span = Span(
                            aligned_nodes[aligned_node_index - 1].span.end,
                            aligned_nodes[aligned_node_index].span.start,
                        )
                        if not is_legal_template_span(
                            candidate_token_span, invalid_token_spans=invalid_token_spans
                        ):
                            # If not a valid span, ignore this attempt
                            continue
                        template_elements.extend(
                            sentence_tokens[
                                candidate_token_span.start : candidate_token_span.end
                            ]
                        )
                    previous_node_was_string = True
        # We need to handle searching before or after the aligned token
        # And we could generate multiple options of different lengths
        # between 1 and _MAXIMUM_ACTION_TEMPLATE_TOKEN_LENGTH
        if prefix_string_end and postfix_string_start:
            for max_token_length_for_template_prefix in range(1, max_length + 1):
                prefix_candidate_token_span = Span(
                    prefix_string_end - max_token_length_for_template_prefix,
                    prefix_string_end,
                )
                if is_legal_template_span(
                    prefix_candidate_token_span, invalid_token_spans=invalid_token_spans
                ):
                    for max_token_length_for_template_postfix in range(1, max_length + 1):
                        postfix_candidate_token_span = Span(
                            postfix_string_start,
                            postfix_string_start + max_token_length_for_template_postfix,
                        )
                        if is_legal_template_span(
                            postfix_candidate_token_span,
                            invalid_token_spans=invalid_token_spans,
                        ):
                            final_template_elements: List[
                                Union[str, SyntaxSemanticsVariable]
                            ] = list(
                                sentence_tokens[
                                    prefix_candidate_token_span.start : prefix_candidate_token_span.end
                                ]
                            )
                            final_template_elements.extend(template_elements)
                            final_template_elements.extend(
                                sentence_tokens[
                                    postfix_candidate_token_span.start : postfix_candidate_token_span.end
                                ]
                            )
                            yield SurfaceTemplateBoundToSemanticNodes(
                                surface_template=SurfaceTemplate(
                                    elements=final_template_elements,
                                    determiner_prefix_slots=[
                                        SLOT for (SLOT, _) in slot_to_semantic_node
                                    ],
                                    language_mode=language_mode,
                                ),
                                slot_to_semantic_node=slot_to_semantic_node,
                            )
        elif prefix_string_end:
            for max_token_length_for_template_prefix in range(1, max_length + 1):
                prefix_candidate_token_span = Span(
                    prefix_string_end - max_token_length_for_template_prefix,
                    prefix_string_end,
                )
                if is_legal_template_span(
                    prefix_candidate_token_span, invalid_token_spans=invalid_token_spans
                ):
                    final_template_elements = list(
                        sentence_tokens[
                            prefix_candidate_token_span.start : prefix_candidate_token_span.end
                        ]
                    )
                    final_template_elements.extend(template_elements)
                    yield SurfaceTemplateBoundToSemanticNodes(
                        surface_template=SurfaceTemplate(
                            elements=final_template_elements,
                            determiner_prefix_slots=[
                                SLOT for (SLOT, _) in slot_to_semantic_node
                            ],
                            language_mode=language_mode,
                        ),
                        slot_to_semantic_node=slot_to_semantic_node,
                    )
        elif postfix_string_start:
            for max_token_length_for_template_postfix in range(1, max_length + 1):
                postfix_candidate_token_span = Span(
                    postfix_string_start,
                    postfix_string_start + max_token_length_for_template_postfix,
                )
                if is_legal_template_span(
                    postfix_candidate_token_span, invalid_token_spans=invalid_token_spans
                ):
                    final_template_elements = list(template_elements)
                    final_template_elements.extend(
                        sentence_tokens[
                            postfix_candidate_token_span.start : postfix_candidate_token_span.end
                        ]
                    )
                    yield SurfaceTemplateBoundToSemanticNodes(
                        surface_template=SurfaceTemplate(
                            elements=final_template_elements,
                            determiner_prefix_slots=[
                                SLOT for (SLOT, _) in slot_to_semantic_node
                            ],
                            language_mode=language_mode,
                        ),
                        slot_to_semantic_node=slot_to_semantic_node,
                    )
        else:
            yield SurfaceTemplateBoundToSemanticNodes(
                surface_template=SurfaceTemplate(
                    elements=template_elements,
                    determiner_prefix_slots=[SLOT for (SLOT, _) in slot_to_semantic_node],
                    language_mode=language_mode,
                ),
                slot_to_semantic_node=slot_to_semantic_node,
            )

    # Generate all the possible verb template alignments
    for candidate_template in candidate_templates_function():
        for aligned_nodes in aligned_object_nodes(
            sum(1 for token in candidate_template if token == AlignmentSlots.ARGUMENT),
            num_arguments_to_alignments_sets,
            language_concept_alignment,
        ):
            # aligned_object_nodes is guaranteed to only give us alignments
            # Which the spans go from left most to right most
            # We also provide a set of invalid token spans
            # for fixed string positions.
            # see: https://github.com/isi-vista/adam/issues/867
            invalid_token_spans = immutableset(
                language_concept_alignment.node_to_language_span.values()
            )
            for (
                surface_template_bound_to_semantic_nodes
            ) in process_aligned_objects_with_template(
                candidate_template, aligned_nodes, invalid_token_spans=invalid_token_spans
            ):
                if surface_template_bound_to_semantic_nodes:
                    ret.append(surface_template_bound_to_semantic_nodes)

    def contains_str(bst: SurfaceTemplateBoundToSemanticNodes) -> bool:
        for element in bst.surface_template.elements:
            if isinstance(element, str):
                return True
        return False

    return immutableset(
        bound_surface_template
        for bound_surface_template in ret
        # For now, we require templates to account for the entire utterance.
        # See https://github.com/isi-vista/adam/issues/789
        if covers_entire_utterance(bound_surface_template, language_concept_alignment)
        and contains_str(bound_surface_template)
    )


def default_post_process_enrichment(
    perception_graph_after_matching: PerceptionGraph,
    immutable_new_nodes: AbstractSet[SemanticNode],
) -> Tuple[PerceptionGraph, AbstractSet[SemanticNode]]:
    return perception_graph_after_matching, immutable_new_nodes


@attrs(frozen=True, slots=True)
class PartialMatchRatio:
    """
    See `compute_match_ratio`
    """

    matching_subgraph: Optional[PerceptionGraphPattern] = attrib(
        validator=optional(instance_of(PerceptionGraphPattern))
    )
    num_nodes_matched: int = attrib(validator=instance_of(int), kw_only=True)
    num_nodes_in_pattern: int = attrib(validator=instance_of(int), kw_only=True)
    matched_exactly: bool = attrib(init=False)
    match_ratio: float = attrib(init=False)

    @matched_exactly.default
    def matched(self) -> bool:
        return self.num_nodes_in_pattern == self.num_nodes_matched

    @match_ratio.default
    def ratio(self) -> float:
        return self.num_nodes_matched / self.num_nodes_in_pattern


def compute_match_ratio(
    pattern: PerceptionGraphTemplate,
    graph: PerceptionGraph,
    ontology: Ontology,
    *,
    graph_logger: Optional[GraphLogger] = None,
    debug_callback: Optional[DebugCallableType] = None,
) -> PartialMatchRatio:
    """
    Computes the fraction of pattern graph nodes of *pattern* which match *graph*.
    """
    hypothesis_pattern_common_subgraph = get_largest_matching_pattern(
        pattern.graph_pattern,
        graph,
        debug_callback=debug_callback,
        graph_logger=graph_logger,
        ontology=ontology,
        match_mode=MatchMode.OBJECT,
    )

    return PartialMatchRatio(
        hypothesis_pattern_common_subgraph,
        num_nodes_matched=(
            len(hypothesis_pattern_common_subgraph.copy_as_digraph().nodes)
            if hypothesis_pattern_common_subgraph
            else 0
        ),
        num_nodes_in_pattern=len(pattern.graph_pattern),
    )


def get_objects_from_perception(
    observed_perception_graph: PerceptionGraph,
) -> List[PerceptionGraph]:
    """
    Utility function to get a list of `PerceptionGraphs` which are independent objects in the scene
    """
    perception_as_digraph = observed_perception_graph.copy_as_digraph()
    perception_as_graph = perception_as_digraph.to_undirected()

    meanings = []

    # 1) Take all of the obj perc that dont have part of relationships with anything else
    root_object_percetion_nodes = []
    for node in perception_as_graph.nodes:
        if isinstance(node, ObjectPerception) and node.debug_handle != "the ground":
            if not any(
                [
                    u == node and str(data["label"]) == "partOf"
                    for u, v, data in perception_as_digraph.edges.data()
                ]
            ):
                root_object_percetion_nodes.append(node)

    # 2) for each of these, walk along the part of relationships backwards,
    # i.e find all of the subparts of the root object
    for root_object_perception_node in root_object_percetion_nodes:
        # Iteratively get all other object perceptions that connect to a root with a part of
        # relation
        all_object_perception_nodes = [root_object_perception_node]
        frontier = [root_object_perception_node]
        updated = True
        while updated:
            updated = False
            new_frontier = []
            for frontier_node in frontier:
                for node in perception_as_graph.neighbors(frontier_node):
                    edge_data = perception_as_digraph.get_edge_data(
                        node, frontier_node, default=-1
                    )
                    if edge_data != -1 and str(edge_data["label"]) == "partOf":
                        new_frontier.append(node)

            if new_frontier:
                all_object_perception_nodes.extend(new_frontier)
                updated = True
                frontier = new_frontier

        # Now we have a list of all perceptions that are connected
        # 3) For each of these objects including root object, get axes, properties,
        # and relations and regions which are between these internal object perceptions
        other_nodes = []
        for node in all_object_perception_nodes:
            for neighbor in perception_as_graph.neighbors(node):
                # Filter out regions that don't have a reference in all object perception nodes
                # TODO: We currently remove colors to achieve a match - otherwise finding
                #  patterns fails.
                if (
                    isinstance(neighbor, Region)
                    and neighbor.reference_object not in all_object_perception_nodes
                    or isinstance(neighbor, RgbColorPerception)
                ):
                    continue
                # Append all other none-object nodes to be kept in the subgraph
                if not isinstance(neighbor, ObjectPerception):
                    other_nodes.append(neighbor)

        generated_subgraph = networkx_utils.subgraph(
            perception_as_digraph, all_object_perception_nodes + other_nodes
        )
        meanings.append(PerceptionGraph(generated_subgraph))

    logging.info(f"Got {len(meanings)} candidate meanings")
    return meanings


def candidate_object_hypotheses(
    language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment,
) -> Sequence[PerceptionGraphTemplate]:
    """
    Given a learning input, returns all possible meaning hypotheses.
    """
    return [
        PerceptionGraphTemplate(
            graph_pattern=PerceptionGraphPattern.from_graph(
                object_
            ).perception_graph_pattern
        )
        for object_ in get_objects_from_perception(
            language_perception_semantic_alignment.perception_semantic_alignment.perception_graph
        )
    ]


def get_slot_from_semantic_node(
    object_concept: Concept, semantic_node: SemanticNode
) -> str:
    slot = ""
    for slot_var, object_node in semantic_node.slot_fillings.items():
        if object_node.concept == object_concept:
            return slot_var.name
    return slot
