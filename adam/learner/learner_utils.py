import logging
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
)
import itertools
from adam.learner.alignments import LanguagePerceptionSemanticAlignment
from attr.validators import instance_of
from networkx import (
    number_weakly_connected_components,
    DiGraph,
    weakly_connected_components,
)
from attr import attrib, attrs
from enum import Enum, auto
from adam.language import LinguisticDescription, TokenSequenceLinguisticDescription
from adam.learner import LearningExample
from adam.learner.alignments import LanguageConceptAlignment
from adam.learner.language_mode import LanguageMode
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.perception import PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.learner.surface_templates import (
    STANDARD_SLOT_VARIABLES,
    SurfaceTemplate,
    SurfaceTemplateBoundToSemanticNodes,
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
from vistautils.span import Span


def pattern_match_to_description(
    *,
    surface_template: SurfaceTemplate,
    pattern: PerceptionGraphTemplate,
    match: PerceptionGraphPatternMatch,
    matched_objects_to_names: Mapping[ObjectSemanticNode, Tuple[str, ...]],
    allow_undescribed: bool = False,
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
    if matched_object_nodes_without_names and not allow_undescribed:
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
    ignore_determiners: bool = False,
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
    # We may need to ignore counting english determiners in our comparison
    # to the template as the way we treat english determiners is currently
    # a hack. See: https://github.com/isi-vista/adam/issues/498
    sized_tokens = (
        len(language_concept_alignment.language.as_token_sequence())
        if not ignore_determiners
        else len(
            [
                token
                for token in language_concept_alignment.language.as_token_sequence()
                if token not in ["a", "the"]
            ]
        )
    )

    # This assumes the slots and the non-slot elements are non-overlapping,
    # which is true for how we construct them.
    return num_covered_tokens == sized_tokens


class AlignmentSlots(Enum):
    """An argument is a slot for an object, and a fixed string is something we wish to learn"""

    Argument = auto()
    FixedString = auto()


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

        # or we have already aligned any of the tokens in between the objects
        # to some other meaning.
        for token_index in range(candidate_token_span.start, candidate_token_span.end):
            if language_concept_alignment.token_index_is_aligned(token_index):
                return False

        for span in invalid_token_spans:
            if candidate_token_span.contains_span(span):
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
        # So we keep track if the previous token was a FixedString indicator
        previous_node_was_string = False

        for token in candidate_template:
            # If the token in our template is an argument we need to assign it a
            # unique SyntaxSemanticsVariable, and map it to the SemanticNode
            if token == AlignmentSlots.Argument:
                slot_semantic_variable = STANDARD_SLOT_VARIABLES[aligned_node_index]
                template_elements.append(slot_semantic_variable)
                slot_to_semantic_node.append(
                    (slot_semantic_variable, aligned_nodes[aligned_node_index].node)
                )
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
                    # If our FixedString is flanked by two Arguments we just want to acquire all the tokens
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
                            yield None
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
            sum(1 for token in candidate_template if token == AlignmentSlots.Argument),
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

    return immutableset(
        bound_surface_template
        for bound_surface_template in ret
        # For now, we require templates to account for the entire utterance.
        # See https://github.com/isi-vista/adam/issues/789
        if covers_entire_utterance(bound_surface_template, language_concept_alignment)
    )
