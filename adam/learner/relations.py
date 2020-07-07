from abc import ABC
from typing import AbstractSet, List, Union, Optional, Iterable, Tuple, Dict
from immutablecollections import ImmutableSet
import itertools
from adam.learner import LanguagePerceptionSemanticAlignment, PerceptionSemanticAlignment
from adam.learner.learner_utils import covers_entire_utterance
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.subset import AbstractTemplateSubsetLearnerNew
from adam.learner.surface_templates import (
    STANDARD_SLOT_VARIABLES,
    SurfaceTemplate,
    SurfaceTemplateBoundToSemanticNodes,
)
from attr import attrib, attrs
from enum import Enum, auto
from adam.learner.template_learner import AbstractTemplateLearnerNew
from adam.perception import MatchMode
from adam.semantics import RelationConcept, SyntaxSemanticsVariable
from immutablecollections import immutableset, immutablesetmultidict
from vistautils.span import Span
from attr.validators import instance_of
from adam.semantics import SemanticNode

_MAXIMUM_RELATION_TEMPLATE_TOKEN_LENGTH = 3


class RelationAlignmentSlots(Enum):
    Argument = auto()
    FixedString = auto()


@attrs(frozen=True, slots=True)
class SemanticNodeWithSpan:
    node: SemanticNode = attrib(validator=instance_of(SemanticNode))
    span: Span = attrib(validator=instance_of(Span))


@attrs
class AbstractRelationTemplateLearnerNew(AbstractTemplateLearnerNew, ABC):
    # pylint:disable=abstract-method
    def _candidate_templates(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
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

        # We currently hypothesize both pre-positions and post-positions,
        # so we look for two token spans aligned to object
        # and hypothesize that the token span between them might realize a relation.

        def is_legal_template_span(candidate_relation_token_span: Span) -> bool:
            # A template token span can't exceed the bounds of the utterance
            if candidate_relation_token_span.start < 0:
                return False
            if candidate_relation_token_span.end > len(sentence_tokens):
                return False
            # or be bigger than our maximum template size...
            if (
                len(candidate_relation_token_span)
                > _MAXIMUM_RELATION_TEMPLATE_TOKEN_LENGTH
            ):
                return False

            # or we have already aligned any of the tokens in between the objects
            # to some other meaning.
            for token_index in range(
                candidate_relation_token_span.start, candidate_relation_token_span.end
            ):
                if language_concept_alignment.token_index_is_aligned(token_index):
                    return False
            return True

        def candidate_relation_templates() -> Iterable[
            Tuple[RelationAlignmentSlots, ...]
        ]:
            # This function returns templates fro the candidate relation templates
            # terminology: (A)rgument - Noun, (F)ixedString - A collection or str tokens that can be a preposition or localiser/coverb, etc.

            # Now, handle two arguments with one function string (e.g. a ball on a table)
            for output in immutableset(
                itertools.permutations(
                    [
                        RelationAlignmentSlots.Argument,
                        RelationAlignmentSlots.Argument,
                        RelationAlignmentSlots.FixedString,
                    ],
                    3,
                )
            ):
                yield output

            # Now, handle two arguments with two function strings (e.g. chyuou dzai zhouzi shang)
            for output in immutableset(
                itertools.permutations(
                    [
                        RelationAlignmentSlots.Argument,
                        RelationAlignmentSlots.Argument,
                        RelationAlignmentSlots.FixedString,
                        RelationAlignmentSlots.FixedString,
                    ],
                    4,
                )
            ):
                yield output

        def in_left_to_right_order(
            semantic_nodes: Tuple[SemanticNodeWithSpan, ...]
        ) -> bool:
            previous_node = semantic_nodes[0]
            for i in range(1, len(semantic_nodes)):
                if not previous_node.span.precedes(semantic_nodes[i].span):
                    return False
                previous_node = semantic_nodes[i]
            return True

        def aligned_object_nodes(
            num_arguments: int
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
            verb_template: Tuple[RelationAlignmentSlots, ...],
            aligned_nodes: Tuple[SemanticNodeWithSpan, ...],
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

            for token in verb_template:
                # if the token in our template is an argument we need to assign it a unique SyntaxSemanticsVariable and map in to the SemanticNode
                if token == RelationAlignmentSlots.Argument:
                    slot_semantic_variable = STANDARD_SLOT_VARIABLES[aligned_node_index]
                    template_elements.append(slot_semantic_variable)
                    slot_to_semantic_node.append(
                        (slot_semantic_variable, aligned_nodes[aligned_node_index].node)
                    )
                    aligned_node_index += 1
                    previous_node_was_string = False
                else:
                    if previous_node_was_string:  # this processes AFFA like AFA
                        continue
                    elif aligned_node_index == 0:
                        prefix_string_end = aligned_nodes[aligned_node_index].span.start
                    elif aligned_node_index == len(aligned_nodes):
                        postfix_string_start = aligned_nodes[
                            aligned_node_index - 1
                        ].span.end
                    else:
                        # If our FixedString is flanked by two Arguments, we just want to learn everything between them
                        if (
                            aligned_nodes[aligned_node_index - 1].span.end
                            != aligned_nodes[aligned_node_index].span.start
                        ):
                            candidate_verb_token_span = Span(
                                aligned_nodes[aligned_node_index - 1].span.end,
                                aligned_nodes[aligned_node_index].span.start,
                            )
                            if not is_legal_template_span(candidate_verb_token_span):
                                yield None
                            template_elements.extend(
                                sentence_tokens[
                                    candidate_verb_token_span.start : candidate_verb_token_span.end
                                ]
                            )
                        previous_node_was_string = True

            # We need to handle searching before or after the aligned token
            # And we could generate multiple options of different lengths
            # between 1 and _MAXIMUM_ACTION_TEMPLATE_TOKEN_LENGTH
            if prefix_string_end and postfix_string_start:
                for max_token_length_for_template_prefix in range(
                    1, _MAXIMUM_RELATION_TEMPLATE_TOKEN_LENGTH + 1
                ):
                    prefix_candidate_verb_token_span = Span(
                        prefix_string_end - max_token_length_for_template_prefix,
                        prefix_string_end,
                    )
                    if is_legal_template_span(prefix_candidate_verb_token_span):
                        for max_token_length_for_template_postfix in range(
                            1, _MAXIMUM_RELATION_TEMPLATE_TOKEN_LENGTH + 1
                        ):
                            postfix_candidate_verb_token_span = Span(
                                postfix_string_start,
                                postfix_string_start
                                + max_token_length_for_template_postfix,
                            )
                            if is_legal_template_span(postfix_candidate_verb_token_span):
                                final_template_elements: List[
                                    Union[str, SyntaxSemanticsVariable]
                                ] = list(
                                    sentence_tokens[
                                        prefix_candidate_verb_token_span.start : prefix_candidate_verb_token_span.end
                                    ]
                                )
                                final_template_elements.extend(template_elements)
                                final_template_elements.extend(
                                    sentence_tokens[
                                        postfix_candidate_verb_token_span.start : postfix_candidate_verb_token_span.end
                                    ]
                                )
                                yield SurfaceTemplateBoundToSemanticNodes(
                                    surface_template=SurfaceTemplate(
                                        elements=final_template_elements,
                                        determiner_prefix_slots=[
                                            SLOT for (SLOT, _) in slot_to_semantic_node
                                        ],
                                        language_mode=self._language_mode,
                                    ),
                                    slot_to_semantic_node=slot_to_semantic_node,
                                )
            elif prefix_string_end:
                for max_token_length_for_template_prefix in range(
                    1, _MAXIMUM_RELATION_TEMPLATE_TOKEN_LENGTH + 1
                ):
                    prefix_candidate_verb_token_span = Span(
                        prefix_string_end - max_token_length_for_template_prefix,
                        prefix_string_end,
                    )
                    if is_legal_template_span(prefix_candidate_verb_token_span):
                        final_template_elements = list(
                            sentence_tokens[
                                prefix_candidate_verb_token_span.start : prefix_candidate_verb_token_span.end
                            ]
                        )
                        final_template_elements.extend(template_elements)
                        yield SurfaceTemplateBoundToSemanticNodes(
                            surface_template=SurfaceTemplate(
                                elements=final_template_elements,
                                determiner_prefix_slots=[
                                    SLOT for (SLOT, _) in slot_to_semantic_node
                                ],
                                language_mode=self._language_mode,
                            ),
                            slot_to_semantic_node=slot_to_semantic_node,
                        )
            elif postfix_string_start:
                for max_token_length_for_template_postfix in range(
                    1, _MAXIMUM_RELATION_TEMPLATE_TOKEN_LENGTH + 1
                ):
                    postfix_candidate_verb_token_span = Span(
                        postfix_string_start,
                        postfix_string_start + max_token_length_for_template_postfix,
                    )
                    if is_legal_template_span(postfix_candidate_verb_token_span):
                        final_template_elements = list(template_elements)
                        final_template_elements.extend(
                            sentence_tokens[
                                postfix_candidate_verb_token_span.start : postfix_candidate_verb_token_span.end
                            ]
                        )
                        yield SurfaceTemplateBoundToSemanticNodes(
                            surface_template=SurfaceTemplate(
                                elements=final_template_elements,
                                determiner_prefix_slots=[
                                    SLOT for (SLOT, _) in slot_to_semantic_node
                                ],
                                language_mode=self._language_mode,
                            ),
                            slot_to_semantic_node=slot_to_semantic_node,
                        )
            else:
                yield SurfaceTemplateBoundToSemanticNodes(
                    surface_template=SurfaceTemplate(
                        elements=template_elements,
                        determiner_prefix_slots=[
                            SLOT for (SLOT, _) in slot_to_semantic_node
                        ],
                        language_mode=self._language_mode,
                    ),
                    slot_to_semantic_node=slot_to_semantic_node,
                )

        for relation_template in candidate_relation_templates():
            for aligned_nodes in aligned_object_nodes(
                sum(
                    1
                    for token in relation_template
                    if token == RelationAlignmentSlots.Argument
                )
            ):
                for (
                    surface_template_bound_to_semantic_nodes
                ) in process_aligned_objects_with_template(
                    relation_template, aligned_nodes
                ):
                    if surface_template_bound_to_semantic_nodes:
                        print(surface_template_bound_to_semantic_nodes)
                        ret.append(surface_template_bound_to_semantic_nodes)

        return immutableset(
            bound_surface_template
            for bound_surface_template in ret
            # For now, we require templates to account for the entire utterance.
            # See https://github.com/isi-vista/adam/issues/789
            if covers_entire_utterance(bound_surface_template, language_concept_alignment)
        )


@attrs
class SubsetRelationLearnerNew(
    AbstractTemplateSubsetLearnerNew, AbstractRelationTemplateLearnerNew
):
    def _can_learn_from(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> bool:
        # You need two objects to have a relation.
        return (
            len(
                language_perception_semantic_alignment.perception_semantic_alignment.semantic_nodes
            )
            > 1
        )

    def _new_concept(self, debug_string: str) -> RelationConcept:
        return RelationConcept(debug_string)

    def _keep_hypothesis(
        self,
        *,
        hypothesis: PerceptionGraphTemplate,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes  # pylint: disable=unused-argument
    ) -> bool:
        return len(hypothesis.graph_pattern) >= 2

    def _hypotheses_from_perception(
        self,
        learning_state: LanguagePerceptionSemanticAlignment,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,
    ) -> AbstractSet[PerceptionGraphTemplate]:
        # For the subset learner, our hypothesis is the entire graph.
        return immutableset(
            [
                PerceptionGraphTemplate.from_graph(
                    learning_state.perception_semantic_alignment.perception_graph,
                    template_variable_to_matched_object_node=bound_surface_template.slot_to_semantic_node,
                )
            ]
        )

    def _preprocess_scene(
        self, perception_semantic_alignment: PerceptionSemanticAlignment
    ) -> PerceptionSemanticAlignment:
        return perception_semantic_alignment

    def _update_hypothesis(
        self,
        previous_pattern_hypothesis: PerceptionGraphTemplate,
        current_pattern_hypothesis: PerceptionGraphTemplate,
    ) -> Optional[PerceptionGraphTemplate]:
        return previous_pattern_hypothesis.intersection(
            current_pattern_hypothesis,
            ontology=self._ontology,
            match_mode=MatchMode.NON_OBJECT,
            allowed_matches=immutablesetmultidict(
                [
                    (node2, node1)
                    for previous_slot, node1 in previous_pattern_hypothesis.template_variable_to_pattern_node.items()
                    for new_slot, node2 in current_pattern_hypothesis.template_variable_to_pattern_node.items()
                    if previous_slot == new_slot
                ]
            ),
        )
