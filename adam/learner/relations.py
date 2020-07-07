from abc import ABC
from typing import AbstractSet, List, Union, Optional

from adam.learner import LanguagePerceptionSemanticAlignment, PerceptionSemanticAlignment
from adam.learner.learner_utils import covers_entire_utterance
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.subset import AbstractTemplateSubsetLearnerNew
from adam.learner.surface_templates import (
    SLOT1,
    SLOT2,
    SurfaceTemplate,
    SurfaceTemplateBoundToSemanticNodes,
)
from adam.learner.template_learner import AbstractTemplateLearnerNew
from adam.perception import MatchMode
from adam.semantics import RelationConcept, SyntaxSemanticsVariable
from attr import attrs
from immutablecollections import immutableset, immutablesetmultidict
from vistautils.span import Span

_MAXIMUM_RELATION_TEMPLATE_TOKEN_LENGTH = 3


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
        sentence_tokens = language_concept_alignment.language.as_token_sequence()

        # We currently hypothesize both pre-positions and post-positions,
        # so we look for two token spans aligned to object
        # and hypothesize that the token span between them might realize a relation.

        # TODO: refactor this with similar code in the verb equivalent of this class
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

        # Example: "a ball on a table"
        for (
            left_object_node,
            span_for_left_object,
        ) in language_concept_alignment.node_to_language_span.items():
            for (
                right_object_node,
                span_for_right_object,
            ) in language_concept_alignment.node_to_language_span.items():
                # Our code will be simpler if we can assume an ordering of the object aligned
                # tokens.
                if not span_for_left_object.precedes(span_for_right_object):
                    continue

                # If the two candidate argument spans are adjacent,
                # there are no tokens available to represent the relation.
                if span_for_left_object.end != span_for_right_object.start:
                    # example: span for "on"
                    candidate_relation_token_span = Span(
                        span_for_left_object.end, span_for_right_object.start
                    )
                    if is_legal_template_span(candidate_relation_token_span):
                        # Hack for Chinese: this gets a node one to the left of the start that may be a part of the preposition
                        leftone = None
                        # Check that the span is valid and that we don't already have a concept for this node
                        if span_for_left_object.start > 0 and Span(
                            span_for_left_object.start - 1, span_for_left_object.start
                        ) not in list(
                            language_concept_alignment.node_to_language_span.values()
                        ):
                            leftone = sentence_tokens[
                                span_for_left_object.start
                                - 1 : span_for_left_object.start
                            ]
                            # for English, we also want to be sure that it's not a determiner
                            if leftone[0] == "the" or leftone[0] == "a":
                                leftone = None

                        # we do the same for the right side of the relation
                        rightone = None
                        if span_for_right_object.end < len(sentence_tokens) and Span(
                            span_for_right_object.end, span_for_right_object.end + 1
                        ) not in list(
                            language_concept_alignment.node_to_language_span.values()
                        ):
                            rightone = sentence_tokens[
                                span_for_right_object.end : span_for_right_object.end + 1
                            ]
                            if rightone[0] == "a" or rightone[0] == "the":
                                rightone = None

                        template_elements: List[Union[SyntaxSemanticsVariable, str]] = []
                        if leftone:
                            template_elements.extend([leftone[0]])
                        template_elements.append(SLOT1)
                        template_elements.extend(
                            sentence_tokens[
                                candidate_relation_token_span.start : candidate_relation_token_span.end
                            ]
                        )
                        template_elements.append(SLOT2)
                        if rightone:
                            template_elements.extend([rightone[0]])
                        ret.append(
                            SurfaceTemplateBoundToSemanticNodes(
                                surface_template=SurfaceTemplate(
                                    elements=template_elements,
                                    determiner_prefix_slots=[SLOT1, SLOT2],
                                    language_mode=self._language_mode,
                                ),
                                slot_to_semantic_node=[
                                    (SLOT1, left_object_node),
                                    (SLOT2, right_object_node),
                                ],
                            )
                        )
        return immutableset(
            bound_surface_template
            for bound_surface_template in ret
            # For now, we require templates to account for the entire utterance.
            # See https://github.com/isi-vista/adam/issues/789
            if covers_entire_utterance(
                bound_surface_template,
                language_concept_alignment,
                token_sequence_count=len(sentence_tokens),
            )
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
