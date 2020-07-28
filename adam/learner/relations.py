from abc import ABC

from typing import AbstractSet, Optional, Iterable, Tuple
import itertools
from adam.learner import LanguagePerceptionSemanticAlignment, PerceptionSemanticAlignment
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.subset import AbstractTemplateSubsetLearnerNew
from adam.learner.surface_templates import SurfaceTemplateBoundToSemanticNodes
from attr import attrs
from adam.learner.template_learner import AbstractTemplateLearnerNew
from adam.perception import MatchMode
from adam.perception.perception_graph import PerceptionGraph
from adam.semantics import RelationConcept, SemanticNode
from immutablecollections import immutableset, immutablesetmultidict
from adam.learner.learner_utils import candidate_templates, AlignmentSlots

_MAXIMUM_RELATION_TEMPLATE_TOKEN_LENGTH = 5


@attrs
class AbstractRelationTemplateLearnerNew(AbstractTemplateLearnerNew, ABC):
    # pylint:disable=abstract-method
    def _candidate_templates(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> AbstractSet[SurfaceTemplateBoundToSemanticNodes]:
        def candidate_relation_templates() -> Iterable[Tuple[AlignmentSlots, ...]]:
            # This function returns templates fro the candidate relation templates
            # terminology: (A)rgument - Noun, (F)ixedString - A collection or str tokens that can be a preposition or localiser/coverb, etc.

            # Now, handle two arguments with one function string (e.g. a ball on a table)
            for output in immutableset(
                itertools.permutations(
                    [
                        AlignmentSlots.Argument,
                        AlignmentSlots.Argument,
                        AlignmentSlots.FixedString,
                    ],
                    3,
                )
            ):
                yield output

            # Now, handle two arguments with two function strings (e.g. chyuou dzai zhouzi shang)
            for output in immutableset(
                itertools.permutations(
                    [
                        AlignmentSlots.Argument,
                        AlignmentSlots.Argument,
                        AlignmentSlots.FixedString,
                        AlignmentSlots.FixedString,
                    ],
                    4,
                )
            ):
                yield output

        return candidate_templates(
            language_perception_semantic_alignment,
            _MAXIMUM_RELATION_TEMPLATE_TOKEN_LENGTH,
            self._language_mode,
            candidate_relation_templates,
        )

    def _enrich_post_process(
        self,
        perception_graph_after_matching: PerceptionGraph,
        immutable_new_nodes: AbstractSet[SemanticNode],
    ) -> Tuple[PerceptionGraph, AbstractSet[SemanticNode]]:
        return perception_graph_after_matching, immutable_new_nodes


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
