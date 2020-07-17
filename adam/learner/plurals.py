import itertools
from abc import ABC
from collections import Counter
from typing import AbstractSet, Iterable, Optional, Tuple

from attr import attrs
from immutablecollections import immutableset, immutablesetmultidict

from adam.learner.alignments import (
    LanguagePerceptionSemanticAlignment,
    PerceptionSemanticAlignment,
)
from adam.learner.attributes import SubsetAttributeLearnerNew
from adam.learner.learner_utils import candidate_templates, AlignmentSlots, \
    pattern_remove_incomplete_region_or_spatial_path
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.subset import (
    AbstractTemplateSubsetLearnerNew,
)
from adam.learner.surface_templates import (
    SurfaceTemplateBoundToSemanticNodes,
)
from adam.learner.template_learner import (
    AbstractTemplateLearnerNew,
)
from adam.ontology import OntologyNode
from adam.perception import MatchMode
from adam.perception.perception_graph import PerceptionGraph
from adam.semantics import ActionConcept, ObjectConcept, AttributeConcept

_MAXIMUM_PLURAL_TEMPLATE_TOKEN_LENGTH = 5

#
# @attrs
# class AbstractPluralTemplateLearnerNew(AbstractTemplateLearnerNew, ABC):
#     # pylint:disable=abstract-method
#     def _candidate_templates(
#             self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
#     ) -> AbstractSet[SurfaceTemplateBoundToSemanticNodes]:
#         def candidate_plural_templates() -> Iterable[Tuple[AlignmentSlots, ...]]:
#             # This function returns templates for the candidate plural templates
#             # Terminology:
#             # (A)rgument - Noun
#             # (F)ixedString - A collection of str tokens which can be the plural marker or modifier
#
#             for i in [2, 3]:
#                 for output in immutableset(
#                         itertools.permutations(
#                             [AlignmentSlots.Argument] + [AlignmentSlots.FixedString] * (i-1), i
#                         )
#                 ):
#                     yield output
#
#         # Generate all the possible plural template alignments
#         return candidate_templates(
#             language_perception_semantic_alignment,
#             _MAXIMUM_PLURAL_TEMPLATE_TOKEN_LENGTH,
#             self._language_mode,
#             candidate_plural_templates,
#         )
#
#
# @attrs
# class SubsetPluralLearnerNew(AbstractTemplateSubsetLearnerNew, AbstractPluralTemplateLearnerNew):
#     def _can_learn_from(
#         self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
#     ) -> bool:
#         concepts = [s.concept for s in
#                     language_perception_semantic_alignment.perception_semantic_alignment.semantic_nodes]
#         counts = Counter(concepts)
#         return max(counts.values()) > 1
#
#     def _new_concept(self, debug_string: str) -> AttributeConcept:
#         return AttributeConcept(debug_string)
#
#     def _keep_hypothesis(
#         self,
#         *,
#         hypothesis: PerceptionGraphTemplate,
#         bound_surface_template: SurfaceTemplateBoundToSemanticNodes
#     ) -> bool:
#         num_template_arguments = len(bound_surface_template.slot_to_semantic_node)
#         return len(hypothesis.graph_pattern) >= 2 * num_template_arguments
#
#     def _hypotheses_from_perception(
#         self,
#         learning_state: LanguagePerceptionSemanticAlignment,
#         bound_surface_template: SurfaceTemplateBoundToSemanticNodes,
#     ) -> AbstractSet[PerceptionGraphTemplate]:
#         # For the subset learner, our hypothesis is the entire graph.
#         return immutableset(
#             [
#                 PerceptionGraphTemplate.from_graph(
#                     learning_state.perception_semantic_alignment.perception_graph,
#                     template_variable_to_matched_object_node=bound_surface_template.slot_to_semantic_node,
#                 )
#             ]
#         )
#
#     def _preprocess_scene(
#         self, perception_semantic_alignment: PerceptionSemanticAlignment
#     ) -> PerceptionSemanticAlignment:
#         return perception_semantic_alignment
#
#     def _update_hypothesis(
#         self,
#         previous_pattern_hypothesis: PerceptionGraphTemplate,
#         current_pattern_hypothesis: PerceptionGraphTemplate,
#     ) -> Optional[PerceptionGraphTemplate]:
#         return previous_pattern_hypothesis.intersection(
#             current_pattern_hypothesis,
#             ontology=self._ontology,
#             match_mode=MatchMode.NON_OBJECT,
#             allowed_matches=immutablesetmultidict(
#                 [
#                     (node2, node1)
#                     for previous_slot, node1 in previous_pattern_hypothesis.template_variable_to_pattern_node.items()
#                     for new_slot, node2 in current_pattern_hypothesis.template_variable_to_pattern_node.items()
#                     if previous_slot == new_slot
#                 ]
#             ),
#         )

@attrs
class SubsetPluralLearnerNew(SubsetAttributeLearnerNew):
    template_to_count_property = {}

    def _can_learn_from(
            self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> bool:
        concepts = [s.concept for s in
                    language_perception_semantic_alignment.perception_semantic_alignment.semantic_nodes]
        counts = Counter(concepts)
        return max(counts.values()) > 1

    def _preprocess_scene(
        self, perception_semantic_alignment: PerceptionSemanticAlignment
    ) -> PerceptionSemanticAlignment:
        nodes = [s for s in perception_semantic_alignment.semantic_nodes]
        counts = Counter([s.concept for s in nodes])
        print(counts)
        digraph = perception_semantic_alignment.perception_graph.copy_as_digraph()
        for node in nodes:
            count = counts[node.concept]
            if count > 1:
                count_node = OntologyNode(str(count))
                digraph.add_node(count_node)
                digraph.add_edge(node, count_node, label=OntologyNode("count"))
        graph_with_counts = PerceptionGraph(digraph, dynamic=perception_semantic_alignment.perception_graph.dynamic)
        return PerceptionSemanticAlignment(graph_with_counts, perception_semantic_alignment.semantic_nodes)

    def _new_concept(self, debug_string: str) -> AttributeConcept:
        return AttributeConcept(debug_string)

    def _hypotheses_from_perception(
        self,
        learning_state: LanguagePerceptionSemanticAlignment,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,
    ) -> AbstractSet[PerceptionGraphTemplate]:
        # This makes a hypothesis for the whole graph, with the wildcard slot
        # at each recognized object.
        return immutableset(
            [
                PerceptionGraphTemplate.from_graph(
                    learning_state.perception_semantic_alignment.perception_graph,
                    template_variable_to_matched_object_node=bound_surface_template.slot_to_semantic_node,
                )
            ]
        )

    def _keep_hypothesis(
        self,
        *,
        hypothesis: PerceptionGraphTemplate,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,  # pylint:disable=unused-argument
    ) -> bool:
        if len(hypothesis.graph_pattern) < 2:
            # We need at least two nodes - a wildcard and a property -
            # for meaningful attribute semantics.
            return False
        return True

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
            trim_after_match=pattern_remove_incomplete_region_or_spatial_path,
        )
