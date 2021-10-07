import collections
import itertools
import typing
from abc import ABC
from typing import AbstractSet, Iterable, Optional, Tuple

from attr import attrs, attrib
from immutablecollections import immutableset, immutablesetmultidict

from adam.learner.alignments import (
    LanguagePerceptionSemanticAlignment,
    PerceptionSemanticAlignment,
)
from adam.learner.learner_utils import (
    candidate_templates,
    AlignmentSlots,
    pattern_remove_incomplete_region_or_spatial_path,
)
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.subset import AbstractTemplateSubsetLearner
from adam.learner.surface_templates import SurfaceTemplateBoundToSemanticNodes
from adam.learner.template_learner import AbstractTemplateLearner
from adam.ontology.phase2_ontology import TWO, HAS_COUNT, MANY
from adam.perception import MatchMode
from adam.perception.perception_graph import PerceptionGraph
from adam.semantics import AttributeConcept, SemanticNode, ObjectSemanticNode

_MAXIMUM_PLURAL_TEMPLATE_TOKEN_LENGTH = 5


@attrs
class AbstractPluralTemplateLearner(AbstractTemplateLearner, ABC):
    # pylint:disable=abstract-method
    def _candidate_templates(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> AbstractSet[SurfaceTemplateBoundToSemanticNodes]:
        def candidate_plural_templates() -> Iterable[Tuple[AlignmentSlots, ...]]:
            # This function returns templates for the candidate plural templates
            # Terminology:
            # (A)rgument - Noun
            # (F)ixedString - A collection of str tokens which can be the plural marker or modifier

            for i in [2, 3]:
                for output in immutableset(
                    itertools.permutations(
                        [AlignmentSlots.ARGUMENT]
                        + [AlignmentSlots.FIXEDSTRING] * (i - 1),
                        i,
                    )
                ):
                    yield output

        # Generate all the possible plural template alignments
        return candidate_templates(
            language_perception_semantic_alignment,
            _MAXIMUM_PLURAL_TEMPLATE_TOKEN_LENGTH,
            self._language_mode,
            candidate_plural_templates,
        )

    def _enrich_post_process(
        self,
        perception_graph_after_matching: PerceptionGraph,
        immutable_new_nodes: AbstractSet[SemanticNode],
    ) -> Tuple[PerceptionGraph, AbstractSet[SemanticNode]]:
        return perception_graph_after_matching, immutable_new_nodes


@attrs
class SubsetPluralLearner(AbstractTemplateSubsetLearner, AbstractPluralTemplateLearner):
    potential_plural_markers: typing.Counter[str] = attrib(
        init=False, default=collections.Counter()
    )

    def _can_learn_from(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> bool:
        concepts = [
            s.concept
            for s in language_perception_semantic_alignment.perception_semantic_alignment.semantic_nodes
        ]
        counts = collections.Counter(concepts)
        return max(counts.values()) > 1 if counts.values() else False

    def _preprocess_scene(
        self, perception_semantic_alignment: PerceptionSemanticAlignment
    ) -> PerceptionSemanticAlignment:
        nodes = [
            s
            for s in perception_semantic_alignment.semantic_nodes
            if isinstance(s, ObjectSemanticNode)
        ]
        counts = collections.Counter([s.concept for s in nodes])
        digraph = perception_semantic_alignment.perception_graph.copy_as_digraph()
        for node in nodes:
            count = counts[node.concept]
            if count > 1:
                if count == 2:
                    count_node = TWO
                else:
                    count_node = MANY
                digraph.add_node(count_node)
                digraph.add_edge(node, count_node, label=HAS_COUNT)
        graph_with_counts = PerceptionGraph(
            digraph, dynamic=perception_semantic_alignment.perception_graph.dynamic
        )
        return PerceptionSemanticAlignment(
            graph_with_counts, perception_semantic_alignment.semantic_nodes
        )

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
        # If we are keeping a hypothesis, use the template for that to update the list of possible plural markers.
        self.potential_plural_markers.update(
            [
                t
                for t in bound_surface_template.surface_template.elements
                if isinstance(t, str)
            ]
        )
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
