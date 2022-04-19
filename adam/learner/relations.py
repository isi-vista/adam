import logging
from adam.utils.networkx_utils import digraph_with_nodes_sorted_by

from attr.validators import instance_of, optional
from pathlib import Path
from networkx import all_shortest_paths

from abc import ABC
from adam.learner.pursuit import AbstractPursuitLearner

from typing import AbstractSet, Optional, Iterable, Tuple, Mapping, Sequence
import itertools
from adam.learner import (
    LanguagePerceptionSemanticAlignment,
    PerceptionSemanticAlignment,
    get_largest_matching_pattern,
)
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.subset import AbstractTemplateSubsetLearner
from adam.learner.surface_templates import (
    SurfaceTemplateBoundToSemanticNodes,
    SLOT1,
    SLOT2,
)
from attr import attrs, attrib
from adam.learner.template_learner import AbstractTemplateLearner
from adam.perception import MatchMode, ObjectPerception
from adam.perception.perception_graph import PerceptionGraph, _graph_node_order
from adam.semantics import (
    RelationConcept,
    SemanticNode,
    Concept,
    SyntaxSemanticsVariable,
    ObjectSemanticNode,
)
from immutablecollections import immutableset, immutablesetmultidict
from adam.learner.learner_utils import candidate_templates, AlignmentSlots

_MAXIMUM_RELATION_TEMPLATE_TOKEN_LENGTH = 5


@attrs
class AbstractRelationTemplateLearner(AbstractTemplateLearner, ABC):
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
                        AlignmentSlots.ARGUMENT,
                        AlignmentSlots.ARGUMENT,
                        AlignmentSlots.FIXEDSTRING,
                    ],
                    3,
                )
            ):
                yield output

            # Now, handle two arguments with two function strings (e.g. chyuou dzai zhouzi shang)
            for output in immutableset(
                itertools.permutations(
                    [
                        AlignmentSlots.ARGUMENT,
                        AlignmentSlots.ARGUMENT,
                        AlignmentSlots.FIXEDSTRING,
                        AlignmentSlots.FIXEDSTRING,
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
class SubsetRelationLearner(
    AbstractTemplateSubsetLearner, AbstractRelationTemplateLearner
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
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,  # pylint: disable=unused-argument
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
                    min_continuous_feature_match_score=self._min_continuous_feature_match_score,
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
        match = previous_pattern_hypothesis.intersection_getting_match(
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
        if match:
            match.confirm_match()
            return match.intersection
        # We don't need this, but Mypy wants it.
        return None


@attrs
class PursuitRelationLearner(AbstractPursuitLearner, AbstractRelationTemplateLearner):
    """
    An implementation of TemplateLearnerNew for the Pursuit learning algorithm over relations
    """

    def _find_identical_hypothesis(
        self,
        new_hypothesis: PerceptionGraphTemplate,
        candidates: Iterable[PerceptionGraphTemplate],
    ) -> Optional[PerceptionGraphTemplate]:
        for candidate in candidates:
            if new_hypothesis.graph_pattern.check_isomorphism(candidate.graph_pattern):
                return candidate
        return None

    @attrs(frozen=True)
    class RelationHypothesisPartialMatch(AbstractPursuitLearner.PartialMatch):
        partial_match_hypothesis: Optional[PerceptionGraphTemplate] = attrib(
            validator=optional(instance_of(PerceptionGraphTemplate))
        )
        num_nodes_matched: int = attrib(validator=instance_of(int), kw_only=True)
        num_nodes_in_pattern: int = attrib(validator=instance_of(int), kw_only=True)

        def matched_exactly(self) -> bool:
            return self.num_nodes_matched == self.num_nodes_in_pattern

        def match_score(self) -> float:
            return self.num_nodes_matched / self.num_nodes_in_pattern

    def _find_partial_match(
        self,
        hypothesis: PerceptionGraphTemplate,
        graph: PerceptionGraph,
        *,
        required_alignments: Mapping[SyntaxSemanticsVariable, ObjectSemanticNode],
    ) -> "AbstractPursuitLearner.PartialMatch":
        pattern = hypothesis.graph_pattern
        hypothesis_pattern_common_subgraph = get_largest_matching_pattern(
            pattern,
            graph,
            debug_callback=self._debug_callback,
            graph_logger=self._hypothesis_logger,
            ontology=self._ontology,
            match_mode=MatchMode.NON_OBJECT,
            allowed_matches=immutablesetmultidict(
                [
                    (hypothesis.template_variable_to_pattern_node[variable], object_node)
                    for variable, object_node in required_alignments.items()
                ]
            ),
        )
        self.debug_counter += 1

        leading_hypothesis_num_nodes = len(pattern)
        num_nodes_matched = (
            len(hypothesis_pattern_common_subgraph.copy_as_digraph().nodes)
            if hypothesis_pattern_common_subgraph
            else 0
        )
        if hypothesis_pattern_common_subgraph:
            partial_hypothesis: Optional[
                PerceptionGraphTemplate
            ] = PerceptionGraphTemplate(
                graph_pattern=hypothesis_pattern_common_subgraph,
                template_variable_to_pattern_node=hypothesis.template_variable_to_pattern_node,
            )
        else:
            partial_hypothesis = None

        return PursuitRelationLearner.RelationHypothesisPartialMatch(
            partial_hypothesis,
            num_nodes_matched=num_nodes_matched,
            num_nodes_in_pattern=leading_hypothesis_num_nodes,
        )

    def _new_concept(self, debug_string: str) -> Concept:
        return RelationConcept(debug_string)

    def _keep_hypothesis(
        self,
        *,
        hypothesis: PerceptionGraphTemplate,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,
    ) -> bool:
        if len(hypothesis.graph_pattern) < 3:
            # We need at least three nodes - two wildcards and a relation type -
            # for meaningful relation semantics.
            return False
        if len(bound_surface_template.slot_to_semantic_node.keys()) < 2:
            # We must have at least two wildcard slots
            return False
        return True

    def _hypotheses_from_perception(
        self,
        learning_state: LanguagePerceptionSemanticAlignment,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,
    ) -> AbstractSet[PerceptionGraphTemplate]:
        return immutableset(
            PerceptionGraphTemplate.from_graph(
                perception_graph=candidate_relation_meaning,
                template_variable_to_matched_object_node=bound_surface_template.slot_to_semantic_node,
                min_continuous_feature_match_score=self._min_continuous_feature_match_score,
            )
            for candidate_relation_meaning in _extract_candidate_relations(
                learning_state.perception_semantic_alignment.perception_graph,
                bound_surface_template.slot_to_semantic_node[SLOT1],
                bound_surface_template.slot_to_semantic_node[SLOT2],
            )
        )

    def _can_learn_from(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> bool:
        # We can only learn relations if there are aligned object semantic nodes,
        # and for there to be any aligned semantic nodes at all
        # there must be aligned object semantic nodes,
        # since relations, verbs, etc. build on object semantic nodes.
        # Furthermore we only need two nodes, because relations only involve two objects.
        return (
            len(
                language_perception_semantic_alignment.perception_semantic_alignment.semantic_nodes
            )
            > 2
        )

    def _preprocess_scene(
        self, perception_semantic_alignment: PerceptionSemanticAlignment
    ) -> PerceptionSemanticAlignment:
        return perception_semantic_alignment

    def log_hypotheses(self, log_output_path: Path) -> None:
        logging.info(
            "Logging %s hypotheses to %s",
            len(self._concept_to_hypotheses_and_scores),
            log_output_path,
        )
        for (
            concept,
            hypotheses_to_scores,
        ) in self._concept_to_hypotheses_and_scores.items():
            for (i, (hypothesis, score)) in enumerate(hypotheses_to_scores.items()):
                hypothesis.render_to_file(
                    concept.debug_string,
                    log_output_path / f"{concept.debug_string}.{i}.score_{score}",
                )

        logging.info(
            "Logging %s lexicon items to %s", len(self._lexicon), log_output_path
        )
        for (
            surface_template,
            meaning_as_perception_graph_template,
        ) in self._lexicon.items():
            meaning_as_perception_graph_template.render_to_file(
                surface_template.to_short_string(),
                log_output_path / f"lexicon-{surface_template.to_short_string()}",
            )


def _extract_candidate_relations(
    whole_scene_perception_graph: PerceptionGraph,
    relation_object_1: ObjectSemanticNode,
    relation_object_2: ObjectSemanticNode,
) -> Sequence[PerceptionGraph]:
    # The directions of edges in the perception graph are not necessarily meaningful
    # from the point-of-view of hypothesis generation, so we need an undirected copy
    # of the graph.
    perception_digraph = whole_scene_perception_graph.copy_as_digraph()
    perception_graph_undirected = perception_digraph.to_undirected(
        # as_view=True loses determinism
        as_view=False
    )

    output_graphs = []

    # The core of our hypothesis for the semantics of a preposition is all nodes
    # along the shortest path between the two objects involved in the perception graph.
    for hypothesis_spine_nodes in all_shortest_paths(
        perception_graph_undirected, relation_object_2, relation_object_1
    ):
        # Along the core of our hypothesis we also want to collect the predecessors and successors
        hypothesis_nodes_mutable = []
        for node in hypothesis_spine_nodes:
            if node not in {relation_object_1, relation_object_2}:
                for successor in perception_digraph.successors(node):
                    if not (
                        isinstance(successor, ObjectPerception)
                        or isinstance(successor, ObjectSemanticNode)
                    ):
                        hypothesis_nodes_mutable.append(successor)
                for predecessor in perception_digraph.predecessors(node):
                    if not (
                        isinstance(predecessor, ObjectPerception)
                        or isinstance(predecessor, ObjectSemanticNode)
                    ):
                        hypothesis_nodes_mutable.append(predecessor)

        hypothesis_nodes_mutable.extend(hypothesis_spine_nodes)

        # We wrap the nodes in an immutable set to remove duplicates
        # while preserving iteration determinism.
        hypothesis_nodes = immutableset(hypothesis_nodes_mutable)

        output_graphs.append(
            PerceptionGraph(
                digraph_with_nodes_sorted_by(
                    perception_digraph.subgraph(hypothesis_nodes), _graph_node_order
                )
            )
        )

    return output_graphs
