import itertools
import logging
from abc import ABC
from pathlib import Path
from typing import AbstractSet, Iterable, Optional, Tuple

from attr import attrs
from immutablecollections import (
    immutabledict,
    immutableset,
)

from adam.learner import (
    get_largest_matching_pattern,
    graph_without_learner,
)
from adam.learner.alignments import (
    LanguagePerceptionSemanticAlignment,
    PerceptionSemanticAlignment,
)
from adam.learner.learner_utils import AlignmentSlots, candidate_templates
from adam.learner.object_recognizer import (
    extract_candidate_objects,
)
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.pursuit import (
    AbstractPursuitLearnerNew,
)
from adam.learner.surface_templates import (
    SurfaceTemplate,
    SurfaceTemplateBoundToSemanticNodes,
)
from adam.learner.template_learner import (
    AbstractTemplateLearnerNew,
)
from adam.perception import ObjectPerception, MatchMode
from adam.perception.perception_graph import PerceptionGraph, PerceptionGraphPattern
from adam.semantics import ObjectConcept, GenericConcept

_MAXIMUM_GENERICS_TEMPLATE_TOKEN_LENGTH = 5


@attrs
class AbstractGenericsTemplateLearnerNew(AbstractTemplateLearnerNew, ABC):
    def _candidate_templates(
            self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> AbstractSet[SurfaceTemplateBoundToSemanticNodes]:
        def candidate_generics_templates() -> Iterable[Tuple[AlignmentSlots, ...]]:
            # This function returns templates for the candidate plural templates
            # Terminology:
            # (A)rgument - Noun
            # (F)ixedString - A collection of str tokens which can be the plural marker or modifier
            for i in [2, 3, 4]:
                for output in immutableset(
                        itertools.permutations(
                            [AlignmentSlots.Argument]
                            + [AlignmentSlots.FixedString] * (i - 1),
                            i,
                        )
                ):
                    yield output

        # Generate all the possible plural template alignments
        return candidate_templates(
            language_perception_semantic_alignment,
            _MAXIMUM_GENERICS_TEMPLATE_TOKEN_LENGTH,
            self._language_mode,
            candidate_generics_templates,
        )


@attrs
class PursuitGenericsLearnerNew(
    AbstractPursuitLearnerNew, AbstractGenericsTemplateLearnerNew
):
    """
    An implementation of pursuit learner for generics
    """

    def _can_learn_from(
            self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> bool:
        # The plural marker signals a generic when we hear a word we know, but the sentence doesn't match the scene
        recognized_concepts = language_perception_semantic_alignment. \
            language_concept_alignment.node_to_language_span.keys()
        # Check if the language is plural - TODO:
        is_plural = True  # language_perception_semantic_alignment.language_concept_alignment.language
        return (
                not language_perception_semantic_alignment.language_concept_alignment.is_entirely_aligned
                and is_plural and recognized_concepts
        )

    def _preprocess_scene(
            self, perception_semantic_alignment: PerceptionSemanticAlignment
    ) -> PerceptionSemanticAlignment:
        return perception_semantic_alignment

    def _new_concept(self, debug_string: str) -> GenericConcept:
        return GenericConcept(debug_string)

    def _hypotheses_from_perception(
            self,
            learning_state: LanguagePerceptionSemanticAlignment,
            bound_surface_template: SurfaceTemplateBoundToSemanticNodes,
    ) -> AbstractSet[PerceptionGraphTemplate]:
        if bound_surface_template.slot_to_semantic_node:
            raise RuntimeError(
                "Object learner should not have slot to semantic node alignments!"
            )

        return immutableset(
            PerceptionGraphTemplate(
                graph_pattern=PerceptionGraphPattern.from_graph(
                    candidate_object
                ).perception_graph_pattern,
                template_variable_to_pattern_node=immutabledict(),
            )
            for candidate_object in extract_candidate_objects(
                learning_state.perception_semantic_alignment.perception_graph
            )
        )

    # I can't spot the difference in arguments pylint claims?
    def _keep_hypothesis(  # pylint: disable=arguments-differ
            self,
            hypothesis: PerceptionGraphTemplate,
            bound_surface_template: SurfaceTemplateBoundToSemanticNodes,  # pylint:disable=unused-argument
    ) -> bool:
        if len(hypothesis.graph_pattern) < 2:
            # A one node graph is to small to meaningfully describe an object
            return False
        if all(isinstance(node, ObjectPerception) for node in hypothesis.graph_pattern):
            # A hypothesis which consists of just sub-object structure
            # with no other content is insufficiently distinctive.
            return False
        return True

    def _find_partial_match(
            self, hypothesis: PerceptionGraphTemplate, graph: PerceptionGraph
    ) -> "ObjectPursuitLearner.ObjectHypothesisPartialMatch":
        pattern = hypothesis.graph_pattern
        hypothesis_pattern_common_subgraph = get_largest_matching_pattern(
            pattern,
            graph,
            debug_callback=self._debug_callback,
            graph_logger=self._hypothesis_logger,
            ontology=self._ontology,
            match_mode=MatchMode.OBJECT,
        )
        self.debug_counter += 1

        leading_hypothesis_num_nodes = len(pattern)
        num_nodes_matched = (
            len(hypothesis_pattern_common_subgraph.copy_as_digraph().nodes)
            if hypothesis_pattern_common_subgraph
            else 0
        )

        return ObjectPursuitLearner.ObjectHypothesisPartialMatch(
            PerceptionGraphTemplate(graph_pattern=hypothesis_pattern_common_subgraph)
            if hypothesis_pattern_common_subgraph
            else None,
            num_nodes_matched=num_nodes_matched,
            num_nodes_in_pattern=leading_hypothesis_num_nodes,
        )

    def _find_identical_hypothesis(
            self,
            new_hypothesis: PerceptionGraphTemplate,
            candidates: Iterable[PerceptionGraphTemplate],
    ) -> Optional[PerceptionGraphTemplate]:
        for candidate in candidates:
            if new_hypothesis.graph_pattern.check_isomorphism(candidate.graph_pattern):
                return candidate
        return None

    # pylint:disable=abstract-method
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
            for (i, hypothesis) in enumerate(hypotheses_to_scores.keys()):
                hypothesis.render_to_file(
                    concept.debug_string, log_output_path / f"{concept.debug_string}.{i}"
                )
