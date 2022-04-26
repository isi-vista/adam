import logging
from abc import ABC, abstractmethod

from typing import AbstractSet, Iterable, List, Tuple, cast, Set

import networkx
from attr import attrib, attrs, evolve
from attr.validators import instance_of
from immutablecollections import immutabledict, immutableset
from more_itertools import one
from vistautils.preconditions import check_state

from adam.learner import ComposableLearner
from adam.learner.alignments import (
    LanguagePerceptionSemanticAlignment,
    PerceptionSemanticAlignment,
)
from adam.learner.language_mode import LanguageMode
from adam.learner.object_recognizer import (
    _get_root_object_perception,
    replace_match_with_object_graph_node,
)
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.surface_templates import (
    SurfaceTemplate,
    SurfaceTemplateBoundToSemanticNodes,
)
from adam.perception import ObjectPerception
from adam.perception.perception_graph import (
    PerceptionGraph,
    PerceptionGraphPatternMatch,
    ENTIRE_SCENE,
)
from adam.semantics import (
    Concept,
    ObjectConcept,
    ObjectSemanticNode,
    SemanticNode,
    FunctionalObjectConcept,
)


class TemplateLearner(ComposableLearner, ABC):
    _language_mode: LanguageMode = attrib(validator=instance_of(LanguageMode))

    @abstractmethod
    def templates_for_concept(self, concept: Concept) -> AbstractSet[SurfaceTemplate]:
        pass


@attrs
class AbstractTemplateLearner(TemplateLearner, ABC):
    """
    Super-class for learners using template-based syntax-semantics mappings.
    """

    _observation_num: int = attrib(init=False, default=0)
    _language_mode: LanguageMode = attrib(validator=instance_of(LanguageMode))
    _min_continuous_feature_match_score: float = attrib(
        validator=instance_of(float), kw_only=True
    )

    def learn_from(
        self,
        language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment,
        offset: int = 0,
    ) -> None:
        logging.info(
            "Observation %s: %s",
            self._observation_num + offset,
            language_perception_semantic_alignment.language_concept_alignment.language.as_token_string(),
        )

        self._observation_num += 1

        if not self._can_learn_from(language_perception_semantic_alignment):
            return

        # Pre-processing steps will be different depending on
        # what sort of structures we are running.
        preprocessed_input = evolve(
            language_perception_semantic_alignment,
            perception_semantic_alignment=self._preprocess_scene(
                language_perception_semantic_alignment.perception_semantic_alignment
            ),
        )

        self._pre_learning_step(preprocessed_input)

        for thing_whose_meaning_to_learn in self._candidate_templates(
            language_perception_semantic_alignment
        ):
            self._learning_step(preprocessed_input, thing_whose_meaning_to_learn)

        self._post_learning_step(preprocessed_input)

    def enrich_during_learning(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> LanguagePerceptionSemanticAlignment:
        (
            perception_post_enrichment,
            newly_recognized_semantic_nodes,
        ) = self._enrich_common(
            language_perception_semantic_alignment.perception_semantic_alignment
        )
        # we try to learn from the given instance
        return LanguagePerceptionSemanticAlignment(
            # We need to link the things we found to the language
            # so later learning stages can (a) know they are already covered
            # and (b) use this information in forming the surface templates.
            language_concept_alignment=language_perception_semantic_alignment.language_concept_alignment.copy_with_new_nodes(
                immutabledict(
                    # TODO: we currently handle only one template per concept
                    (
                        semantic_node,
                        one(self.templates_for_concept(semantic_node.concept)),
                    )
                    for semantic_node in newly_recognized_semantic_nodes
                    # We make an exception for a specific type of ObjectConcept which
                    # Indicates that we know this root is an object but we don't know
                    # how to refer to it linguisticly
                    if not isinstance(semantic_node.concept, FunctionalObjectConcept)
                ),
                filter_out_duplicate_alignments=True,
                # It's okay if we recognize objects we know how to describe,
                # but they just happen not to be mentioned in the linguistic description.
                fail_if_surface_templates_do_not_match_language=False,
            ),
            perception_semantic_alignment=perception_post_enrichment,
        )

    def enrich_during_description(
        self, perception_semantic_alignment: PerceptionSemanticAlignment
    ) -> PerceptionSemanticAlignment:
        # The other information returned by _enrich_common is only needed by
        # enrich_during_learning.
        return self._enrich_common(perception_semantic_alignment)[0]

    @abstractmethod
    def _enrich_post_process(
        self,
        perception_graph_after_matching: PerceptionGraph,
        immutable_new_nodes: AbstractSet[SemanticNode],
    ) -> Tuple[PerceptionGraph, AbstractSet[SemanticNode]]:
        """
        Allows a learner to do specific enrichment post-processing if needed
        """

    @abstractmethod
    def _match_template(
        self,
        *,
        concept: Concept,
        pattern: PerceptionGraphTemplate,
        perception_graph: PerceptionGraph,
        confidence: float,
    ) -> Iterable[Tuple[PerceptionGraphPatternMatch, SemanticNode]]:
        """
        Try to match our model of the semantics to the perception graph,
        returning an iterable of such matches (some of which might be duplicates)
        """

    def _enrich_common(
        self, perception_semantic_alignment: PerceptionSemanticAlignment
    ) -> Tuple[PerceptionSemanticAlignment, AbstractSet[SemanticNode]]:
        """
        Shared code between `enrich_during_learning` and `enrich_during_description`.
        """
        preprocessing_result = self._preprocess_scene(perception_semantic_alignment)

        preprocessed_perception_graph = preprocessing_result.perception_graph

        # This accumulates our output.
        match_to_score: List[Tuple[SemanticNode, float]] = []

        # In the case of objects only, we alter the perception graph once they
        # are recognized by replacing the matched portion of the graph with the
        # ObjectSemanticNodes.  We gather them as we match and do the replacement below.
        matched_objects: List[Tuple[SemanticNode, PerceptionGraphPatternMatch]] = []

        # We pull this out into a function because we do matching in two passes:
        # first against templates whose meanings we are sure of (=have lexicalized)
        # and then, if no match has been found, against those we are still learning.
        def match_template(
            *, concept: Concept, pattern: PerceptionGraphTemplate, score: float
        ) -> None:
            matches_with_nodes = self._match_template(
                concept=concept,
                pattern=pattern.copy_with_temporal_scopes(ENTIRE_SCENE)
                if preprocessed_perception_graph.dynamic
                and not pattern.graph_pattern.dynamic
                else pattern,
                perception_graph=preprocessed_perception_graph,
                confidence=score,
            )
            # The template may have zero, one, or many matches, so we loop over the matches found
            # Note that, with the exception of the object learners,
            # some matches may be essentially identical to each other.
            # This is fine because the corresponding semantic nodes will be equal,
            # so they'll get thrown out when we take the immutable set of new nodes.
            for match_with_node in matches_with_nodes:
                (match, semantic_node_for_match) = match_with_node
                match_to_score.append((semantic_node_for_match, score))
                # We want to replace object matches with their semantic nodes,
                # but we don't want to alter the graph while matching it,
                # so we accumulate these to replace later.
                if isinstance(concept, ObjectConcept):
                    matched_objects.append((semantic_node_for_match, match))

            # For each template whose semantics we are certain of (=have been added to the lexicon)

        for (concept, graph_pattern, score) in self._primary_templates():
            check_state(isinstance(graph_pattern, PerceptionGraphTemplate))
            if (
                preprocessed_perception_graph.dynamic
                == graph_pattern.graph_pattern.dynamic
            ):
                match_template(concept=concept, pattern=graph_pattern, score=score)
            elif (
                preprocessed_perception_graph.dynamic
                and not graph_pattern.graph_pattern.dynamic
            ):
                match_template(
                    concept=concept,
                    pattern=graph_pattern.copy_with_temporal_scopes(ENTIRE_SCENE),
                    score=score,
                )
            elif (
                not preprocessed_perception_graph.dynamic
                and graph_pattern.graph_pattern.dynamic
            ):
                match_template(
                    concept=concept,
                    pattern=graph_pattern.copy_remove_temporal_scopes(),
                    score=score,
                )
            else:
                logging.debug(
                    f"Unable to try and match {concept} to {preprocessed_perception_graph} "
                    f"because both patterns must be static or dynamic"
                )
        if not match_to_score:
            # Try to match against patterns being learned
            # only if no lexicalized pattern was matched.
            for (concept, graph_pattern, score) in self._fallback_templates():
                # we may have multiple pattern hypotheses for a single concept, in which case we only want to identify the concept once
                if not any(m[0].concept == concept for m in match_to_score):
                    match_template(concept=concept, pattern=graph_pattern, score=score)

        perception_graph_after_matching = perception_semantic_alignment.perception_graph

        # Replace any objects found
        def by_pattern_complexity(pair):
            _, pattern_match = pair
            return pattern_match.matched_pattern.pattern_complexity()

        matched_objects.sort(key=by_pattern_complexity, reverse=True)
        already_replaced: Set[ObjectPerception] = set()
        new_nodes: List[SemanticNode] = []

        for (matched_object_node, pattern_match) in matched_objects:
            root = _get_root_object_perception(
                pattern_match.matched_sub_graph._graph,  # pylint:disable=protected-access
                immutableset(
                    pattern_match.matched_sub_graph._graph.nodes,  # pylint:disable=protected-access
                    disable_order_check=True,
                ),
            )
            if root not in already_replaced:
                try:
                    replacement_result = replace_match_with_object_graph_node(
                        matched_object_node=cast(ObjectSemanticNode, matched_object_node),
                        current_perception=perception_graph_after_matching,
                        pattern_match=pattern_match,
                    )
                    perception_graph_after_matching = (
                        replacement_result.perception_graph_after_replacement
                    )
                    already_replaced.update(
                        replacement_result.removed_nodes  # type: ignore
                    )
                    new_nodes.append(matched_object_node)
                except networkx.exception.NetworkXError:
                    logging.info(
                        f"Matched pattern for {matched_object_node} "
                        f"contains nodes that are already replaced"
                    )
            else:
                logging.info(
                    f"Matched pattern for {matched_object_node} "
                    f"but root object {root} already replaced."
                )
        # If objects, only include the replaced graph nodes in the enrichment
        if new_nodes and isinstance(new_nodes[0], ObjectSemanticNode):
            immutable_new_nodes = immutableset(new_nodes)
        else:
            immutable_new_nodes = immutableset(node for (node, _) in match_to_score)

        # Keep recursively enriching so we can capture plurals. Do it only if we matched objects in the scene.
        if new_nodes and isinstance(new_nodes[0], ObjectSemanticNode):
            rec = self._enrich_common(
                perception_semantic_alignment.copy_with_updated_graph_and_added_nodes(
                    new_graph=perception_graph_after_matching,
                    new_nodes=immutable_new_nodes,
                )
            )
            return rec[0], set(immutable_new_nodes).union(rec[1])

        (
            perception_graph_after_post_processing,
            nodes_after_post_processing,
        ) = self._enrich_post_process(
            perception_graph_after_matching, immutable_new_nodes
        )

        return (
            perception_semantic_alignment.copy_with_updated_graph_and_added_nodes(
                new_graph=perception_graph_after_post_processing,
                new_nodes=nodes_after_post_processing,
            ),
            nodes_after_post_processing,
        )

    def _pre_learning_step(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> None:
        """
        Perform any necessary steps before the learning logic.
        """

    @abstractmethod
    def _learning_step(
        self,
        language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,
    ) -> None:
        """
        Perform the actual learning logic.
        """

    def _post_learning_step(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> None:
        """
        Perform any necessary steps after the learning logic.
        """

    @abstractmethod
    def _can_learn_from(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> bool:
        """
        Check that the learner is capable of learning from this sort of learning example
        (at training time).
        """

    @abstractmethod
    def _preprocess_scene(
        self, perception_semantic_alignment: PerceptionSemanticAlignment
    ) -> PerceptionSemanticAlignment:
        """
        Does any preprocessing necessary before attempting to process a scene.
        """

    @abstractmethod
    def _candidate_templates(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> AbstractSet[SurfaceTemplateBoundToSemanticNodes]:
        r"""
        We treat learning as acquiring an association between "templates"
        over the token sequence and `PerceptionGraphTemplate`\ s.

        This method determines the surface template we are trying to learn semantics for
        for this particular training example.
        """

    @abstractmethod
    def _primary_templates(
        self,
    ) -> Iterable[Tuple[Concept, PerceptionGraphTemplate, float]]:
        """
        Our high-confidence (e.g. lexicalized) templates to match when describing a scene.
        """

    @abstractmethod
    def _fallback_templates(
        self,
    ) -> Iterable[Tuple[Concept, PerceptionGraphTemplate, float]]:
        """
        Get the secondary templates to try during description if none of the primary templates
        matched.
        """
