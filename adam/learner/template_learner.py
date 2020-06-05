import logging
from abc import ABC, abstractmethod
from typing import AbstractSet, Iterable, List, Mapping, Sequence, Tuple, Union, cast

from more_itertools import one

from adam.language import LinguisticDescription, TokenSequenceLinguisticDescription
from adam.learner import LanguageLearner, LearningExample, NewStyleLearner
from adam.learner.alignments import (
    LanguageConceptAlignment,
    LanguagePerceptionSemanticAlignment,
    PerceptionSemanticAlignment,
)
from adam.learner.learner_utils import (
    pattern_match_to_description,
    pattern_match_to_semantic_node,
)
from adam.learner.object_recognizer import (
    PerceptionGraphFromObjectRecognizer,
    replace_match_with_object_graph_node,
)
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.surface_templates import (
    SurfaceTemplateBoundToSemanticNodes,
    SurfaceTemplate,
)
from adam.perception import PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception.perception_graph import PerceptionGraph, PerceptionGraphPatternMatch
from adam.semantics import Concept, ObjectConcept, ObjectSemanticNode, SemanticNode
from attr import attrib, attrs, evolve
from immutablecollections import immutabledict, immutableset
from vistautils.preconditions import check_state


@attrs
class AbstractTemplateLearner(
    LanguageLearner[
        DevelopmentalPrimitivePerceptionFrame, TokenSequenceLinguisticDescription
    ],
    ABC,
):
    _observation_num: int = attrib(init=False, default=0)

    def observe(
        self,
        learning_example: LearningExample[
            DevelopmentalPrimitivePerceptionFrame, LinguisticDescription
        ],
    ) -> None:
        logging.info(
            "Observation %s: %s",
            self._observation_num,
            learning_example.linguistic_description.as_token_string(),
        )

        self._observation_num += 1

        self._assert_valid_input(learning_example)

        # Pre-processing steps will be different depending on
        # what sort of structures we are running.
        preprocessed_input = self._preprocess_scene_for_learning(
            learning_example.linguistic_description,
            self._extract_perception_graph(learning_example.perception),
        )

        logging.info(f"Learner observing {preprocessed_input}")

        surface_template = self._extract_surface_template(
            preprocessed_input.language_concept_alignment
        )
        self._learning_step(
            preprocessed_input.language_concept_alignment, surface_template
        )

    def describe(
        self, perception: PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame]
    ) -> Mapping[LinguisticDescription, float]:
        self._assert_valid_input(perception)

        original_perception_graph = self._extract_perception_graph(perception)
        preprocessing_result = self._preprocess_scene_for_description(
            original_perception_graph
        )

        preprocessed_perception_graph = preprocessing_result.perception_graph
        matched_objects_to_names = (
            preprocessing_result.description_to_matched_object_node.inverse()
        )
        # This accumulates our output.
        match_to_score: List[
            Tuple[TokenSequenceLinguisticDescription, PerceptionGraphTemplate, float]
        ] = []

        # We pull this out into a function because we do matching in two passes:
        # first against templates whose meanings we are sure of (=have lexicalized)
        # and then, if no match has been found, against those we are still learning.
        def match_template(
            *,
            description_template: SurfaceTemplate,
            pattern: PerceptionGraphTemplate,
            score: float,
        ) -> None:
            # try to see if (our model of) its semantics is present in the situation.
            matcher = pattern.graph_pattern.matcher(
                preprocessed_perception_graph,
                matching_objects=False,
                # debug_callback=self._debug_callback,
            )
            for match in matcher.matches(use_lookahead_pruning=True):
                # if there is a match, use that match to describe the situation.
                match_to_score.append(
                    (
                        pattern_match_to_description(
                            surface_template=description_template,
                            pattern=pattern,
                            match=match,
                            matched_objects_to_names=matched_objects_to_names,
                        ),
                        pattern,
                        score,
                    )
                )
                # A template only has to match once; we don't care about finding additional matches.
                return

        # For each template whose semantics we are certain of (=have been added to the lexicon)
        for (surface_template, graph_pattern, score) in self._primary_templates():
            match_template(
                description_template=surface_template, pattern=graph_pattern, score=score
            )

        if not match_to_score:
            # Try to match against patterns being learned
            # only if no lexicalized pattern was matched.
            for (surface_template, graph_pattern, score) in self._fallback_templates():
                match_template(
                    description_template=surface_template,
                    pattern=graph_pattern,
                    score=score,
                )

        return immutabledict(self._post_process_descriptions(match_to_score))

    @abstractmethod
    def _assert_valid_input(
        self,
        to_check: Union[
            LearningExample[DevelopmentalPrimitivePerceptionFrame, LinguisticDescription],
            PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame],
        ],
    ) -> None:
        """
        Check that the learner is capable of handling this sort of learning example
        (at training time) or perception (at description time).
        """

    @abstractmethod
    def _extract_perception_graph(
        self, perception: PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame]
    ) -> PerceptionGraph:
        """
        Transforms the observed *perception* into a `PerceptionGraph`.

        This should just do the basic transformation.
        Leave further processing on the graph for `_preprocess_scene_for_learning`
        and `preprocess_scene_for_description`.
        """

    @abstractmethod
    def _preprocess_scene_for_learning(
        self, language: LinguisticDescription, perception: PerceptionGraph
    ) -> LanguagePerceptionSemanticAlignment:
        """
        Does any preprocessing necessary before the learning process begins.

        This will typically share some common code with `_preprocess_scene_for_description`.
        """

    @abstractmethod
    def _preprocess_scene_for_description(
        self, perception_graph: PerceptionGraph
    ) -> PerceptionGraphFromObjectRecognizer:
        """
        Does any preprocessing necessary before attempting to describe a scene.

        This will typically share some common code with `_preprocess_scene_for_learning`.
        """

    @abstractmethod
    def _extract_surface_template(
        self, language_concept_alignment: LanguageConceptAlignment
    ) -> SurfaceTemplate:
        r"""
        We treat learning as acquiring an association between "templates"
        over the token sequence and `PerceptionGraphTemplate`\ s.

        This method determines the surface template we are trying to learn semantics for
        for this particular training example.
        """

    def _learning_step(
        self,
        preprocessed_input: LanguageConceptAlignment,
        surface_template: SurfaceTemplate,
    ) -> None:
        pass

    def _primary_templates(
        self
    ) -> Iterable[Tuple[SurfaceTemplate, PerceptionGraphTemplate, float]]:
        pass

    def _fallback_templates(
        self
    ) -> Iterable[Tuple[SurfaceTemplate, PerceptionGraphTemplate, float]]:
        pass

    def _post_process_descriptions(
        self,
        match_results: Sequence[
            Tuple[TokenSequenceLinguisticDescription, PerceptionGraphTemplate, float]
        ],
    ) -> Mapping[TokenSequenceLinguisticDescription, float]:
        return immutabledict(
            (description, score) for (description, _, score) in match_results
        )


class TemplateLearner(NewStyleLearner, ABC):
    @abstractmethod
    def templates_for_concept(self, concept: Concept) -> AbstractSet[SurfaceTemplate]:
        pass


@attrs
class AbstractTemplateLearnerNew(TemplateLearner, ABC):
    """
    Super-class for learners using template-based syntax-semantics mappings.
    """

    _observation_num: int = attrib(init=False, default=0)

    def learn_from(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> None:
        logging.info(
            "Observation %s: %s",
            self._observation_num,
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

        for thing_whose_meaning_to_learn in self._candidate_templates(
            language_perception_semantic_alignment
        ):
            self._learning_step(preprocessed_input, thing_whose_meaning_to_learn)

    def enrich_during_learning(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> LanguagePerceptionSemanticAlignment:
        (
            perception_post_enrichment,
            newly_recognized_semantic_nodes,
        ) = self._enrich_common(
            language_perception_semantic_alignment.perception_semantic_alignment
        )

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
                ),
                filter_out_duplicate_alignments=True,
            ),
            perception_semantic_alignment=perception_post_enrichment,
        )

    def enrich_during_description(
        self, perception_semantic_alignment: PerceptionSemanticAlignment
    ) -> PerceptionSemanticAlignment:
        # The other information returned by _enrich_common is only needed by
        # enrich_during_learning.
        return self._enrich_common(perception_semantic_alignment)[0]

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
            # try to see if (our model of) its semantics is present in the situation.
            matcher = pattern.graph_pattern.matcher(
                preprocessed_perception_graph,
                matching_objects=False,
                # debug_callback=self._debug_callback,
            )
            for match in matcher.matches(use_lookahead_pruning=True):
                # if there is a match, use that match to describe the situation.
                semantic_node_for_match = pattern_match_to_semantic_node(
                    concept=concept, pattern=pattern, match=match
                )
                match_to_score.append((semantic_node_for_match, score))
                # We want to replace object matches with their semantic nodes,
                # but we don't want to alter the graph while matching it,
                # so we accumulate these to replace later.
                if isinstance(semantic_node_for_match, ObjectConcept):
                    matched_objects.append((semantic_node_for_match, match))
                # A template only has to match once; we don't care about finding additional matches.
                return

        # For each template whose semantics we are certain of (=have been added to the lexicon)
        for (concept, graph_pattern, score) in self._primary_templates():
            check_state(isinstance(graph_pattern, PerceptionGraphTemplate))
            match_template(concept=concept, pattern=graph_pattern, score=score)

        if not match_to_score:
            # Try to match against patterns being learned
            # only if no lexicalized pattern was matched.
            for (concept, graph_pattern, score) in self._fallback_templates():
                match_template(concept=concept, pattern=graph_pattern, score=score)

        perception_graph_after_matching = perception_semantic_alignment.perception_graph

        # Replace any objects found
        for (matched_object_node, pattern_match) in matched_objects:
            perception_graph_after_matching = replace_match_with_object_graph_node(
                matched_object_node=cast(ObjectSemanticNode, matched_object_node),
                current_perception=perception_graph_after_matching,
                pattern_match=pattern_match,
            )

        new_nodes = immutableset(node for (node, _) in match_to_score)

        return (
            perception_semantic_alignment.copy_with_updated_graph_and_added_nodes(
                new_graph=perception_graph_after_matching, new_nodes=new_nodes
            ),
            new_nodes,
        )

    @abstractmethod
    def _learning_step(
        self,
        language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,
    ) -> None:
        """
        Perform the actual learning logic.
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
        self
    ) -> Iterable[Tuple[Concept, PerceptionGraphTemplate, float]]:
        """
        Our high-confidence (e.g. lexicalized) templates to match when describing a scene.
        """

    @abstractmethod
    def _fallback_templates(
        self
    ) -> Iterable[Tuple[Concept, PerceptionGraphTemplate, float]]:
        """
        Get the secondary templates to try during description if none of the primary templates
        matched.
        """
