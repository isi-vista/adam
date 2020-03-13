import logging
from abc import ABC, abstractmethod
from typing import Mapping, List, Tuple, Union, Iterable, Sequence, Optional

from adam.language import TokenSequenceLinguisticDescription, LinguisticDescription
from adam.learner import LanguageLearner, LearningExample
from adam.learner.learner_utils import pattern_match_to_description
from adam.learner.object_recognizer import PerceptionGraphFromObjectRecognizer
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.surface_templates import SurfaceTemplate
from adam.perception import PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception.perception_graph import (
    LanguageAlignedPerception,
    PerceptionGraph,
    DebugCallableType,
)
from attr import attrib, attrs
from immutablecollections import immutabledict, ImmutableDict


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
        logging.info("Observation %s", self._observation_num)
        self._observation_num += 1

        self._assert_valid_input(learning_example)

        # Some learners need to track the alignment between perceived objects
        # and portions of the input language, so internally we operate over
        # LanguageAlignedPerceptions.
        original_language_aligned_perception = LanguageAlignedPerception(
            language=learning_example.linguistic_description,
            perception_graph=self._extract_perception_graph(learning_example.perception),
        )

        # Pre-processing steps will be different depending on
        # what sort of structures we are running.
        preprocessed_input = self._preprocess_scene_for_learning(
            original_language_aligned_perception
        )

        logging.info(f"Pursuit learner observing {preprocessed_input}")

        surface_template = self._extract_surface_template(preprocessed_input)
        self._learning_step(preprocessed_input, surface_template)

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
        self, language_aligned_perception: LanguageAlignedPerception
    ) -> LanguageAlignedPerception:
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
        self, preprocessed_input: LanguageAlignedPerception
    ) -> SurfaceTemplate:
        r"""
        We treat learning as acquiring an association between "templates"
        over the token sequence and `PerceptionGraphTemplate`\ s.

        This method determines the surface template we are trying to learn semantics for
        for this particular training example.
        """

    @abstractmethod
    def _learning_step(
        self,
        preprocessed_input: LanguageAlignedPerception,
        surface_template: SurfaceTemplate,
    ) -> None:
        pass

    @abstractmethod
    def _primary_templates(
        self
    ) -> Iterable[Tuple[SurfaceTemplate, PerceptionGraphTemplate, float]]:
        pass

    @abstractmethod
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
