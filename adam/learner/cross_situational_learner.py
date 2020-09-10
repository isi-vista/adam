from abc import ABC, abstractmethod

from immutablecollections import ImmutableSet, immutableset, ImmutableDict, immutabledict
from typing import Optional, Dict, AbstractSet, Iterable, Tuple, Set, Mapping

from attr import attrs, attrib, Factory
from attr.validators import instance_of, optional, in_

from adam.learner import (
    SurfaceTemplate,
    LanguagePerceptionSemanticAlignment,
    SurfaceTemplateBoundToSemanticNodes,
)
from adam.learner.learner_utils import (
    compute_match_ratio,
    pattern_match_to_semantic_node,
    PartialMatchRatio,
)
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.perception import MatchMode
from adam.semantics import Concept, SemanticNode
from vistautils.range import Range

from adam.learner.template_learner import AbstractTemplateLearnerNew


# Cross Situational Learner was originally implemented by Justin Martine on branch
# https://github.com/isi-vista/adam/pull/565 - Due to the age of the branch at time
# of refactor it was simplier to start a new branch than rebase
from adam.ontology.ontology import Ontology
from adam.perception.perception_graph import (
    DebugCallableType,
    GraphLogger,
    PerceptionGraphPatternMatch,
    PerceptionGraph,
)


@attrs
class AbstractCrossSituationalLearner(AbstractTemplateLearnerNew, ABC):
    """
    An Abstract Implementation of the Cross Situation Learning Model

    This learner aims to learn via storing all possible meanings and narrowing down to one meaning
    by caclulating association scores and probability based off those association scores for each
    utterance situation pair. It does so be associating all words to certain meanings. For new words
    meanings that are not associated strongly to another word already are associated evenly. For
    words encountered before, words are associated more strongly to meanings encountered with that
    word before and less strongly to newer meanings. Lastly, very familar word meaning pairs are
    associated together only, these would be words generally considered lexicalized. Once
    associations are made a probability for each word meaning pair being correct is caclulated.
    Finally if the proibability is high enough the word is lexicalized. More information can be
    found here: https://onlinelibrary.wiley.com/doi/full/10.1111/j.1551-6709.2010.01104.x
    """

    @attrs(slots=True, eq=False)
    class Hypothesis:
        pattern_template: PerceptionGraphTemplate = attrib(
            validator=instance_of(PerceptionGraphTemplate)
        )
        association_score: float = attrib(validator=instance_of(float), default=0)
        probability: float = attrib(validator=in_(Range.open(0, 1)), default=0)
        observation_count: int = attrib(init=False, default=1)

    _ontology: Ontology = attrib(validator=instance_of(Ontology), kw_only=True)
    _observation_num = attrib(init=False, default=0)
    _surface_template_to_concept: Dict[SurfaceTemplate, Concept] = attrib(
        init=False, default=Factory(dict)
    )
    _concept_to_surface_template: Dict[Concept, SurfaceTemplate] = attrib(
        init=False, default=Factory(dict)
    )
    _concept_to_hypotheses: Dict[
        Concept, ImmutableSet["AbstractCrossSituationalLearner.Hypothesis"]
    ] = attrib(init=False, default=Factory(dict))

    # Learner Internal Values
    _smoothing_parameter: float = attrib(
        validator=in_(Range.greater_than(0.0)), kw_only=True
    )
    """
    This smoothing factor is added to the scores of all hypotheses
    when forming a probability distribution over hypotheses.
    This should be a small value, at most 0.1 and possibly much less.
    See section 3.3 of the Cross-Situational paper.
    """
    _expected_number_of_meanings: float = attrib(
        validator=in_(Range.greater_than(0.0)), kw_only=True
    )
    _graph_match_confirmation_threshold: float = attrib(default=0.9, kw_only=True)
    _lexicon_entry_threshold: float = attrib(default=0.8, kw_only=True)
    _minimum_observation_amount: int = attrib(default=5, kw_only=True)

    # Debug Values
    _debug_callback: Optional[DebugCallableType] = attrib(default=None)
    _graph_logger: Optional[GraphLogger] = attrib(
        validator=optional(instance_of(GraphLogger)), default=None
    )

    def _learning_step(
        self,
        language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,
    ) -> None:
        """
        Try to learn the semantics of a `SurfaceTemplate` given the assumption
        that its argument slots (if any) are bound to objects according to
        *bound_surface_template*.

        For example, "try to learn the meaning of 'red' given the language 'red car'
        and an alignment of 'car' to particular perceptions in the perception graph.
        """
        # Generate all possible meanings from the Graph

        meanings_from_perception = self._hypotheses_from_perception(
            language_perception_semantic_alignment, bound_surface_template
        )
        meanings_to_pattern_template = immutabledict(
            (meaning, PerceptionGraphTemplate.from_graph(meaning, immutabledict()))
            for meaning in meanings_from_perception
        )
        meanings_to_remove: Set[PerceptionGraphTemplate] = set()

        def check_and_remove_meaning(
            hypothesis: "AbstractCrossSituationalLearner.Hypothesis",
            *,
            ontology: Ontology,
        ) -> None:
            match = compute_match_ratio(
                hypothesis.pattern_template,
                language_perception_semantic_alignment.perception_semantic_alignment.perception_graph,
                ontology=ontology,
            )
            if match and match.matching_subgraph:
                for meaning in meanings_from_perception:
                    if match.matching_subgraph.check_isomorphism(
                        meanings_to_pattern_template[meaning].graph_pattern
                    ):
                        meanings_to_remove.add(meaning)

        for (concept, hypotheses) in self._concept_to_hypotheses.items():
            for hypothesis in hypotheses:
                if hypothesis.probability > self._lexicon_entry_threshold:
                    check_and_remove_meaning(hypothesis, ontology=self._ontology)

        meanings_after_preprocessing = immutableset(
            meaning
            for meaning in meanings_from_perception
            if meaning not in meanings_to_remove
        )

        # We have seen this template before and already have a concept for it
        # So we attempt to verify our already picked concept
        if bound_surface_template.surface_template in self._surface_template_to_concept:
            # We don't directly associate surface templates with perceptions.
            # Instead we mediate the relationship with "concept" objects.
            # These don't matter now, but the split might be helpful in the future
            # when we might have multiple ways of expressing the same idea.
            concept = self._surface_template_to_concept[
                bound_surface_template.surface_template
            ]
        else:
            concept = self._new_concept(
                debug_string=bound_surface_template.surface_template.to_short_string()
            )

        meaning_to_probability = self._get_probabilities(
            concept, meanings_after_preprocessing, meanings_to_pattern_template
        )

        raise NotImplementedError()

    def _get_probabilities(
        self,
        concept: Concept,
        meanings: Iterable[PerceptionGraph],
        meaning_to_pattern: Mapping[PerceptionGraph, PerceptionGraphTemplate],
    ) -> ImmutableDict[PerceptionGraph, float]:
        """
        Update all Concept meaning probabilities for given word as defined by the papenr below
        p(m|c) = assoc(m, c) + lambda / sum(for c' in C)(assoc(c', w)) + (beta * lambda)
        where w and m are given concept and meanings, lambda is a smoothing factor, M is all
        meanings encountered, beta is the expected number of meaning types.
        https://onlinelibrary.wiley.com/doi/full/10.1111/j.1551-6709.2010.01104.x (3)
        """
        meaning_to_probability: Dict[PerceptionGraph, float] = dict()

        for meaning in meanings:
            if self._find_similiar_hypotesis(meaning, meanings, meaning_to_pattern) == (
                None,
                None,
            ):
                new_hypothesis = AbstractCrossSituationalLearner.Hypothesis(
                    pattern_template=meaning_to_pattern[meaning]
                )
                normalizing_factor = (
                    self._expected_number_of_meanings * self._smoothing_parameter
                )

                # for other_meaning in

                new_hypothesis.probability = (
                    self._smoothing_parameter / normalizing_factor
                )

        raise NotImplementedError()

        return immutabledict(meaning_to_probability)

    def _find_similiar_hypotesis(
        self,
        new_meaning: PerceptionGraph,
        candidates: Iterable[PerceptionGraph],
        meaning_to_pattern: Mapping[PerceptionGraph, PerceptionGraphTemplate],
    ) -> Tuple[Optional[PartialMatchRatio], Optional[PerceptionGraphTemplate]]:
        """
        Finds the entry in candidates most similar to new_meaning and returns it with the ratio.
        """
        candidates_iter = iter(candidates)
        match = None
        while match is None:
            try:
                old_meaning = next(candidates_iter)
            except StopIteration:
                return None, None

            try:
                match = compute_match_ratio(
                    meaning_to_pattern[old_meaning], new_meaning, ontology=self._ontology
                )
            except RuntimeError:
                # Occurs when no matches of the pattern are found in the graph. This seems to
                # to indicate some full matches and some matches with no intersection at all
                pass

        for candidate in candidates:
            try:
                new_match = compute_match_ratio(
                    meaning_to_pattern[candidate], new_meaning, ontology=self._ontology
                )
            except RuntimeError:
                # Occurs when no matches of the pattern are found in the graph. This seems to
                # to indicate some full matches and some matches with no intersection at all
                new_match = None
            if new_match and new_match.match_ratio > match.match_ratio:
                match = new_match
                old_meaning = candidate
        if (
            match.match_ratio >= self._graph_match_confirmation_threshold
            and match.matching_subgraph
            and old_meaning
        ):
            return match, meaning_to_pattern[old_meaning]
        else:
            return None, None

    def templates_for_concept(self, concept: Concept) -> AbstractSet[SurfaceTemplate]:
        if concept in self._concept_to_surface_template:
            return immutableset([self._concept_to_surface_template[concept]])
        else:
            return immutableset()

    @abstractmethod
    def _new_concept(self, debug_string: str) -> Concept:
        """
        Create a new `Concept` of the appropriate type with the given *debug_string*.
        """

    @abstractmethod
    def _hypotheses_from_perception(
        self,
        learning_state: LanguagePerceptionSemanticAlignment,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,
    ) -> Iterable[PerceptionGraph]:
        """
        Get a hypothesis for the meaning of *surface_template* from a given *learning_state*.
        """

    def _primary_templates(
        self
    ) -> Iterable[Tuple[Concept, PerceptionGraphTemplate, float]]:
        return (
            (concept, hypothesis.pattern_template, hypothesis.probability)
            for (concept, hypotheses) in self._concept_to_hypotheses.items()
            # We are confident in a hypothesis if it's above our _lexicon_entry_threshold
            # and we've seen this concept our _minimum_observation_amount
            for hypothesis in hypotheses
            if hypothesis.observation_count >= self._minimum_observation_amount
            and hypothesis.probability >= self._lexicon_entry_threshold
        )

    def _fallback_templates(
        self
    ) -> Iterable[Tuple[Concept, PerceptionGraphTemplate, float]]:
        # Alternate hypotheses either below our _lexicon_entry_threshold or our _minimum_observation_amount
        return (
            (concept, hypothesis.pattern_template, hypothesis.probability)
            for (concept, hypotheses) in self._concept_to_hypotheses.items()
            for hypothesis in hypotheses
            if hypothesis.observation_count < self._minimum_observation_amount
            or hypothesis.probability < self._lexicon_entry_threshold
        )

    def _match_template(
        self,
        *,
        concept: Concept,
        pattern: PerceptionGraphTemplate,
        perception_graph: PerceptionGraph,
    ) -> Optional[Tuple[PerceptionGraphPatternMatch, SemanticNode]]:
        """
        Try to match our model of the semantics to the perception graph
        """
        partial_match = compute_match_ratio(
            pattern,
            perception_graph,
            ontology=self._ontology,
            graph_logger=self._graph_logger,
            debug_callback=self._debug_callback,
        )

        if (
            partial_match.match_ratio >= self._graph_match_confirmation_threshold
            and partial_match.matching_subgraph
        ):
            # if there is a match, which is above our minimum match ratio
            # Use that pattern to try and find a match in the scene
            # There should be one
            # TODO: This currently means we match to the graph multiple times. Ho
            matcher = partial_match.matching_subgraph.matcher(
                perception_graph,
                match_mode=MatchMode.NON_OBJECT,
                debug_callback=self._debug_callback,
            )
            for match in matcher.matches(use_lookahead_pruning=True):
                semantic_node_for_match = pattern_match_to_semantic_node(
                    concept=concept, pattern=pattern, match=match
                )
                # A template only has to match once; we don't care about finding additional matches.
                return match, semantic_node_for_match
            # We raise an error if we find a partial match but don't manage to match it to the scene
            raise RuntimeError(
                f"Partial Match found for {concept} below match ratio however pattern "
                f"subgraph was unable to match to perception graph.\n"
                f"Partial Match: {partial_match}\n"
                f"Perception Graph: {perception_graph}"
            )
        return None
