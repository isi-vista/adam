import logging
from abc import ABC, abstractmethod

from immutablecollections import ImmutableSet, immutableset
from more_itertools import first

from typing import Dict, Optional, AbstractSet, Iterable, Tuple

from attr import attrs, attrib, Factory
from attr.validators import instance_of, optional

from adam.learner import (
    SurfaceTemplateBoundToSemanticNodes,
    LanguagePerceptionSemanticAlignment,
    SurfaceTemplate,
)
from adam.learner.learner_utils import compute_match_ratio, pattern_match_to_semantic_node
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.template_learner import AbstractTemplateLearnerNew
from adam.ontology.ontology import Ontology
from adam.perception import MatchMode
from adam.perception.perception_graph import (
    GraphLogger,
    DebugCallableType,
    PerceptionGraph,
    PerceptionGraphPatternMatch,
    PerceptionGraphPattern,
)
from adam.random_utils import RandomChooser
from adam.semantics import Concept, SemanticNode


# PbV was originally implemented by Ashwin on branch https://github.com/isi-vista/adam/pull/564
# and was adapted by Jacob Lichtefeld into the new style learner. Due to the age of the branch
# at time of writing it was simplier to start a new branch than rebase
@attrs
class AbstractProposeButVerifyLearner(AbstractTemplateLearnerNew, ABC):
    """
    An Abstract Implementation of the Propose but Verify Learning Model

    This model does not retain any information pertaining to
    disconfirmed hypothesis and is based on a "win-stay, lose-shift" ideology.

    Reference: https://www.ling.upenn.edu/~ycharles/papers/pursuit-final.pdf
    Section 1 of this paper on pursuit learning includes an introduction to the
    'propose but verify' learning approach.
    """

    _ontology: Ontology = attrib(validator=instance_of(Ontology), kw_only=True)
    _observation_num = attrib(init=False, default=0)
    _concept_to_hypotheses: Dict[Concept, ImmutableSet[PerceptionGraphTemplate]] = attrib(
        init=False, default=Factory(dict)
    )
    _surface_template_to_concept: Dict[SurfaceTemplate, Concept] = attrib(
        init=False, default=Factory(dict)
    )
    _concept_to_surface_template: Dict[Concept, SurfaceTemplate] = attrib(
        init=False, default=Factory(dict)
    )
    # Internal Params
    _rng: RandomChooser = attrib(validator=instance_of(RandomChooser))
    _graph_logger: Optional[GraphLogger] = attrib(
        validator=optional(instance_of(GraphLogger)), default=None
    )
    _debug_callback: Optional[DebugCallableType] = attrib(default=None)
    _graph_match_confirmation_threshold: float = attrib(default=0.9, kw_only=True)
    _minimum_match_ratio: float = attrib(default=0.8, kw_only=True)

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
        concept = None
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

            # What is our current hypotheses about what this template might mean?
            pattern_hypotheses = self._concept_to_hypotheses[concept]

            # We have a hypothesis, now we check if the current scene can verify it
            # So we try to match our hypothesis to the scene accepting a partial match
            partial_match = compute_match_ratio(
                first(pattern_hypotheses),
                language_perception_semantic_alignment.perception_semantic_alignment.perception_graph,
                ontology=self._ontology,
                graph_logger=self._graph_logger,
                debug_callback=self._debug_callback,
            )

            # Now we want to see if our hypothesis is confirmed. We do this by seeing if the
            # *match_ratio* is above the required value
            if partial_match.match_ratio >= self._minimum_match_ratio:
                logging.debug(
                    f"Hypothesis for {concept} is confirmed with ratio: {partial_match.match_ratio}"
                )
                # We've verified our hypothesis so we don't need to learn anything from this scene.
                # Note - We currently don't do any generalizing of our hypothesis space
                # Which might be needed for anything other than objects
                # Or the `minimum_match_ratio` value should be evaluated for those cases
                # As the values from the pursuit paper are focused around objects
                return

        # If we either haven't seen this semantic template before or our hypothesis
        # Wasn't confirmed We initialize a new concept or replace the current one.
        if not concept:
            concept = self._new_concept(
                debug_string=bound_surface_template.surface_template.to_short_string()
            )
        self._surface_template_to_concept[
            bound_surface_template.surface_template
        ] = concept
        self._concept_to_surface_template[
            concept
        ] = bound_surface_template.surface_template

        self._concept_to_hypotheses[concept] = immutableset(
            self._hypotheses_from_perception(
                language_perception_semantic_alignment, bound_surface_template
            )
        )

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
    ) -> AbstractSet[PerceptionGraphTemplate]:
        """
        Get a hypothesis for the meaning of *surface_template* from a given *learning_state*.
        """

    def _primary_templates(
        self
    ) -> Iterable[Tuple[Concept, PerceptionGraphTemplate, float]]:
        return (
            (concept, hypotheses[0], 1.0)
            for (concept, hypotheses) in self._concept_to_hypotheses.items()
            # We are confident in a hypothesis if we don't have any alternatives.
            if len(hypotheses) == 1
        )

    def _fallback_templates(
        self
    ) -> Iterable[Tuple[Concept, PerceptionGraphTemplate, float]]:
        # Alternate hypotheses stored in the beam.
        return (
            (concept, hypothesis, 1.0)
            for (concept, hypotheses) in self._concept_to_hypotheses.items()
            for hypothesis in hypotheses[1:]
            if len(hypotheses) > 1
        )

    def concepts_to_patterns(self) -> Dict[Concept, PerceptionGraphPattern]:
        return {k: v.graph_pattern for k, v, _ in self._primary_templates()}

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
            partial_match.match_ratio >= self._minimum_match_ratio
            and partial_match.matching_subgraph
        ):
            # if there is a match, which is above our minimum match ratio
            # Use that pattern to try and find a match in the scene
            # There should be one
            # TODO: This currently means we match to the graph multiple times. Reduce this?
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
