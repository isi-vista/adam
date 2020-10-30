import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import AbstractSet, Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

from attr import Factory, attrib, attrs
from attr.validators import instance_of
from immutablecollections import ImmutableSet, immutabledict, immutableset

from adam.language import TokenSequenceLinguisticDescription
from adam.learner.alignments import LanguagePerceptionSemanticAlignment
from adam.learner.learner_utils import pattern_match_to_semantic_node
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.surface_templates import (
    SurfaceTemplate,
    SurfaceTemplateBoundToSemanticNodes,
)
from adam.learner.template_learner import (
    AbstractTemplateLearner,
    AbstractTemplateLearnerNew,
)
from adam.ontology.ontology import Ontology
from adam.perception import MatchMode
from adam.perception.deprecated import LanguageAlignedPerception
from adam.perception.perception_graph import (
    DebugCallableType,
    PerceptionGraph,
    PerceptionGraphPatternMatch,
    PerceptionGraphPattern,
)
from adam.semantics import Concept, SemanticNode


@attrs
class AbstractSubsetLearner(AbstractTemplateLearner, ABC):
    _surface_template_to_hypothesis: Dict[
        SurfaceTemplate, PerceptionGraphTemplate
    ] = attrib(init=False, default=Factory(dict))
    _ontology: Ontology = attrib(validator=instance_of(Ontology), kw_only=True)
    _debug_callback: Optional[DebugCallableType] = attrib(default=None, kw_only=True)

    @abstractmethod
    def _update_hypothesis(
            self,
            previous_pattern_hypothesis: PerceptionGraphTemplate,
            current_pattern_hypothesis: PerceptionGraphTemplate,
    ) -> Optional[PerceptionGraphTemplate]:
        """
        Method to handle how to intersect hypothesis to possibly update hypothesis
        """

    def _learning_step(
            self,
            preprocessed_input: LanguageAlignedPerception,
            surface_template: SurfaceTemplate,
    ) -> None:
        if surface_template in self._surface_template_to_hypothesis:
            # If already observed, get the largest matching subgraph of the pattern in the
            # current observation and
            # previous pattern hypothesis
            # TODO: We should relax this requirement for learning: issue #361
            previous_pattern_hypothesis = self._surface_template_to_hypothesis[
                surface_template
            ]

            updated_hypothesis = self._update_hypothesis(
                previous_pattern_hypothesis,
                self._hypothesis_from_perception(preprocessed_input),
            )

            if updated_hypothesis:
                # Update the leading hypothesis
                self._surface_template_to_hypothesis[
                    surface_template
                ] = updated_hypothesis
            else:
                logging.warning(
                    "Intersection of graphs had empty result; keeping original pattern"
                )

        else:
            # If it's a new description, learn a new hypothesis/pattern, generated as a pattern
            # graph frm the
            # perception graph.
            self._surface_template_to_hypothesis[
                surface_template
            ] = self._hypothesis_from_perception(preprocessed_input)

    @abstractmethod
    def _hypothesis_from_perception(
            self, preprocessed_input: LanguageAlignedPerception
    ) -> PerceptionGraphTemplate:
        pass

    def _primary_templates(
            self
    ) -> Iterable[Tuple[SurfaceTemplate, PerceptionGraphTemplate, float]]:
        return (
            (surface_template, hypothesis, 1.0)
            for (
            surface_template,
            hypothesis,
        ) in self._surface_template_to_hypothesis.items()
        )

    def _fallback_templates(
            self
    ) -> Iterable[Tuple[SurfaceTemplate, PerceptionGraphTemplate, float]]:
        return tuple()

    def _post_process_descriptions(
            self,
            match_results: Sequence[
                Tuple[TokenSequenceLinguisticDescription, PerceptionGraphTemplate, float]
            ],
    ) -> Mapping[TokenSequenceLinguisticDescription, float]:
        if not match_results:
            return immutabledict()

        largest_pattern_num_nodes = max(
            len(template.graph_pattern) for (_, template, _) in match_results
        )

        return immutabledict(
            (description, len(template.graph_pattern) / largest_pattern_num_nodes)
            for (description, template, score) in match_results
        )


@attrs
class AbstractSubsetLearnerNew(AbstractTemplateLearnerNew, ABC):
    _beam_size: int = attrib(validator=instance_of(int), kw_only=True)
    _concept_to_hypotheses: Dict[Concept, ImmutableSet[PerceptionGraphTemplate]] = attrib(
        init=False, default=Factory(dict)
    )
    concept_to_surface_template: Dict[Concept, SurfaceTemplate] = attrib(
        init=False, default=Factory(dict)
    )
    surface_template_to_concept: Dict[SurfaceTemplate, Concept] = attrib(
        init=False, default=Factory(dict)
    )
    _ontology: Ontology = attrib(validator=instance_of(Ontology), kw_only=True)
    _debug_callback: Optional[DebugCallableType] = attrib(default=None, kw_only=True)

    _known_bad_patterns: Set[SurfaceTemplate] = attrib(init=False, default=Factory(set))

    @abstractmethod
    def _update_hypothesis(
            self,
            previous_pattern_hypothesis: PerceptionGraphTemplate,
            current_pattern_hypothesis: PerceptionGraphTemplate,
    ) -> Optional[PerceptionGraphTemplate]:
        """
        Method to handle how to intersect hypothesis to possibly update hypothesis
        """

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
        if bound_surface_template.surface_template in self._known_bad_patterns:
            # We tried to learn an alignment for this surface template previously
            # and it didn't work out.
            # For example, early on, we might think 'the' could be an object,
            # but eventually we will become quite sure it isn't one.
            return

        if bound_surface_template.surface_template in self.surface_template_to_concept:
            # We've seen this template before and already have some hypothesis about what it means
            # which we need to confirm or refine.
            # If already observed, get the largest matching subgraph of the pattern in the
            # current observation and
            # previous pattern hypothesis

            # We don't directly associate surface templates with perceptions.
            # Instead we mediate the relationship with "concept" objects.
            # These don't matter now, but the split might be helpful in the future
            # when we might have multiple ways of expressing the same idea.
            concept_for_surface_template = self.surface_template_to_concept[
                bound_surface_template.surface_template
            ]
            # What is our current hypotheses about what this template might mean?
            previous_pattern_hypotheses = self._concept_to_hypotheses[
                concept_for_surface_template
            ]

            # If we tracked only a single hypothesis for each template, things would be simple:
            # our new hypotheses would simply be what remains consistent
            # between our previous belief and our current observation.
            # But since we can track multiple hypotheses
            # (and can generate multiple hypotheses from a single observation),
            # we need to try to match what is in common between each of our existing hypotheses
            # and each hypothesis we might generate form the current perception alone.
            # Storing the results of all these intersections as new hypotheses could
            # lead to an explosion in our number of hypotheses over time,
            # so we use a beam search to consider only the best possibilities,
            # which we defined as those having the most nodes.

            hypotheses_from_current_perception = self._hypotheses_from_perception(
                language_perception_semantic_alignment,
                bound_surface_template=bound_surface_template,
            )
            updated_hypotheses_maybe_null = [
                self._update_hypothesis(
                    previous_pattern_hypothesis, hypothesis_from_current_perception
                )
                for previous_pattern_hypothesis in previous_pattern_hypotheses
                for hypothesis_from_current_perception in hypotheses_from_current_perception
            ]

            def should_keep_hypothesis(hypothesis: PerceptionGraphTemplate) -> bool:
                if len(hypothesis.template_variable_to_pattern_node) != len(
                        bound_surface_template.slot_to_semantic_node
                ):
                    # We've managed to lose our wildcard slot somehow.
                    return False
                return self._keep_hypothesis(
                    hypothesis=hypothesis, bound_surface_template=bound_surface_template
                )

            # Remove all Nones resulting from empty intersections,
            # as well as any hypotheses which fail learner-specific conditions.
            updated_hypotheses = [
                hypothesis
                for hypothesis in updated_hypotheses_maybe_null
                if hypothesis and should_keep_hypothesis(hypothesis=hypothesis)
            ]
            # Sort hypotheses by decreasing order of size
            updated_hypotheses.sort(key=lambda x: len(x.graph_pattern), reverse=True)
            # Confine our search by beam size
            if len(updated_hypotheses) > self._beam_size:
                updated_hypotheses = updated_hypotheses[: self._beam_size]

            # Update the leading hypotheses
            self._concept_to_hypotheses[concept_for_surface_template] = immutableset(
                updated_hypotheses
            )

            if not updated_hypotheses:
                logging.debug(
                    "All hypotheses failed for surface template %s for learner "
                    "%s; assuming template is not of the target type",
                    type(self),
                    bound_surface_template.surface_template,
                )
                self._known_bad_patterns.add(bound_surface_template.surface_template)
        else:
            # # Skip if a template is already recognized in perception (prevents learning "two" in two "ball" s)
            # if any(cand.surface_template in self.surface_template_to_concept for cand in
            #        self._candidate_templates(language_perception_semantic_alignment)):
            #     return

            # If it's a new template, learn a new hypothesis/pattern, generated as a pattern
            # graph from the perception graph.
            concept = self._new_concept(
                debug_string=bound_surface_template.surface_template.to_short_string()
            )
            self.surface_template_to_concept[
                bound_surface_template.surface_template
            ] = concept
            self.concept_to_surface_template[
                concept
            ] = bound_surface_template.surface_template

            self._concept_to_hypotheses[concept] = immutableset(
                self._hypotheses_from_perception(
                    language_perception_semantic_alignment, bound_surface_template
                )
            )

    def templates_for_concept(self, concept: Concept) -> AbstractSet[SurfaceTemplate]:
        if concept in self.concept_to_surface_template:
            return immutableset([self.concept_to_surface_template[concept]])
        else:
            return immutableset()

    @abstractmethod
    def _new_concept(self, debug_string: str) -> Concept:
        """
        Create a new `Concept` of the appropriate type with the given *debug_string*.
        """

    @abstractmethod
    def _keep_hypothesis(
            self,
            *,
            hypothesis: PerceptionGraphTemplate,
            bound_surface_template: SurfaceTemplateBoundToSemanticNodes,
    ) -> bool:
        """
        Should a candidate hypothesis for the meaning of *bound_surface_template* be kept,
        or should it be discarded.

        Typically this is checking things like whether the hypothesis has gotten too small
        to plausibly encode the semantics of the template.

        This method may assume that the slots of *hypothesis* match
        the slots of the *bound_surface_template*;
        if not, the hypothesis would be automatically rejected.
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


@attrs  # pylint:disable=abstract-method
class AbstractTemplateSubsetLearner(AbstractSubsetLearner, AbstractTemplateLearner, ABC):
    def log_hypotheses(self, log_output_path: Path) -> None:
        logging.info(
            "Logging %s hypotheses to %s",
            len(self._surface_template_to_hypothesis),
            log_output_path,
        )
        for (
            surface_template,
            hypothesis,
        ) in self._surface_template_to_hypothesis.items():
            template_string = surface_template.to_short_string()
            hypothesis.render_to_file(template_string, log_output_path / template_string)


class AbstractTemplateSubsetLearnerNew(
    AbstractSubsetLearnerNew, AbstractTemplateLearnerNew, ABC
):
    # pylint:disable=abstract-method
    def log_hypotheses(self, log_output_path: Path) -> None:
        logging.info(
            "Logging %s hypotheses to %s",
            len(self._concept_to_hypotheses),
            log_output_path,
        )
        for (concept, hypotheses) in self._concept_to_hypotheses.items():
            for (i, hypothesis) in enumerate(hypotheses):
                hypothesis.render_to_file(
                    concept.debug_string, log_output_path / f"{concept.debug_string}.{i}"
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
        matcher = pattern.graph_pattern.matcher(
            perception_graph,
            match_mode=MatchMode.NON_OBJECT,
            # debug_callback=self._debug_callback,
        )
        for match in matcher.matches(use_lookahead_pruning=True):
            # if there is a match, use that match to describe the situation.
            semantic_node_for_match = pattern_match_to_semantic_node(
                concept=concept, pattern=pattern, match=match
            )
            # A template only has to match once; we don't care about finding additional matches.
            return match, semantic_node_for_match
        return None
