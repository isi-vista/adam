"""
Interfaces for language learning code.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Generic, Mapping, Optional, Any

from adam.learner.alignments import (
    LanguagePerceptionSemanticAlignment,
    PerceptionSemanticAlignment,
)

from adam.ontology.ontology import Ontology
from attr import Factory, attrib, attrs
from attr.validators import instance_of
from immutablecollections import immutabledict
from more_itertools import first
from networkx import DiGraph, isolates

from adam.language import LinguisticDescription, LinguisticDescriptionT
from adam.ontology.phase1_ontology import LEARNER
from adam.perception import PerceptionT, PerceptualRepresentation, ObjectPerception
from adam.perception.perception_graph import (
    PerceptionGraph,
    PerceptionGraphPattern,
    DebugCallableType,
    PerceptionGraphPatternMatch,
    GraphLogger,
)
from adam.utils.networkx_utils import subgraph


@attrs(frozen=True)
class LearningExample(Generic[PerceptionT, LinguisticDescriptionT]):
    """
    A `PerceptualRepresentation` of a situation and its `LinguisticDescription`
    that a `LanguageLearner` can learn from.
    """

    # attrs can't check the generic types, so we just check the super-types
    perception: PerceptualRepresentation[PerceptionT] = attrib(  # type:ignore
        validator=instance_of(PerceptualRepresentation)
    )
    """
    The `LanguageLearner`'s perception of the `Situation`
    """
    linguistic_description: LinguisticDescriptionT = attrib(  # type:ignore
        validator=instance_of(LinguisticDescription)
    )
    """
    A human-language description of the `Situation`
    """


class TopLevelLanguageLearner(ABC, Generic[PerceptionT, LinguisticDescriptionT]):
    r"""
    Models an infant learning language.

    A `LanguageLearner` learns language by observing a sequence of `LearningExample`\ s.

    A `LanguageLearner` can describe new situations given a `PerceptualRepresentation`\ .
    """

    @abstractmethod
    def observe(
        self, learning_example: LearningExample[PerceptionT, LinguisticDescription]
    ) -> None:
        """
        Observe a `LearningExample`, possibly updating internal state.
        """

    @abstractmethod
    def describe(
        self, perception: PerceptualRepresentation[PerceptionT]
    ) -> Mapping[LinguisticDescription, float]:
        r"""
        Given a `PerceptualRepresentation` of a situation, produce one or more
        `LinguisticDescription`\ s of it.

        The descriptions are returned as a mapping from linguistic descriptions to their scores.
        The scores are not defined other than "higher is better."

        It is possible that the learner cannot produce a description, in which case an empty
        mapping is returned.
        """

    @abstractmethod
    def log_hypotheses(self, log_output_path: Path) -> None:
        """
        Log some representation of the learner's current hypothesized semantics for words/phrases
        to *log_output_path*
        """


@attrs
class MemorizingLanguageLearner(
    Generic[PerceptionT, LinguisticDescriptionT],
    TopLevelLanguageLearner[PerceptionT, LinguisticDescription],
):
    """
    A trivial implementation of `LanguageLearner` which just memorizes situations it has seen before
    and cannot produce descriptions of any other situations.

    If this learner observes the same `PerceptualRepresentation` multiple times, only the final
    description is memorized.

    This implementation is only useful for testing.
    """

    _memorized_situations: Dict[
        PerceptualRepresentation[PerceptionT], LinguisticDescription
    ] = attrib(init=False, default=Factory(dict))

    def observe(
        self, learning_example: LearningExample[PerceptionT, LinguisticDescription]
    ) -> None:
        self._memorized_situations[
            learning_example.perception
        ] = learning_example.linguistic_description

    def describe(
        self, perception: PerceptualRepresentation[PerceptionT]
    ) -> Mapping[LinguisticDescription, float]:
        memorized_description = self._memorized_situations.get(perception)
        if memorized_description:
            return immutabledict(((memorized_description, 1.0),))
        else:
            return immutabledict()

    def log_hypotheses(self, log_output_path: Path) -> None:
        pass


def get_largest_matching_pattern(
    pattern: PerceptionGraphPattern,
    graph: PerceptionGraph,
    *,
    debug_callback: Optional[DebugCallableType] = None,
    graph_logger: Optional[GraphLogger] = None,
    ontology: Ontology,
    match_ratio: Optional[float] = None,
    matching_objects: bool
) -> Optional[PerceptionGraphPattern]:
    """ Helper function to return the largest matching `PerceptionGraphPattern`
    for learner from a perception pattern and graph pair."""
    matching = pattern.matcher(
        graph, debug_callback=debug_callback, matching_objects=matching_objects
    )
    return matching.relax_pattern_until_it_matches(
        graph_logger=graph_logger, ontology=ontology, min_ratio=match_ratio
    )


def graph_without_learner(perception_graph: PerceptionGraph) -> PerceptionGraph:
    """ Helper function to return a `PerceptionGraph`
    without a ground object and its related nodes."""
    graph = perception_graph.copy_as_digraph()
    # Get the learner node
    learner_node_candidates = [
        node
        for node in graph.nodes()
        if isinstance(node, ObjectPerception) and node.debug_handle == LEARNER.handle
    ]
    if len(learner_node_candidates) > 1:
        raise RuntimeError("More than one learners in perception.")
    elif len(learner_node_candidates) == 1:
        learner_node = first(learner_node_candidates)
        # Remove learner
        graph.remove_node(learner_node)
        # remove remaining islands
        islands = list(isolates(graph))
        graph.remove_nodes_from(islands)
    return PerceptionGraph(graph)


class ComposableLearner(ABC):
    @abstractmethod
    def learn_from(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> None:
        """
        Learn from a `LanguagePerceptionSemanticAlignment` describing a situation. This may update
        some internal state.
        """

    @abstractmethod
    def enrich_during_learning(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> LanguagePerceptionSemanticAlignment:
        """
        Given a `LanguagePerceptionSemanticAlignment` wrapping a learning example, return such
        an updated alignment enriched with some extra semantic alignment information.

        The learner may have no information to add, in which case it can simply return the alignment
        it was passed.
        """

    @abstractmethod
    def enrich_during_description(
        self, perception_semantic_alignment: PerceptionSemanticAlignment
    ) -> PerceptionSemanticAlignment:
        """
        Given a `PerceptionSemanticAlignment` wrapping a perception to be described, return an
        updated alignment enriched with some extra semantic alignment information.

        The learner may have no information to add, in which case it can simply return the alignment
        it was passed.
        """

    @abstractmethod
    def log_hypotheses(self, log_output_path: Path) -> None:
        """
        Log some representation of the learner's current hypothesized semantics for words/phrases to
        *log_output_path*.
        """
