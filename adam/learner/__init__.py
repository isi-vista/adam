"""
Interfaces for language learning code.
"""

from abc import ABC, abstractmethod
from typing import Dict, Generic, Mapping, Optional, Any

from attr import Factory, attrib, attrs
from attr.validators import instance_of
from immutablecollections import immutabledict
from more_itertools import first
from networkx import DiGraph, isolates

from adam.language import LinguisticDescription, LinguisticDescriptionT
from adam.ontology.phase1_ontology import LEARNER
from adam.perception import PerceptionT, PerceptualRepresentation, ObjectPerception
from adam.perception.perception_graph import PerceptionGraph, PerceptionGraphPattern, DebugCallableType


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


class LanguageLearner(ABC, Generic[PerceptionT, LinguisticDescriptionT]):
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


@attrs
class MemorizingLanguageLearner(
    Generic[PerceptionT, LinguisticDescriptionT],
    LanguageLearner[PerceptionT, LinguisticDescription],
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


def get_largest_matching_pattern(
    pattern: PerceptionGraphPattern,
    graph: PerceptionGraph,
    *,
    debug_callback: Optional[DebugCallableType] = None,
) -> PerceptionGraphPattern:
    """ Helper function to return the largest matching pattern for learner from a perception pattern and graph pair."""
    # Initialize matcher in debug version to keep largest subgraph
    matching = pattern.matcher(graph)
    debug_sink: Dict[Any, Any] = {}
    match_mapping = list(
        matching.matches(debug_mapping_sink=debug_sink, debug_callback=debug_callback)
    )

    if match_mapping:
        # if matched, get the match
        return match_mapping[0].matched_pattern
    else:
        # otherwise get the largest subgraph and initialze new PatternGraph from it
        matched_pattern_nodes = debug_sink.keys()
        matching_sub_digraph = pattern.copy_as_digraph().subgraph(matched_pattern_nodes)
        return PerceptionGraphPattern(matching_sub_digraph)


def graph_without_learner(graph: DiGraph) -> PerceptionGraph:
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
