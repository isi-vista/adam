from typing import Dict, Generic, Mapping, Optional, Tuple

from immutablecollections import immutabledict
from more_itertools import first
from networkx import DiGraph, isolates

from adam.language import (
    LinguisticDescription,
    LinguisticDescriptionT,
    TokenSequenceLinguisticDescription,
)
from adam.learner import LanguageLearner, LearningExample
from adam.ontology.phase1_ontology import LEARNER
from adam.perception import ObjectPerception, PerceptionT, PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception.perception_graph import (
    DebugCallableType,
    PerceptionGraph,
    PerceptionGraphPattern,
    PerceptionGraphPatternMatch,
)
from attr import Factory, attrib, attrs


@attrs(slots=True)
class SubsetLanguageLearner(
    Generic[PerceptionT, LinguisticDescriptionT],
    LanguageLearner[PerceptionT, LinguisticDescription],
):
    """
    An implementation of `LanguageLearner` for subset learning based approach for single object detection.
    """

    _descriptions_to_pattern_hypothesis: Dict[
        Tuple[str, ...], PerceptionGraphPattern
    ] = attrib(init=False, default=Factory(dict))
    _debug_callback: Optional[DebugCallableType] = attrib(default=None, kw_only=True)

    def observe(
        self, learning_example: LearningExample[PerceptionT, LinguisticDescription]
    ) -> None:
        perception = learning_example.perception
        if len(perception.frames) != 1:
            raise RuntimeError("Subset learner can only handle single frames for now")
        if isinstance(perception.frames[0], DevelopmentalPrimitivePerceptionFrame):
            original_perception_graph = PerceptionGraph.from_frame(
                perception.frames[0]
            ).copy_as_digraph()
        else:
            raise RuntimeError("Cannot process perception type.")

        # Remove learner from the perception
        observed_perception_graph = graph_without_learner(original_perception_graph)
        observed_linguistic_description = (
            learning_example.linguistic_description.as_token_sequence()
        )

        if observed_linguistic_description in self._descriptions_to_pattern_hypothesis:
            # If already observed, get the largest matching subgraph of the pattern in the current observation and
            # previous pattern hypothesis
            # TODO: We should relax this requirement for learning: issue #361
            previous_pattern_hypothesis = self._descriptions_to_pattern_hypothesis[
                observed_linguistic_description
            ]

            # Get largest subgraph match using the pattern and the graph
            hypothesis_pattern_common_subgraph = get_largest_matching_pattern(
                previous_pattern_hypothesis,
                observed_perception_graph,
                debug_callback=self._debug_callback,
            )
            # Update the leading hypothesis
            self._descriptions_to_pattern_hypothesis[
                observed_linguistic_description
            ] = hypothesis_pattern_common_subgraph

        else:
            # If it's a new description, learn a new hypothesis/pattern, generated as a pattern graph frm the
            # perception graph.
            observed_pattern_graph = PerceptionGraphPattern.from_graph(
                observed_perception_graph.copy_as_digraph()
            ).perception_graph_pattern
            self._descriptions_to_pattern_hypothesis[
                observed_linguistic_description
            ] = observed_pattern_graph

    def describe(
        self, perception: PerceptualRepresentation[PerceptionT]
    ) -> Mapping[LinguisticDescription, float]:
        if len(perception.frames) != 1:
            raise RuntimeError("Subset learner can only handle single frames for now")
        if isinstance(perception.frames[0], DevelopmentalPrimitivePerceptionFrame):
            original_perception_graph = PerceptionGraph.from_frame(
                perception.frames[0]
            ).copy_as_digraph()
        else:
            raise RuntimeError("Cannot process perception type.")
        observed_perception_graph = graph_without_learner(original_perception_graph)

        # get the learned description for which there are the maximum number of matching properties (i.e. most specific)
        max_matching_subgraph_size = 0
        learned_description = None
        for (
            description,
            pattern_hypothesis,
        ) in self._descriptions_to_pattern_hypothesis.items():
            # get the largest common match
            common_pattern = get_largest_matching_pattern(
                pattern_hypothesis,
                observed_perception_graph,
                debug_callback=self._debug_callback,
            )
            common_pattern_size = len(common_pattern.copy_as_digraph().nodes)
            if common_pattern_size > max_matching_subgraph_size:
                learned_description = description
                max_matching_subgraph_size = common_pattern_size
        if learned_description:
            return immutabledict(
                ((TokenSequenceLinguisticDescription(learned_description), 1.0),)
            )
        else:
            return immutabledict()


def get_largest_matching_pattern(
    pattern: PerceptionGraphPattern,
    graph: PerceptionGraph,
    *,
    debug_callback: Optional[DebugCallableType] = None,
) -> PerceptionGraphPattern:
    """ Helper function to return the largest matching sub-pattern
    for a perception pattern and graph pair
    *which is discovered during the match search process*.
    This is not guaranteed to be the largest possible partial match.
    Improvement of this is pending https://github.com/isi-vista/adam/issues/461
    """
    matching = pattern.matcher(graph, debug_callback=debug_callback)
    attempted_match = matching.first_match_or_failure_info()

    if isinstance(attempted_match, PerceptionGraphPatternMatch):
        # if matched, get the match
        return attempted_match.matched_pattern
    else:
        return attempted_match.largest_match_pattern_subgraph


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
