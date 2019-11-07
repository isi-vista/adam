from typing import Dict, Generic, Mapping, Tuple, Any, Optional, Set, List

from attr import Factory, attrib, attrs
from immutablecollections import immutabledict, ImmutableDict, immutableset
from more_itertools import first
from networkx import DiGraph, isolates

from adam.language import (
    LinguisticDescriptionT,
    TokenSequenceLinguisticDescription,
    LinguisticDescription,
)
from adam.learner import LanguageLearner, LearningExample
from adam.learner.object_recognizer import ObjectRecognizer
from adam.ontology.phase1_ontology import LEARNER
from adam.perception import PerceptionT, PerceptualRepresentation, ObjectPerception
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception.perception_graph import (
    PerceptionGraph,
    PerceptionGraphPattern,
    PrepositionPattern,
    MatchedObjectPerceptionPredicate,
    PerceptionGraphNode,
    NodePredicate,
    DebugCallableType
)
from adam.perception_matcher._matcher import GraphMatching

def graph_without_learner(graph: DiGraph):
    # Get the learner node
    learner_node_candidates = [
        node
        for node in graph.nodes()
        if isinstance(node, ObjectPerception) and node.debug_handle == LEARNER.handle
    ]
    if learner_node_candidates:
        learner_node = first(learner_node_candidates)
        # Remove learner
        graph.remove_node(learner_node)
        # remove remaining islands
        islands = list(isolates(graph))
        graph.remove_nodes_from(islands)
    return graph

# Constants used to map locations in a prepositional phrase for mapping
_MODIFIED = "MODIFIED"
_GROUNDED = "GROUNDED"

PrepositionSurfaceTemplate = Tuple[str, ...]
"""
This is a surface string pattern for a preposition. 
It should contain the strings MODIFIED and GROUNDED as stand-ins for the particular words
a preposition may be used with. For example, "MODIFIED on a GROUNDED". 
"""


@attrs
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
    _debug_callback: Optional[DebugCallableType] = attrib(default=None)

    _descriptions_to_prepositions: Dict[PrepositionSurfaceTemplate, PrepositionPattern] = attrib(
        init=False, default=Factory(dict)
    )
    _object_recognizer: ObjectRecognizer = attrib(init=False, default=ObjectRecognizer())


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
            )
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

    # TODO: Remap the object recognizer to use perceived objects rather than the complete list
    # Eventually the perceived_objects map in the singature will get passed to the object recognizer rather than letting
    # the recognize use the complete default list.
    def _observe_static_prepositions(  # pylint:disable=unused-argument
        self,
        perception_graph: PerceptionGraph,
        linguistic_description: LinguisticDescription,
        perceived_objects: Optional[Mapping[str, PerceptionGraphPattern]] = None,
    ) -> None:
        observed_linguistic_description = linguistic_description.as_token_sequence()
        perception_graph_object_perception, object_handle_to_object_match_node = self._object_recognizer.match_objects(
            perception_graph
        )
        object_match_nodes = []
        token_indices_of_matched_object_words = []

        for (idx, token) in enumerate(observed_linguistic_description):
            if token in object_handle_to_object_match_node.keys():
                token_indices_of_matched_object_words.append(idx)
                object_match_nodes.append(object_handle_to_object_match_node[token])

        if len(object_match_nodes) != 2:
            raise RuntimeError(
                f"Learning a preposition with more than two recognized objects is not currently supported. "
                f"Found {len(object_match_nodes)} from {object_handle_to_object_match_node.keys()} and "
                f"{observed_linguistic_description}."
            )

        # If we have to reorder the bounds so that the smallest number is first we want the nodes to match ordering
        (
            token_offset_of_modified_word,
            token_offset_of_ground_word,
        ) = token_indices_of_matched_object_words
        if token_offset_of_modified_word < token_offset_of_ground_word:
            (
                object_match_node_for_modified,
                object_match_node_for_ground,
            ) = object_match_nodes
        else:
            # the matches are in the wrong order; we want to modifier ordered first
            # TODO: English-specific
            (token_offset_of_ground_word, token_offset_of_modified_word) = (
                token_offset_of_modified_word,
                token_offset_of_ground_word,
            )
            (
                object_match_node_for_ground,
                object_match_node_for_modified,
            ) = object_match_nodes

        # This is the lingustics description we learned
        prepositional_phrase_tokens = observed_linguistic_description[
            token_offset_of_modified_word : token_offset_of_ground_word + 1
        ]

        # for learning, we need to represent this in a way which abstracts
        # from the particular modified and ground word.
        preposition_surface_template = list(prepositional_phrase_tokens)
        preposition_surface_template[0] = _MODIFIED
        preposition_surface_template[-1] = _GROUNDED
        # we need these to be immutable after creation because we use them as dictionary keys.
        preposition_surface_template = tuple(preposition_surface_template)

        # This is the template_variables_to_object_match_nodes of sentence locations to pattern nodes
        template_variables_to_object_match_nodes: ImmutableDict[str, Any] = immutabledict(
            [
                (_MODIFIED, object_match_node_for_modified),
                (_GROUNDED, object_match_node_for_ground),
            ]
        )

        # The next step is to create a perception graph pattern to represent
        # the prepositional phrase semantics.
        # We take the potentially relevant parts of the perception to be
        # the object match nodes...
        nodes_for_preposition_pattern = list(object_match_nodes)
        #  and their adjacent nodes.
        nodes_for_preposition_pattern.extend(
            perception_graph_object_perception._graph.successors(  # pylint:disable=protected-access
                object_match_node_for_modified
            )
        )
        nodes_for_preposition_pattern.extend(
            perception_graph_object_perception._graph.successors(  # pylint:disable=protected-access
                object_match_node_for_ground
            )
        )
        preposition_pattern_graph = perception_graph_object_perception._graph.subgraph(  # pylint:disable=protected-access
            nodes=immutableset(nodes_for_preposition_pattern)
        )

        preposition_pattern = PrepositionPattern(
            graph_pattern=PerceptionGraphPattern(graph=preposition_pattern_graph),
            object_map=template_variables_to_object_match_nodes,
        )

        if preposition_surface_template in self._description_to_preposition_pattern:
            # We have seen this preposition situation before.
            # Our learning strategy is to assume the true semantics of the preposition
            # is what is in common between what we saw this time and what we saw last time.
            self._description_to_preposition_pattern[
                preposition_surface_template
            ] = self._description_to_preposition_pattern[
                preposition_surface_template
            ].intersection(
                preposition_pattern
            )
        else:
            # This is the first time we've seen a preposition situation like this one.
            # Remember our hypothesis about the semantics of the preposition.
            self._description_to_preposition_pattern[
                preposition_surface_template
            ] = preposition_pattern

    def _describe_preposition(  # pylint:disable=unused-argument
        self,
        perception_graph: PerceptionGraph,
        perceived_objects: Optional[Mapping[str, PerceptionGraphPattern]] = None,
    ) -> Mapping[LinguisticDescription, float]:
        # TODO: this might be clearer if the return were package in an object
        observed_perception_graph, description_to_node = self._object_recognizer.match_objects(  # pylint:disable=unused-variable
            perception_graph
        )
        # TODO: check if immutabledict has a method for inversion
        node_to_description: Mapping[
            PerceptionGraphNode, Tuple[str, ...]
        ] = immutabledict(
            (node, description) for description, node in description_to_node.items()
        )
        description_to_score: List[Tuple[TokenSequenceLinguisticDescription, float]] = []

        def replace_object_names(
            description: Tuple[str, ...],
            object_map: Mapping[str, MatchedObjectPerceptionPredicate],
            alignment: Mapping[NodePredicate, PerceptionGraphNode],
        ) -> Tuple[str, ...]:
            rtnr: List[str] = []
            for token in description:
                if token in object_map.keys():
                    rtnr.extend(node_to_description[alignment[object_map[token]]])
                else:
                    rtnr.append(token)
            return tuple(rtnr)

        for description, pattern in self._description_to_preposition_pattern.items():
            matcher = pattern.graph_pattern.matcher(observed_perception_graph)
            for match in matcher.matches():
                description_to_score.append(
                    (
                        TokenSequenceLinguisticDescription(
                            replace_object_names(
                                description, pattern.object_map, match.alignment
                            )
                        ),
                        1.0,
                    )
                )

        return immutabledict(description_to_score)


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
