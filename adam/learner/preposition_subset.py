import logging
from typing import Any, Dict, Generic, List, Mapping, Tuple, Iterable, Optional

from immutablecollections import ImmutableSet, immutabledict, immutableset
from more_itertools import flatten
from networkx import all_shortest_paths, DiGraph

from adam.language import (
    LinguisticDescription,
    LinguisticDescriptionT,
    TokenSequenceLinguisticDescription,
)
from adam.learner import LanguageLearner, LearningExample
from adam.learner.object_recognizer import ObjectRecognizer
from adam.learner.preposition_pattern import PrepositionPattern, _GROUND, _MODIFIED
from adam.perception import PerceptionT, PerceptualRepresentation, ObjectPerception
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception.perception_graph import (
    MatchedObjectPerceptionPredicate,
    NodePredicate,
    PerceptionGraph,
    PerceptionGraphNode,
    MatchedObjectNode,
    _graph_node_order,
    GraphLogger,
)
from attr import Factory, attrib, attrs

from adam.utils.networkx_utils import digraph_with_nodes_sorted_by, subgraph

PrepositionSurfaceTemplate = Tuple[str, ...]
"""
This is a surface string pattern for a preposition. 
It should contain the strings MODIFIED and GROUND as stand-ins for the particular words
a preposition may be used with. For example, "MODIFIED on a GROUND". 
"""


@attrs
class PrepositionSubsetLanguageLearner(
    Generic[PerceptionT, LinguisticDescriptionT],
    LanguageLearner[PerceptionT, LinguisticDescription],
):
    _surface_template_to_preposition_pattern: Dict[
        PrepositionSurfaceTemplate, PrepositionPattern
    ] = attrib(init=False, default=Factory(dict))

    _object_recognizer: ObjectRecognizer = attrib(init=False, default=ObjectRecognizer())
    _debug_file: Optional[str] = attrib(kw_only=True, default=None)
    _graph_logger: Optional[GraphLogger] = attrib(default=None)

    def _print(self, graph: DiGraph, name: str) -> None:
        if self._debug_file:
            with open(self._debug_file, "a") as doc:
                doc.write("=== " + name + "===\n")
                for node in graph:
                    doc.write("\t" + str(node) + "\n")
                for edge in graph.edges:
                    doc.write("\t" + str(edge) + "\n")

    def observe(
        self, learning_example: LearningExample[PerceptionT, LinguisticDescription]
    ) -> None:
        perception = learning_example.perception
        if len(perception.frames) != 1:
            raise RuntimeError("Subset learner can only handle single frames for now")
        if isinstance(perception.frames[0], DevelopmentalPrimitivePerceptionFrame):
            original_perception = PerceptionGraph.from_frame(perception.frames[0])
        else:
            raise RuntimeError("Cannot process perception type.")

        if self._graph_logger:
            self._graph_logger.log_graph(
                original_perception,
                logging.INFO,
                "*** Preposition subset learner observing %s",
                learning_example.linguistic_description,
            )

        observed_linguistic_description = (
            learning_example.linguistic_description.as_token_sequence()
        )

        # DEBUG
        self._print(
            original_perception.copy_as_digraph(),
            " ".join(observed_linguistic_description),
        )

        recognized_object_perception = self._object_recognizer.match_objects(
            original_perception
        )

        if self._graph_logger:
            self._graph_logger.log_graph(
                recognized_object_perception.perception_graph,
                logging.INFO,
                "Perception post-object-recognition",
            )

        object_match_nodes = []
        token_indices_of_matched_object_words = []

        # DEBUG
        self._print(
            recognized_object_perception.perception_graph.copy_as_digraph(),
            "Recognized Perception",
        )

        for (idx, token) in enumerate(observed_linguistic_description):
            if (
                token
                in recognized_object_perception.description_to_matched_object_node.keys()
            ):
                logging.info("Aligned word %s to a recognized object", token)
                token_indices_of_matched_object_words.append(idx)
                object_match_nodes.append(
                    recognized_object_perception.description_to_matched_object_node[token]
                )

        if len(object_match_nodes) != 2:
            raise RuntimeError(
                f"Learning a preposition with more than two recognized objects is not currently supported. "
                f"Found {len(object_match_nodes)} from {recognized_object_perception.description_to_matched_object_node.keys()} and "
                f"{observed_linguistic_description}."
            )

        # If we have to reorder the bounds so that the smallest number is first we want the nodes to match ordering
        (  # pylint:disable=unbalanced-tuple-unpacking
            token_offset_of_modified_word,
            token_offset_of_ground_word,
        ) = token_indices_of_matched_object_words
        if token_offset_of_modified_word < token_offset_of_ground_word:
            (  # pylint:disable=unbalanced-tuple-unpacking
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
            (  # pylint:disable=unbalanced-tuple-unpacking
                object_match_node_for_ground,
                object_match_node_for_modified,
            ) = object_match_nodes

        # This is the lingustics description we learned
        prepositional_phrase_tokens = observed_linguistic_description[
            token_offset_of_modified_word : token_offset_of_ground_word + 1
        ]

        # for learning, we need to represent this in a way which abstracts
        # from the particular modified and ground word.
        preposition_surface_template_mutable = list(prepositional_phrase_tokens)
        preposition_surface_template_mutable[0] = _MODIFIED
        preposition_surface_template_mutable[-1] = _GROUND
        # TODO: Remove this hard coded insert of an article
        # see: https://github.com/isi-vista/adam/issues/434
        preposition_surface_template_mutable.insert(0, "a")
        # we need these to be immutable after creation because we use them as dictionary keys.
        preposition_surface_template = tuple(preposition_surface_template_mutable)

        logging.info("Identified preposition template: %s", preposition_surface_template)

        # This is the template_variables_to_object_match_nodes of sentence locations to pattern nodes
        template_variables_to_object_match_nodes: ImmutableSet[
            Tuple[str, Any]
        ] = immutableset(
            [
                (_MODIFIED, object_match_node_for_modified),
                (_GROUND, object_match_node_for_ground),
            ]
        )

        preposition_pattern = self._make_preposition_hypothesis(
            object_match_node_for_ground,
            object_match_node_for_modified,
            recognized_object_perception.perception_graph,
            template_variables_to_object_match_nodes,
        )

        if self._graph_logger:
            self._graph_logger.log_graph(
                preposition_pattern.graph_pattern,
                logging.INFO,
                "Preposition hypothesis from current perception",
            )
        # DEBUG
        self._print(
            preposition_pattern.graph_pattern.copy_as_digraph(),
            "New: " + " ".join(preposition_surface_template),
        )

        if preposition_surface_template in self._surface_template_to_preposition_pattern:
            # We have seen this preposition situation before.
            # Our learning strategy is to assume the true semantics of the preposition
            # is what is in common between what we saw this time and what we saw last time.
            previous_hypothesis = self._surface_template_to_preposition_pattern[
                preposition_surface_template
            ].graph_pattern
            if self._graph_logger:
                self._graph_logger.log_graph(
                    previous_hypothesis,
                    logging.INFO,
                    "We have seen this preposition template before. Here is our hypothesis "
                    "prior to seeing this example",
                )

            previous_hypothesis_as_digraph = previous_hypothesis.copy_as_digraph()
            self._print(previous_hypothesis_as_digraph, "pre-intersection current: ")
            self._print(
                preposition_pattern.graph_pattern.copy_as_digraph(),
                "pre-intersection intersecting",
            )
            new_hypothesis = self._surface_template_to_preposition_pattern[
                preposition_surface_template
            ].intersection(preposition_pattern)

            if new_hypothesis:
                self._surface_template_to_preposition_pattern[
                    preposition_surface_template
                ] = new_hypothesis

                if self._graph_logger:
                    self._graph_logger.log_graph(
                        new_hypothesis.graph_pattern, logging.INFO, "New hypothesis"
                    )

                self._print(
                    self._surface_template_to_preposition_pattern[
                        preposition_surface_template
                    ].graph_pattern.copy_as_digraph(),
                    "post-intersection: ",
                )
            else:
                raise RuntimeError("Intersection is the empty pattern :-(")
        else:
            logging.info(
                "This is a new preposition template; accepting current hypothesis"
            )
            # This is the first time we've seen a preposition situation like this one.
            # Remember our hypothesis about the semantics of the preposition.
            self._surface_template_to_preposition_pattern[
                preposition_surface_template
            ] = preposition_pattern

        # DEBUG
        self._print(
            self._surface_template_to_preposition_pattern[
                preposition_surface_template
            ].graph_pattern.copy_as_digraph(),
            "Saved: " + " ".join(preposition_surface_template),
        )

    def _make_preposition_hypothesis(
        self,
        object_match_node_for_ground: MatchedObjectNode,
        object_match_node_for_modified: MatchedObjectNode,
        perception_graph_post_object_recognition: PerceptionGraph,
        template_variables_to_object_match_nodes: Iterable[Tuple[str, MatchedObjectNode]],
    ) -> PrepositionPattern:
        """
        Create a hypothesis for the semantics of a preposition based on the observed scene.
        
        Our current implementation is to just include the content 
        on the path between the recognized object nodes.
        """
        # The directions of edges in the perception graph are not necessarily meaningful
        # from the point-of-view of hypothesis generation, so we need an undirected copy
        # of the graph.
        perception_graph = perception_graph_post_object_recognition.copy_as_digraph()
        self._print(perception_graph, "pre-Undirected perception graph")

        # as_view=True loses determinism
        perception_graph_as_undirected = perception_graph_post_object_recognition.copy_as_digraph().to_undirected(
            as_view=False
        )

        self._print(perception_graph_as_undirected, "Undirected perception graph")

        for path in all_shortest_paths(
            perception_graph_as_undirected,
            object_match_node_for_ground,
            object_match_node_for_modified,
        ):
            self._print(
                subgraph(perception_graph_as_undirected, nodes=path), "Got a path!"
            )

        # The core of our hypothesis for the semantics of a preposition is all nodes
        # along the shortest path between the two objects involved in the perception graph.
        hypothesis_spine_nodes: ImmutableSet[PerceptionGraphNode] = immutableset(
            flatten(
                # if there are multiple paths between the object match nodes,
                # we aren't sure which are relevant, so we include them all in our hypothesis
                # and figure we can trim out irrelevant stuff as we make more observations.
                all_shortest_paths(
                    perception_graph_as_undirected,
                    object_match_node_for_ground,
                    object_match_node_for_modified,
                )
            )
        )

        self._print(
            subgraph(perception_graph_as_undirected, nodes=hypothesis_spine_nodes),
            "Spine nodes",
        )

        # Along the core of our hypothesis we also want to collect the predecessors and successors
        hypothesis_nodes_mutable = []
        for node in hypothesis_spine_nodes:
            if node not in [object_match_node_for_ground, object_match_node_for_modified]:
                for successor in perception_graph.successors(node):
                    if not isinstance(successor, ObjectPerception):
                        hypothesis_nodes_mutable.append(successor)
                for predecessor in perception_graph.predecessors(node):
                    if not isinstance(predecessor, ObjectPerception):
                        hypothesis_nodes_mutable.append(predecessor)

        hypothesis_nodes_mutable.extend(hypothesis_spine_nodes)

        # We wrap the nodes in an immutable set to remove duplicates
        hypothesis_nodes = immutableset(hypothesis_nodes_mutable)

        preposition_pattern_graph = digraph_with_nodes_sorted_by(
            subgraph(
                perception_graph_post_object_recognition.copy_as_digraph(),
                nodes=hypothesis_nodes,
            ),
            _graph_node_order,
        )
        return PrepositionPattern.from_graph(
            preposition_pattern_graph, template_variables_to_object_match_nodes
        )

    def describe(
        self, perception: PerceptualRepresentation[PerceptionT]
    ) -> Mapping[LinguisticDescription, float]:
        if len(perception.frames) != 1:
            raise RuntimeError("Subset learner can only handle single frames for now")
        if isinstance(perception.frames[0], DevelopmentalPrimitivePerceptionFrame):
            original_perception = PerceptionGraph.from_frame(perception.frames[0])
        else:
            raise RuntimeError("Cannot process perception type.")

        # DEBGU
        self._print(original_perception.copy_as_digraph(), "Describe Original")

        recognized_object_perception = self._object_recognizer.match_objects(
            original_perception
        )

        # DEBGU
        self._print(
            recognized_object_perception.perception_graph.copy_as_digraph(),
            "Describe Recoginzed Objects",
        )

        object_match_node_to_object_handle: Mapping[
            PerceptionGraphNode, str
        ] = immutabledict(
            (node, description)
            for description, node in recognized_object_perception.description_to_matched_object_node.items()
        )

        # this will be our output
        description_to_score: List[Tuple[TokenSequenceLinguisticDescription, float]] = []

        def replace_template_variables_with_object_names(
            preposition_surface_template: Tuple[str, ...],
            object_variable_name_to_object_match_pattern_node: Mapping[
                str, MatchedObjectPerceptionPredicate
            ],
            pattern_node_to_aligned_perception_node: Mapping[
                NodePredicate, PerceptionGraphNode
            ],
        ) -> Tuple[str, ...]:
            rtnr: List[str] = []
            # each entry in a preposition surface object_match_node is either a token
            # (typically a preposition) or one of the two placeholders
            # MODIFIED and GROUND
            for token_or_surface_template_variable in preposition_surface_template:
                if (
                    token_or_surface_template_variable
                    in object_variable_name_to_object_match_pattern_node.keys()
                ):
                    # If we have a placeholder, we need to figure out what object should
                    # fill it in this particular situation.

                    # This will be either MODIFIED or GROUND
                    surface_template_variable = token_or_surface_template_variable
                    # Get the corresponding variable in the preposition perception pattern.
                    object_match_variable_node = object_variable_name_to_object_match_pattern_node[
                        surface_template_variable
                    ]
                    # This variable should have matched against an object that we recognized
                    # with the object matcher, which would have introduced an object_match_node
                    object_match_node = pattern_node_to_aligned_perception_node[
                        object_match_variable_node
                    ]
                    # and for each of these object matches, we were provided with a name,
                    # which is what we use in the linguistic description.
                    rtnr.append(object_match_node_to_object_handle[object_match_node])
                else:
                    # tokens are just copied directly to the description
                    token = token_or_surface_template_variable
                    rtnr.append(token)
            return tuple(rtnr)

        # For each preposition we've ever seen...
        for (
            preposition_surface_template,
            preposition_pattern,
        ) in self._surface_template_to_preposition_pattern.items():
            # try to see if (our model of) its semantics is present in the situation.
            matcher = preposition_pattern.graph_pattern.matcher(
                recognized_object_perception.perception_graph
            )
            for match in matcher.matches():
                # if it is, use that preposition to describe the situation.
                description_to_score.append(
                    (
                        TokenSequenceLinguisticDescription(
                            # we generate the description by taking the preposition surface template
                            # which has MODIFIER and GROUND variables,
                            # and replacing those variables by the actual names
                            # of the matched objects.
                            replace_template_variables_with_object_names(
                                preposition_surface_template,
                                preposition_pattern.object_variable_name_to_pattern_node,
                                match.pattern_node_to_matched_graph_node,
                            )
                        ),
                        1.0,
                    )
                )

        return immutabledict(description_to_score)
