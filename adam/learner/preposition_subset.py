from pathlib import Path
from typing import Any, Dict, Generic, List, Mapping, Tuple, Iterable

from immutablecollections import ImmutableSet, immutabledict, immutableset
from more_itertools import flatten
from networkx import all_shortest_paths

from adam.language import (
    LinguisticDescription,
    LinguisticDescriptionT,
    TokenSequenceLinguisticDescription,
)
from adam.learner import LanguageLearner, LearningExample
from adam.learner.object_recognizer import ObjectRecognizer
from adam.learner.preposition_pattern import PrepositionPattern, _GROUND, _MODIFIED
from adam.learner.subset import graph_without_learner
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
)
from attr import Factory, attrib, attrs

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
        # DEBUG CODE
        observed_perception_graph.render_to_file(
            "observed", Path(f"/nas/home/jacobl/adam-root/outputs/observed.pdf")
        )
        observed_linguistic_description = (
            learning_example.linguistic_description.as_token_sequence()
        )

        perception_graph_post_object_recognition, object_handle_to_object_match_node = self._object_recognizer.match_objects(
            observed_perception_graph
        )
        # DEBUG
        perception_graph_post_object_recognition.render_to_file(
            "with_objects", Path(f"/nas/home/jacobl/adam-root/outputs/with_objects.pdf")
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
            perception_graph_post_object_recognition,
            template_variables_to_object_match_nodes,
        )

        if preposition_surface_template in self._surface_template_to_preposition_pattern:
            # We have seen this preposition situation before.
            # Our learning strategy is to assume the true semantics of the preposition
            # is what is in common between what we saw this time and what we saw last time.
            self._surface_template_to_preposition_pattern[
                preposition_surface_template
            ] = self._surface_template_to_preposition_pattern[
                preposition_surface_template
            ].intersection(
                preposition_pattern
            )
        else:
            # This is the first time we've seen a preposition situation like this one.
            # Remember our hypothesis about the semantics of the preposition.
            self._surface_template_to_preposition_pattern[
                preposition_surface_template
            ] = preposition_pattern

        # DEBUG CODE TO BE REMOVED
        graph_name = "_".join(preposition_surface_template)
        self._surface_template_to_preposition_pattern[
            preposition_surface_template
        ].graph_pattern.render_to_file(
            graph_name, Path(f"/nas/home/jacobl/adam-root/outputs/preposition.pdf")
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
        perception_graph_as_undirected = perception_graph_post_object_recognition.copy_as_digraph().to_undirected(
            as_view=True
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

        preposition_pattern_graph = perception_graph_post_object_recognition.copy_as_digraph().subgraph(
            nodes=hypothesis_nodes
        )
        PerceptionGraph(graph=preposition_pattern_graph).render_to_file(
            "pattern", Path("/nas/home/jacobl/adam-root/outputs/pattern.pdf")
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
            original_perception_graph = PerceptionGraph.from_frame(
                perception.frames[0]
            ).copy_as_digraph()
        else:
            raise RuntimeError("Cannot process perception type.")
        observed_perception_graph = graph_without_learner(original_perception_graph)

        # TODO: this might be clearer if the return were package in an object
        perception_graph_with_object_matches, handle_to_object_match_node = self._object_recognizer.match_objects(
            observed_perception_graph
        )
        # TODO: check if immutabledict has a method for inversion
        object_match_node_to_object_handle: Mapping[
            PerceptionGraphNode, str
        ] = immutabledict(
            (node, description)
            for description, node in handle_to_object_match_node.items()
        )

        # DEBUG CODE TO REMOVE
        perception_graph_with_object_matches.render_to_file(
            "to_match", Path("/nas/home/jacobl/adam-root/outputs/to_match.pdf")
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
                perception_graph_with_object_matches
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
