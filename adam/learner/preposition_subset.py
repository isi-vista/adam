from typing import Any, Dict, Generic, List, Mapping, Tuple

from immutablecollections import ImmutableSet, immutabledict, immutableset

from adam.language import (
    LinguisticDescription,
    LinguisticDescriptionT,
    TokenSequenceLinguisticDescription,
)
from adam.learner import LanguageLearner, LearningExample
from adam.learner.object_recognizer import ObjectRecognizer
from adam.learner.preposition_pattern import PrepositionPattern, _GROUND, _MODIFIED
from adam.learner.subset import graph_without_learner
from adam.perception import PerceptionT, PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception.perception_graph import (
    MatchedObjectPerceptionPredicate,
    NodePredicate,
    PerceptionGraph,
    PerceptionGraphNode,
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
        observed_linguistic_description = (
            learning_example.linguistic_description.as_token_sequence()
        )

        perception_graph_object_perception, object_handle_to_object_match_node = self._object_recognizer.match_objects(
            observed_perception_graph
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

        # The next step is to create a perception graph pattern to represent
        # the prepositional phrase semantics.
        # We take the potentially relevant parts of the perception to be
        # the object match nodes...
        nodes_for_preposition_pattern = list(object_match_nodes)
        #  and their adjacent nodes.
        nodes_for_preposition_pattern.extend(
            perception_graph_object_perception.copy_as_digraph().successors(
                object_match_node_for_modified
            )
        )
        nodes_for_preposition_pattern.extend(
            perception_graph_object_perception.copy_as_digraph().successors(
                object_match_node_for_ground
            )
        )
        preposition_pattern_graph = perception_graph_object_perception.copy_as_digraph().subgraph(
            nodes=immutableset(nodes_for_preposition_pattern)
        )

        preposition_pattern = PrepositionPattern.from_graph(
            preposition_pattern_graph, template_variables_to_object_match_nodes
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
            # pylint:disable=unused-variable
            observed_perception_graph
        )
        # TODO: check if immutabledict has a method for inversion
        object_match_node_to_object_handle: Mapping[
            PerceptionGraphNode, str
        ] = immutabledict(
            (node, description)
            for description, node in handle_to_object_match_node.items()
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
