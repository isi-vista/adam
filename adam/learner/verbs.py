import logging
from abc import ABC
from typing import Iterable, List, Optional, Tuple, Union

from attr.validators import instance_of, optional
from networkx import DiGraph

from adam.language import LinguisticDescription
from adam.learner import LearningExample, get_largest_matching_pattern
from adam.learner.object_recognizer import (
    ObjectRecognizer,
    PerceptionGraphFromObjectRecognizer,
)
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.pursuit import AbstractPursuitLearner
from adam.learner.subset import AbstractSubsetLearner
from adam.learner.surface_templates import STANDARD_SLOT_VARIABLES, SurfaceTemplate
from adam.learner.template_learner import AbstractTemplateLearner
from adam.learner.verb_pattern import VerbPattern, _AGENT, _PATIENT
from adam.perception import PerceptionT, PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception.perception_graph import (
    LanguageAlignedPerception,
    MatchedObjectNode,
    PerceptionGraph,
    PerceptionGraphPattern,
    TemporalScope,
)
from attr import attrib, attrs
from immutablecollections import immutabledict, immutableset


@attrs
class AbstractVerbTemplateLearner(AbstractTemplateLearner, ABC):
    # mypy doesn't realize that fields without defaults can come after those with defaults
    # if they are keyword-only.
    _object_recognizer: ObjectRecognizer = attrib(  # type: ignore
        validator=instance_of(ObjectRecognizer), kw_only=True
    )

    def _assert_valid_input(
        self,
        to_check: Union[
            PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame],
            LearningExample[DevelopmentalPrimitivePerceptionFrame, LinguisticDescription],
        ],
    ) -> None:
        if isinstance(to_check, LearningExample):
            perception = to_check.perception
        else:
            perception = to_check
        if len(perception.frames) != 2:
            raise RuntimeError(
                "Expected exactly two frames in a perception for verb learning"
            )

    def _extract_perception_graph(
        self, perception: PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame]
    ) -> PerceptionGraph:
        return PerceptionGraph.from_dynamic_perceptual_representation(perception)

    def _preprocess_scene_for_learning(
        self, language_aligned_perception: LanguageAlignedPerception
    ) -> LanguageAlignedPerception:
        post_recognition_object_perception_alignment = self._object_recognizer.match_objects_with_language(
            language_aligned_perception
        )
        return post_recognition_object_perception_alignment

    def _preprocess_scene_for_description(
        self, perception_graph: PerceptionGraph
    ) -> PerceptionGraphFromObjectRecognizer:
        return self._object_recognizer.match_objects(perception_graph)

    def _extract_surface_template(
        self, preprocessed_input: LanguageAlignedPerception
    ) -> SurfaceTemplate:
        if len(preprocessed_input.aligned_nodes) > len(STANDARD_SLOT_VARIABLES):
            raise RuntimeError("Input has too many aligned nodes for us to handle.")

        return SurfaceTemplate.from_language_aligned_perception(
            preprocessed_input,
            object_node_to_template_variable=immutabledict(
                zip(preprocessed_input.aligned_nodes, STANDARD_SLOT_VARIABLES)
            ),
        )


@attrs
class SubsetVerbLearner(AbstractSubsetLearner, AbstractVerbTemplateLearner):
    def _hypothesis_from_perception(
        self, preprocessed_input: LanguageAlignedPerception
    ) -> PerceptionGraphTemplate:
        return PerceptionGraphTemplate.from_graph(
            preprocessed_input.perception_graph,
            template_variable_to_matched_object_node=immutabledict(
                zip(STANDARD_SLOT_VARIABLES, preprocessed_input.aligned_nodes)
            ),
        )


class VerbPursuitLearner(AbstractPursuitLearner):
    """
    An implementation of pursuit learner for learning verb semantics
    """

    # Variables for tracking verb phrase information. These are filled in observe.
    object_match_node_for_agent: Optional[MatchedObjectNode] = None
    object_match_node_for_patient: Optional[MatchedObjectNode] = None
    object_match_node_for_goal: Optional[MatchedObjectNode] = None
    object_match_node_for_theme: Optional[MatchedObjectNode] = None
    object_match_node_for_instrument: Optional[MatchedObjectNode] = None
    template_variables_to_object_match_nodes: Optional[
        Iterable[Tuple[str, MatchedObjectNode]]
    ] = None

    def observe(
        self,
        learning_example: LearningExample[PerceptionT, LinguisticDescription],
        object_recognizer: Optional[ObjectRecognizer] = None,
    ) -> None:
        perception = learning_example.perception
        if len(perception.frames) != 2:
            raise RuntimeError("Verb learner can only handle double-frame perceptions")
        if isinstance(perception.frames[0], DevelopmentalPrimitivePerceptionFrame):
            original_perception = PerceptionGraph.from_dynamic_perceptual_representation(  # type: ignore
                perception
            )
        else:
            raise RuntimeError("Cannot process perception type.")
        if not object_recognizer:
            raise RuntimeError("Verb learner is missing object recognizer")

        observed_linguistic_description = tuple(
            [
                t.lower()
                for t in learning_example.linguistic_description.as_token_sequence()
            ]
        )

        # Convert the observed perception to a version with recognized objects
        recognized_object_perception = object_recognizer.match_objects_with_language(
            original_perception
        )

        # Get the match nodes and their word indices
        token_idx_of_words_to_object_match_nodes = {}
        for (idx, token) in enumerate(observed_linguistic_description):
            if (
                token
                in recognized_object_perception.description_to_matched_object_node.keys()
            ):
                token_idx_of_words_to_object_match_nodes[
                    idx
                ] = recognized_object_perception.description_to_matched_object_node[token]

        # if we have one, assume it's the agent
        # if we have two or more, assume the phrase is between the agent and the next item
        # TODO: The current approach is english specific.
        sorted_indices = sorted(token_idx_of_words_to_object_match_nodes.keys())
        agent_idx = sorted_indices[0]

        # This is the lingustics description we learned
        if len(sorted_indices) > 1:
            verb_phrase_tokens = observed_linguistic_description[
                agent_idx : sorted_indices[1] + 1
            ]
        else:
            verb_phrase_tokens = observed_linguistic_description[agent_idx:]

        # TODO we need to come up with a syntactically intelligent way of parsing other positions
        verb_surface_template_mutable = list(verb_phrase_tokens)
        verb_surface_template_mutable[0] = _AGENT
        verb_surface_template_mutable[-1] = _PATIENT

        # we need these to be immutable after creation because we use them as dictionary keys.
        verb_surface_template = tuple(verb_surface_template_mutable)
        logging.info("Identified verb template: %s", verb_surface_template)

        self.object_match_node_for_agent = token_idx_of_words_to_object_match_nodes[
            sorted_indices[0]
        ]
        if len(sorted_indices) > 1:
            self.object_match_node_for_patient = token_idx_of_words_to_object_match_nodes[
                sorted_indices[1]
            ]

        # This is the template_variables_to_object_match_nodes of sentence locations to pattern nodes
        vars_to_nodes_list: List[Tuple[str, MatchedObjectNode]] = [
            (_AGENT, self.object_match_node_for_agent)
        ]
        if len(sorted_indices) > 1:
            if self.object_match_node_for_patient is not None:
                vars_to_nodes_list.append((_PATIENT, self.object_match_node_for_patient))

        self.template_variables_to_object_match_nodes = immutableset(vars_to_nodes_list)
        self.learn_with_pursuit(
            observed_perception_graph=recognized_object_perception.perception_graph,
            items_to_learn=(verb_surface_template,),
        )

    def _candidate_perceptions(
        self, observed_perception_graph: PerceptionGraph
    ) -> List[PerceptionGraph]:
        # TODO: Discuss which part of the graph is relevant for verbs
        # For now, we are extracting the part of the graph that changes (i.e edges that are marked only before,
        # during, or after)
        # This will be the graph representing the candidate:
        difference_digraph = DiGraph()

        # We add any edge that might be marking a change, i.e anything that doesn't have both AFTER
        # and BEFORE in it.
        # TODO add DURING once we implement it
        for (
            source,
            target,
            label,
        ) in observed_perception_graph.copy_as_digraph().edges.data("label"):
            if not (
                TemporalScope.AFTER in label.temporal_specifiers
                and TemporalScope.BEFORE in label.temporal_specifiers
            ):

                difference_digraph.add_edge(source, target, label=label)

        return [PerceptionGraph(graph=difference_digraph, dynamic=True)]

    def _hypothesis_from_perception(self, perception: PerceptionGraph) -> VerbPattern:
        if not perception.dynamic:
            raise RuntimeError("Perception for verb must be dynamic")
        if not self.template_variables_to_object_match_nodes:
            raise RuntimeError(
                "Empty template variables:", self.template_variables_to_object_match_nodes
            )
        return VerbPattern.from_graph(
            perception.copy_as_digraph(), self.template_variables_to_object_match_nodes
        )

    @attrs(frozen=True)
    class VerbHypothesisPartialMatch(AbstractPursuitLearner.PartialMatch):
        partial_match_hypothesis: Optional[VerbPattern] = attrib(
            validator=optional(instance_of(VerbPattern))
        )
        num_nodes_matched: int = attrib(validator=instance_of(int), kw_only=True)
        num_nodes_in_pattern: int = attrib(validator=instance_of(int), kw_only=True)

        def matched_exactly(self) -> bool:
            return self.num_nodes_matched == self.num_nodes_in_pattern

        def match_score(self) -> float:
            return self.num_nodes_matched / self.num_nodes_in_pattern

    def _find_partial_match(
        self, hypothesis: VerbPattern, graph: PerceptionGraph
    ) -> "VerbPursuitLearner.VerbHypothesisPartialMatch":
        pattern = hypothesis.graph_pattern
        hypothesis_pattern_common_subgraph = get_largest_matching_pattern(
            pattern,
            graph,
            debug_callback=self._debug_callback,
            graph_logger=self._hypothesis_logger,
            ontology=self._ontology,
            matching_objects=True,
        )
        self.debug_counter += 1

        leading_hypothesis_num_nodes = len(pattern)
        num_nodes_matched = (
            len(hypothesis_pattern_common_subgraph.copy_as_digraph().nodes)
            if hypothesis_pattern_common_subgraph
            else 0
        )
        if hypothesis_pattern_common_subgraph:
            partial_hypothesis: Optional[VerbPattern] = VerbPattern(
                graph_pattern=hypothesis_pattern_common_subgraph,
                object_variable_name_to_pattern_node=hypothesis.object_variable_name_to_pattern_node,
            )
        else:
            partial_hypothesis = None

        return VerbPursuitLearner.VerbHypothesisPartialMatch(
            partial_hypothesis,
            num_nodes_matched=num_nodes_matched,
            num_nodes_in_pattern=leading_hypothesis_num_nodes,
        )

    def _find_identical_hypothesis(
        self, new_hypothesis: VerbPattern, candidates: Iterable[VerbPattern]
    ) -> Optional[VerbPattern]:
        """
        Finds the first hypothesis object, if any, in *candidates*
        which is isomorphic to *new_hypothesis*.
        """
        for candidate in candidates:
            if self._are_isomorphic(new_hypothesis, candidate):
                return candidate
        return None

    def _are_isomorphic(self, h: VerbPattern, hypothesis: VerbPattern) -> bool:
        # Check mapping equality of verb patterns
        first_mapping = h.object_variable_name_to_pattern_node
        second_mapping = hypothesis.object_variable_name_to_pattern_node
        are_equal_mappings = len(first_mapping) == len(second_mapping) and all(
            k in second_mapping and second_mapping[k].is_equivalent(v)
            for k, v in first_mapping.items()
        )
        return are_equal_mappings and h.graph_pattern.check_isomorphism(
            hypothesis.graph_pattern
        )
