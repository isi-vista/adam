from typing import Iterable, Optional, Sequence, Union

from attr.validators import instance_of, optional

from adam.language import LinguisticDescription, TokenSequenceLinguisticDescription
from adam.learner import LearningExample, get_largest_matching_pattern
from adam.learner.object_recognizer import (
    ObjectRecognizer,
    PerceptionGraphFromObjectRecognizer,
)
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.prepositions import preposition_hypothesis_from_perception
from adam.learner.pursuit import AbstractPursuitLearner
from adam.learner.surface_templates import (
    SLOT1,
    SLOT2,
    SurfaceTemplate,
    SurfaceTemplateVariable,
)
from adam.perception import PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception.perception_graph import (
    LanguageAlignedPerception,
    PerceptionGraph,
    MatchedObjectNode,
)
from attr import attrib, attrs
from immutablecollections import immutabledict, ImmutableDict


class PrepositionPursuitLearner(
    AbstractPursuitLearner[
        DevelopmentalPrimitivePerceptionFrame, TokenSequenceLinguisticDescription
    ]
):
    """
    An implementation of pursuit learner for preposition leaning
    """

    _object_recognizer: ObjectRecognizer = attrib(validator=instance_of(ObjectRecognizer))

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

        if len(perception.frames) != 1:
            raise RuntimeError("Pursuit learner can only handle single frames for now")
        if not isinstance(perception.frames[0], DevelopmentalPrimitivePerceptionFrame):
            raise RuntimeError(f"Cannot process frame type: {type(perception.frames[0])}")

    def _extract_perception_graph(
        self, perception: PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame]
    ) -> PerceptionGraph:
        return PerceptionGraph.from_frame(perception.frames[0])

    def _preprocess_scene_for_learning(
        self, language_aligned_perception: LanguageAlignedPerception
    ) -> LanguageAlignedPerception:
        post_recognition_object_perception_alignment = self._object_recognizer.match_objects_with_language(
            language_aligned_perception
        )
        num_matched_objects = len(
            post_recognition_object_perception_alignment.node_to_language_span
        )
        if num_matched_objects != 2:
            raise RuntimeError(
                f"Learning a preposition with more than two recognized objects "
                f"is not currently supported. Found {num_matched_objects} for "
                f"{language_aligned_perception.language}."
            )
        return post_recognition_object_perception_alignment

    def _preprocess_scene_for_description(
        self, perception_graph: PerceptionGraph
    ) -> PerceptionGraphFromObjectRecognizer:
        return self._object_recognizer.match_objects(perception_graph)

    def _extract_surface_template(
        self, preprocessed_input: LanguageAlignedPerception
    ) -> SurfaceTemplate:
        return SurfaceTemplate.from_language_aligned_perception(
            preprocessed_input,
            object_node_to_template_variable=immutabledict(
                [
                    (preprocessed_input.aligned_nodes[0], SLOT1),
                    (preprocessed_input.aligned_nodes[1], SLOT2),
                ]
            ),
        )

    def _candidate_hypotheses(
        self, language_aligned_perception: LanguageAlignedPerception
    ) -> Sequence[PerceptionGraphTemplate]:
        # We represent prepositions as regex-like templates over the surface strings.
        # As an English-specific hack, the leftmost recognized object
        # is always taken to be the object modified, and the right one the ground.
        template_variables_to_object_match_nodes: ImmutableDict[
            SurfaceTemplateVariable, MatchedObjectNode
        ] = immutabledict(
            [
                (SLOT1, language_aligned_perception.aligned_nodes[0]),
                (SLOT2, language_aligned_perception.aligned_nodes[1]),
            ]
        )

        return [
            preposition_hypothesis_from_perception(
                language_aligned_perception,
                template_variables_to_object_match_nodes=template_variables_to_object_match_nodes,
            )
        ]

    @attrs(frozen=True)
    class PrepositionHypothesisPartialMatch(AbstractPursuitLearner.PartialMatch):
        partial_match_hypothesis: Optional[PerceptionGraphTemplate] = attrib(
            validator=optional(instance_of(PerceptionGraphTemplate))
        )
        num_nodes_matched: int = attrib(validator=instance_of(int), kw_only=True)
        num_nodes_in_pattern: int = attrib(validator=instance_of(int), kw_only=True)

        def matched_exactly(self) -> bool:
            return self.num_nodes_matched == self.num_nodes_in_pattern

        def match_score(self) -> float:
            return self.num_nodes_matched / self.num_nodes_in_pattern

    def _find_partial_match(
        self, hypothesis: PerceptionGraphTemplate, graph: PerceptionGraph
    ) -> "PrepositionPursuitLearner.PrepositionHypothesisPartialMatch":
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
            partial_hypothesis: Optional[
                PerceptionGraphTemplate
            ] = PerceptionGraphTemplate(
                graph_pattern=hypothesis_pattern_common_subgraph,
                object_variable_name_to_pattern_node=hypothesis.template_variable_to_pattern_node,
            )
        else:
            partial_hypothesis = None

        return PrepositionPursuitLearner.PrepositionHypothesisPartialMatch(
            partial_hypothesis,
            num_nodes_matched=num_nodes_matched,
            num_nodes_in_pattern=leading_hypothesis_num_nodes,
        )

    def _find_identical_hypothesis(
        self,
        new_hypothesis: PerceptionGraphTemplate,
        candidates: Iterable[PerceptionGraphTemplate],
    ) -> Optional[PerceptionGraphTemplate]:
        """
        Finds the first hypothesis object, if any, in *candidates*
        which is isomorphic to *new_hypothesis*.
        """
        for candidate in candidates:
            if self._are_isomorphic(new_hypothesis, candidate):
                return candidate
        return None

    def _are_isomorphic(
        self, h: PerceptionGraphTemplate, hypothesis: PerceptionGraphTemplate
    ) -> bool:
        # Check mapping equality of preposition patterns
        first_mapping = h.template_variable_to_pattern_node
        second_mapping = hypothesis.template_variable_to_pattern_node
        are_equal_mappings = len(first_mapping) == len(second_mapping) and all(
            k in second_mapping and second_mapping[k].is_equivalent(v)
            for k, v in first_mapping.items()
        )
        return are_equal_mappings and h.graph_pattern.check_isomorphism(
            hypothesis.graph_pattern
        )
