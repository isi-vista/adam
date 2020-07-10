from abc import ABC
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union
from adam.learner.language_mode import LanguageMode
from more_itertools import flatten
from networkx import all_shortest_paths, subgraph

from adam.language import LinguisticDescription
from adam.learner import LearningExample, get_largest_matching_pattern
from adam.learner.learner_utils import assert_static_situation
from adam.learner.object_recognizer import (
    ObjectRecognizer,
    PerceptionGraphFromObjectRecognizer,
)
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.pursuit import AbstractPursuitLearner
from adam.learner.subset import AbstractTemplateSubsetLearner
from adam.learner.surface_templates import SLOT1, SLOT2, SurfaceTemplate
from adam.learner.template_learner import AbstractTemplateLearner
from adam.perception import ObjectPerception, PerceptualRepresentation, MatchMode
from adam.perception.deprecated import LanguageAlignedPerception
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception.perception_graph import (
    PerceptionGraph,
    PerceptionGraphNode,
    _graph_node_order,
)
from adam.semantics import ObjectSemanticNode, SyntaxSemanticsVariable
from adam.utils.networkx_utils import digraph_with_nodes_sorted_by
from attr import attrib, attrs
from attr.validators import instance_of, optional
from immutablecollections import ImmutableDict, ImmutableSet, immutabledict, immutableset


@attrs
class AbstractPrepositionTemplateLearner(AbstractTemplateLearner, ABC):
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
        assert_static_situation(to_check)

    def _extract_perception_graph(
        self, perception: PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame]
    ) -> PerceptionGraph:
        return PerceptionGraph.from_frame(perception.frames[0])

    def _preprocess_scene_for_learning(
        self, language_concept_alignment: LanguageAlignedPerception
    ) -> LanguageAlignedPerception:
        post_recognition_object_perception_alignment = self._object_recognizer.match_objects_with_language_old(
            language_concept_alignment
        )
        num_matched_objects = len(
            post_recognition_object_perception_alignment.node_to_language_span
        )
        if num_matched_objects != 2:
            raise RuntimeError(
                f"Learning a preposition with more than two recognized objects "
                f"is not currently supported. Found {num_matched_objects} for "
                f"{language_concept_alignment.language}."
            )
        return post_recognition_object_perception_alignment

    def _preprocess_scene_for_description(
        self, perception_graph: PerceptionGraph, allow_undescribed: bool = False
    ) -> PerceptionGraphFromObjectRecognizer:
        return self._object_recognizer.match_objects_old(perception_graph)

    def _extract_surface_template(
        self,
        language_concept_alignment: LanguageAlignedPerception,
        language_mode: LanguageMode = LanguageMode.ENGLISH,
    ) -> SurfaceTemplate:
        return language_concept_alignment.to_surface_template(
            object_node_to_template_variable=immutabledict(
                [
                    (language_concept_alignment.aligned_nodes[0], SLOT1),
                    (language_concept_alignment.aligned_nodes[1], SLOT2),
                ]
            ),
            determiner_prefix_slots=[SLOT1, SLOT2],
            language_mode=language_mode,
        )


def preposition_hypothesis_from_perception(
    scene_aligned_perception: LanguageAlignedPerception,
    template_variables_to_object_match_nodes: Mapping[
        SyntaxSemanticsVariable, ObjectSemanticNode
    ],
) -> PerceptionGraphTemplate:
    """
        Create a hypothesis for the semantics of a preposition based on the observed scene.

        Our current implementation is to just include the content
        on the path between the recognized object nodes
        and one hop away from that path.
        """

    # The directions of edges in the perception graph are not necessarily meaningful
    # from the point-of-view of hypothesis generation, so we need an undirected copy
    # of the graph.
    perception_digraph = scene_aligned_perception.perception_graph.copy_as_digraph()
    perception_graph_undirected = perception_digraph.to_undirected(
        # as_view=True loses determinism
        as_view=False
    )

    if {SLOT1, SLOT2} != set(template_variables_to_object_match_nodes.keys()):
        raise RuntimeError(
            "Can only make a preposition hypothesis if the recognized "
            "objects are aligned to SurfaceTemplateVariables SLOT1 and SLOT2"
        )

    slot1_object = template_variables_to_object_match_nodes[SLOT1]
    slot2_object = template_variables_to_object_match_nodes[SLOT2]

    # The core of our hypothesis for the semantics of a preposition is all nodes
    # along the shortest path between the two objects involved in the perception graph.
    hypothesis_spine_nodes: ImmutableSet[PerceptionGraphNode] = immutableset(
        flatten(
            # if there are multiple paths between the object match nodes,
            # we aren't sure which are relevant, so we include them all in our hypothesis
            # and figure we can trim out irrelevant stuff as we make more observations.
            all_shortest_paths(perception_graph_undirected, slot2_object, slot1_object)
        )
    )

    # Along the core of our hypothesis we also want to collect the predecessors and successors
    hypothesis_nodes_mutable = []
    for node in hypothesis_spine_nodes:
        if node not in {slot1_object, slot2_object}:
            for successor in perception_digraph.successors(node):
                if not isinstance(successor, ObjectPerception):
                    hypothesis_nodes_mutable.append(successor)
            for predecessor in perception_digraph.predecessors(node):
                if not isinstance(predecessor, ObjectPerception):
                    hypothesis_nodes_mutable.append(predecessor)

    hypothesis_nodes_mutable.extend(hypothesis_spine_nodes)

    # We wrap the nodes in an immutable set to remove duplicates
    # while preserving iteration determinism.
    hypothesis_nodes = immutableset(hypothesis_nodes_mutable)

    preposition_sub_graph = PerceptionGraph(
        digraph_with_nodes_sorted_by(
            subgraph(perception_digraph, hypothesis_nodes), _graph_node_order
        )
    )

    return PerceptionGraphTemplate.from_graph(
        preposition_sub_graph, template_variables_to_object_match_nodes
    )


@attrs
class PrepositionPursuitLearner(
    AbstractPursuitLearner, AbstractPrepositionTemplateLearner
):
    """
    An implementation of pursuit learner for preposition leaning
    """

    def _candidate_hypotheses(
        self, language_aligned_perception: LanguageAlignedPerception
    ) -> Sequence[PerceptionGraphTemplate]:
        # We represent prepositions as regex-like templates over the surface strings.
        # As an English-specific hack, the leftmost recognized object
        # is always taken to be the object modified, and the right one the ground.
        template_variables_to_object_match_nodes: ImmutableDict[
            SyntaxSemanticsVariable, ObjectSemanticNode
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
            match_mode=MatchMode.OBJECT,
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
                template_variable_to_pattern_node=hypothesis.template_variable_to_pattern_node,
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

    def log_hypotheses(self, log_output_path: Path) -> None:
        for (surface_template, hypothesis) in self._lexicon.items():
            template_string = surface_template.to_short_string()
            hypothesis.render_to_file(template_string, log_output_path / template_string)

    def _update_hypothesis(
        self,
        previous_pattern_hypothesis: PerceptionGraphTemplate,
        current_pattern_hypothesis: PerceptionGraphTemplate,
    ) -> Optional[PerceptionGraphTemplate]:
        return previous_pattern_hypothesis.intersection(
            current_pattern_hypothesis,
            ontology=self._ontology,
            match_mode=MatchMode.NON_OBJECT,
        )


@attrs
class SubsetPrepositionLearner(
    AbstractTemplateSubsetLearner, AbstractPrepositionTemplateLearner
):
    def _hypothesis_from_perception(
        self, preprocessed_input: LanguageAlignedPerception
    ) -> PerceptionGraphTemplate:
        # We represent prepositions as regex-like templates over the surface strings.
        # As an English-specific hack, the leftmost recognized object
        # is always taken to be the object modified, and the right one the ground.
        template_variables_to_object_match_nodes: ImmutableDict[
            SyntaxSemanticsVariable, ObjectSemanticNode
        ] = immutabledict(
            [
                (SLOT1, preprocessed_input.aligned_nodes[0]),
                (SLOT2, preprocessed_input.aligned_nodes[1]),
            ]
        )

        return preposition_hypothesis_from_perception(
            preprocessed_input,
            template_variables_to_object_match_nodes=template_variables_to_object_match_nodes,
        )

    def _update_hypothesis(
        self,
        previous_pattern_hypothesis: PerceptionGraphTemplate,
        current_pattern_hypothesis: PerceptionGraphTemplate,
    ) -> Optional[PerceptionGraphTemplate]:
        return previous_pattern_hypothesis.intersection(
            current_pattern_hypothesis,
            ontology=self._ontology,
            match_mode=MatchMode.NON_OBJECT,
        )
