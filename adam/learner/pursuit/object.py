import logging
from random import Random
from typing import Iterable, List, Optional, Sequence, Union

from attr.validators import instance_of, optional

from adam.language import LinguisticDescription, TokenSequenceLinguisticDescription
from adam.learner import (
    LearningExample,
    get_largest_matching_pattern,
    graph_without_learner,
)
from adam.learner.object_recognizer import PerceptionGraphFromObjectRecognizer
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.pursuit import AbstractPursuitLearner, HypothesisLogger
from adam.learner.surface_templates import SurfaceTemplate
from adam.ontology.phase1_ontology import GAILA_PHASE_1_ONTOLOGY
from adam.ontology.phase1_spatial_relations import Region
from adam.perception import ObjectPerception, PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
    RgbColorPerception,
)
from adam.perception.perception_graph import (
    LanguageAlignedPerception,
    PerceptionGraph,
    PerceptionGraphPattern,
)
from adam.utils import networkx_utils
from attr import attrib, attrs, evolve
from immutablecollections import immutabledict
from vistautils.parameters import Parameters


class ObjectPursuitLearner(
    AbstractPursuitLearner[
        DevelopmentalPrimitivePerceptionFrame, TokenSequenceLinguisticDescription
    ]
):
    """
    An implementation of pursuit learner for object recognition
    """

    def _assert_valid_input(
        self,
        to_check: Union[
            LearningExample[DevelopmentalPrimitivePerceptionFrame, LinguisticDescription],
            PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame],
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
        return evolve(
            language_aligned_perception,
            perception_graph=self._common_preprocessing(
                language_aligned_perception.perception_graph
            ),
        )

    def _preprocess_scene_for_description(
        self, perception_graph: PerceptionGraph
    ) -> PerceptionGraphFromObjectRecognizer:
        return PerceptionGraphFromObjectRecognizer(
            self._common_preprocessing(perception_graph),
            description_to_matched_object_node=immutabledict(),
        )

    def _common_preprocessing(self, perception_graph: PerceptionGraph) -> PerceptionGraph:
        return graph_without_learner(perception_graph)

    def _extract_surface_template(
        self, preprocessed_input: LanguageAlignedPerception
    ) -> SurfaceTemplate:
        return SurfaceTemplate(preprocessed_input.language.as_token_sequence())

    def _candidate_hypotheses(
        self, language_aligned_perception: LanguageAlignedPerception
    ) -> Sequence[PerceptionGraphTemplate]:
        """
        Given a learning input, returns all possible meaning hypotheses.
        """
        return [
            self._hypothesis_from_perception(object_)
            for object_ in self._candidate_perceptions(
                language_aligned_perception.perception_graph
            )
        ]

    def get_objects_from_perception(
        self, observed_perception_graph: PerceptionGraph
    ) -> List[PerceptionGraph]:
        perception_as_digraph = observed_perception_graph.copy_as_digraph()
        perception_as_graph = perception_as_digraph.to_undirected()

        meanings = []

        # 1) Take all of the obj perc that dont have part of relationships with anything else
        root_object_percetion_nodes = []
        for node in perception_as_graph.nodes:
            if isinstance(node, ObjectPerception) and node.debug_handle != "the ground":
                if not any(
                    [
                        u == node and str(data["label"]) == "partOf"
                        for u, v, data in perception_as_digraph.edges.data()
                    ]
                ):
                    root_object_percetion_nodes.append(node)

        # 2) for each of these, walk along the part of relationships backwards,
        # i.e find all of the subparts of the root object
        for root_object_perception_node in root_object_percetion_nodes:
            # Iteratively get all other object perceptions that connect to a root with a part of
            # relation
            all_object_perception_nodes = [root_object_perception_node]
            frontier = [root_object_perception_node]
            updated = True
            while updated:
                updated = False
                new_frontier = []
                for frontier_node in frontier:
                    for node in perception_as_graph.neighbors(frontier_node):
                        edge_data = perception_as_digraph.get_edge_data(
                            node, frontier_node, default=-1
                        )
                        if edge_data != -1 and str(edge_data["label"]) == "partOf":
                            new_frontier.append(node)

                if new_frontier:
                    all_object_perception_nodes.extend(new_frontier)
                    updated = True
                    frontier = new_frontier

            # Now we have a list of all perceptions that are connected
            # 3) For each of these objects including root object, get axes, properties,
            # and relations and regions which are between these internal object perceptions
            other_nodes = []
            for node in all_object_perception_nodes:
                for neighbor in perception_as_graph.neighbors(node):
                    # Filter out regions that don't have a reference in all object perception nodes
                    # TODO: We currently remove colors to achieve a match - otherwise finding
                    #  patterns fails.
                    if (
                        isinstance(neighbor, Region)
                        and neighbor.reference_object not in all_object_perception_nodes
                        or isinstance(neighbor, RgbColorPerception)
                    ):
                        continue
                    # Append all other none-object nodes to be kept in the subgraph
                    if not isinstance(neighbor, ObjectPerception):
                        other_nodes.append(neighbor)

            subgraph = networkx_utils.subgraph(
                perception_as_digraph, all_object_perception_nodes + other_nodes
            )
            meanings.append(PerceptionGraph(subgraph))
        logging.info(f"Got {len(meanings)} candidate meanings")
        return meanings

    def _hypothesis_from_perception(
        self, perception: PerceptionGraph
    ) -> PerceptionGraphTemplate:
        return PerceptionGraphTemplate(
            PerceptionGraphPattern.from_graph(perception).perception_graph_pattern
        )

    def _candidate_perceptions(self, observed_perception_graph) -> List[PerceptionGraph]:
        return self.get_objects_from_perception(observed_perception_graph)

    def _matches(
        self,
        *,
        hypothesis: PerceptionGraphTemplate,
        observed_perception_graph: PerceptionGraph,
    ) -> bool:
        matcher = hypothesis.graph_pattern.matcher(
            observed_perception_graph, matching_objects=True
        )
        return any(
            matcher.matches(
                use_lookahead_pruning=True, graph_logger=self._hypothesis_logger
            )
        )

    @attrs(frozen=True)
    class ObjectHypothesisPartialMatch(AbstractPursuitLearner.PartialMatch):
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
    ) -> "ObjectPursuitLearner.ObjectHypothesisPartialMatch":
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

        return ObjectPursuitLearner.ObjectHypothesisPartialMatch(
            PerceptionGraphTemplate(hypothesis_pattern_common_subgraph)
            if hypothesis_pattern_common_subgraph
            else None,
            num_nodes_matched=num_nodes_matched,
            num_nodes_in_pattern=leading_hypothesis_num_nodes,
        )

    def _find_identical_hypothesis(
        self,
        new_hypothesis: PerceptionGraphTemplate,
        candidates: Iterable[PerceptionGraphTemplate],
    ) -> Optional[PerceptionGraphTemplate]:
        for candidate in candidates:
            if new_hypothesis.graph_pattern.check_isomorphism(candidate.graph_pattern):
                return candidate
        return None

    def _are_isomorphic(
        self, h: PerceptionGraphTemplate, hypothesis: PerceptionGraphTemplate
    ) -> bool:
        return h.graph_pattern.check_isomorphism(hypothesis.graph_pattern)

    @staticmethod
    def from_parameters(
        params: Parameters, *, graph_logger: Optional[HypothesisLogger] = None
    ) -> "ObjectPursuitLearner":  # type: ignore
        log_word_hypotheses_dir = params.optional_creatable_directory(
            "log_word_hypotheses_dir"
        )
        if log_word_hypotheses_dir:
            logging.info("Hypotheses will be logged to %s", log_word_hypotheses_dir)

        rng = Random()
        rng.seed(params.optional_integer("random_seed", default=0))

        return ObjectPursuitLearner(
            learning_factor=params.floating_point("learning_factor"),
            graph_match_confirmation_threshold=params.floating_point(
                "graph_match_confirmation_threshold"
            ),
            lexicon_entry_threshold=params.floating_point("lexicon_entry_threshold"),
            smoothing_parameter=params.floating_point("smoothing_parameter"),
            graph_logger=graph_logger,
            log_word_hypotheses_to=log_word_hypotheses_dir,
            rng=rng,
            ontology=GAILA_PHASE_1_ONTOLOGY,
        )
