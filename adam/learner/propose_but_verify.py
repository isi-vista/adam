import logging
from pathlib import Path
from random import Random
from typing import Dict, Generic, List, Mapping, Optional, Set, Tuple

from attr.validators import instance_of, optional
from immutablecollections import immutabledict
from vistautils.parameters import Parameters

from adam.language import (
    LinguisticDescription,
    LinguisticDescriptionT,
    TokenSequenceLinguisticDescription,
)
from adam.learner import (
    LanguageLearner,
    LearningExample,
    get_largest_matching_pattern,
    graph_without_learner,
)
from adam.ontology.ontology import Ontology
from adam.ontology.phase1_ontology import GAILA_PHASE_1_ONTOLOGY
from adam.ontology.phase1_spatial_relations import Region
from adam.perception import ObjectPerception, PerceptionT, PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
    RgbColorPerception,
)
from adam.perception.perception_graph import (
    DebugCallableType,
    GraphLogger,
    PerceptionGraph,
    PerceptionGraphPattern,
)
from adam.utils import networkx_utils
from attr import Factory, attrib, attrs


@attrs()
class ProposeButVerifyLanguageLearner(
    Generic[PerceptionT, LinguisticDescriptionT],
    LanguageLearner[PerceptionT, LinguisticDescription],
):

    _ontology: Ontology = attrib(validator=instance_of(Ontology), kw_only=True)
    _observation_num = attrib(init=False, default=0)
    _rng: Random = attrib(validator=instance_of(Random))
    _words_to_hypotheses: Dict[str, PerceptionGraphPattern] = attrib(
        init=False, default=Factory(dict)
    )
    _log_word_hypotheses_to: Optional[Path] = attrib(
        validator=optional(instance_of(Path)), default=None
    )
    _graph_logger: Optional[GraphLogger] = attrib(
        validator=optional(instance_of(GraphLogger)), default=None
    )
    debug_counter = 0
    _debug_callback: Optional[DebugCallableType] = attrib(default=None)
    _graph_match_confirmation_threshold: float = attrib(default=0.9, kw_only=True)
    _word_to_logger: Dict[str, GraphLogger] = attrib(init=False, default=Factory(dict))
    _rendered_word_hypothesis_pair_ids: Set[str] = attrib(
        init=False, default=Factory(set)
    )

    @staticmethod
    def from_parameters(
        params: Parameters, *, graph_logger: Optional[GraphLogger] = None
    ) -> "ProposeButVerifyLanguageLearner":  # type: ignore
        log_word_hypotheses_dir = params.optional_creatable_directory(
            "log_word_hypotheses_dir"
        )
        if log_word_hypotheses_dir:
            logging.info("Hypotheses will be logged to %s", log_word_hypotheses_dir)

        rng = Random()
        rng.seed(params.optional_integer("random_seed", default=0))

        return ProposeButVerifyLanguageLearner(
            graph_match_confirmation_threshold=params.floating_point(
                "graph_match_confirmation_threshold"
            ),
            graph_logger=graph_logger,
            log_word_hypotheses_to=log_word_hypotheses_dir,
            rng=rng,
            ontology=GAILA_PHASE_1_ONTOLOGY,
        )

    # observe method
    def observe(
        self, learning_example: LearningExample[PerceptionT, LinguisticDescription]
    ) -> None:

        logging.info("Observation %s", self._observation_num)
        self._observation_num += 1

        perception = learning_example.perception
        if len(perception.frames) != 1:
            raise RuntimeError("Pursuit learner can only handle single frames for now")
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

        self.learn_propose_but_verify(
            observed_perception_graph, observed_linguistic_description
        )

    # learn method
    def learn_propose_but_verify(
        self,
        observed_perception_graph: PerceptionGraph,
        observed_linguistic_description: Tuple[str, ...],
    ) -> None:
        logging.info(f"Pursuit learner observing {observed_linguistic_description}")
        for word in observed_linguistic_description:
            if word in ("a", "the"):
                continue
            logging.info(f"Considering '{word}'")
            if word not in self._words_to_hypotheses:
                # a) If word has never been seen before, go through initialization step
                self.initialization_step(word, observed_perception_graph)
            else:
                # b) If word has been seen before, test the recorded hypothesis
                if self.test_hypothesis_step(word, observed_perception_graph):
                    # If hypothesis is confirmed, do nothing
                    continue
                else:
                    # If hypothesis is not confirmed, treat word as an unseen word, and go
                    # through the initialization step again (to randomly select meaning)
                    self.initialization_step(word, observed_perception_graph)
            if self._log_word_hypotheses_to:
                self._log_hypotheses(word)

    def initialization_step(self, word: str, observed_perception_graph: PerceptionGraph):
        # If the word has never been seen before OR word does not match any
        # 'object' in the current frame, learn a new hypothesis/pattern
        # generated as a pattern graph from the perception.
        meanings = [
            PerceptionGraphPattern.from_graph(object_).perception_graph_pattern
            for object_ in self.get_objects_from_perception(observed_perception_graph)
        ]

        #  Pick a random meaning for our hypothesis (Independent of any prior data)
        pattern_hypothesis = self._rng.choice(meanings)

        if self._graph_logger:
            self._graph_logger.log_graph(
                pattern_hypothesis,
                logging.INFO,
                "Initializing (randomly chosen) meaning for %s ",
                word,
            )

        self._words_to_hypotheses[word] = pattern_hypothesis

    # Test Hypothesis
    def test_hypothesis_step(
        self, word: str, observed_perception_graph: PerceptionGraph
    ) -> bool:
        # Compare existing hypothesis with observation
        # Return whether
        hypotheses_for_word = self._words_to_hypotheses[word]
        logging.info("Current hypothesis is %s", abs(hash((word, hypotheses_for_word))))
        self.debug_counter += 1
        # If the leading hypothesis sufficiently matches the observation, reinforce it
        # To do, we check how much of the leading pattern hypothesis matches the perception
        partial_match = self._compute_match_ratio(
            hypotheses_for_word, observed_perception_graph
        )

        hypothesis_is_confirmed = partial_match.matched_exactly()
        # If the hypothesis is confirmed, we leave it as it is
        if hypothesis_is_confirmed and partial_match.matching_subgraph:
            logging.info("Hypothesis is confirmed")
        # If the hypothesis is disconfirmed
        else:
            logging.info(
                f"Hypothesis is disconfirmed (Match ratio: {partial_match.match_ratio()}"
            )
        return hypothesis_is_confirmed

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

        descriptions = []
        for word, meaning_pattern in self._words_to_hypotheses.items():
            # Use PerceptionGraphPattern.matcher and matcher.matches() for a complete match
            matcher = meaning_pattern.matcher(
                observed_perception_graph, matching_objects=True
            )
            if any(
                matcher.matches(
                    use_lookahead_pruning=True, graph_logger=self._graph_logger
                )
            ):
                learned_description = TokenSequenceLinguisticDescription(("a", word))
                descriptions.append((learned_description, 1.0))
        return immutabledict(descriptions)

    def _log_hypotheses(self, word: str) -> None:
        assert self._log_word_hypotheses_to
        # if the user has asked us
        # to log the progress of the learner's hypotheses about word meanings,
        # then we use a GraphLogger per-word to write diagram's
        # of each word's hypotheses into their own sub-directory
        if word in self._word_to_logger:
            graph_logger = self._word_to_logger[word]
        else:
            log_directory_for_word = self._log_word_hypotheses_to / word

            graph_logger = GraphLogger(
                log_directory=log_directory_for_word, enable_graph_rendering=True
            )
            self._word_to_logger[word] = graph_logger

        def compute_hypothesis_id(h: PerceptionGraphPattern) -> str:
            # negative hashes cause the dot rendered to crash
            return str(abs(hash((word, h))))

        if word in self._words_to_hypotheses:
            logging.info("The word %s has been seen before", word)
            hypothesised_meaning = self._words_to_hypotheses[word]
            hypothesis_id = compute_hypothesis_id(hypothesised_meaning)
            if hypothesis_id not in self._rendered_word_hypothesis_pair_ids:
                graph_logger.log_graph(
                    hypothesised_meaning,
                    logging.INFO,
                    "Rendering hypothesised " "meaning %s " "for %s",
                    hypothesis_id,
                    word,
                    graph_name=str(hypothesis_id),
                )
                self._rendered_word_hypothesis_pair_ids.add(hypothesis_id)
        else:
            logging.info(
                "The word %s has never been seen before",
                word,
                "\nTHere are no current hypotheses for it",
            )

    # Helper method, from pursuit.py
    @staticmethod
    def get_objects_from_perception(
        observed_perception_graph: PerceptionGraph
    ) -> List[PerceptionGraph]:
        perception_as_digraph = observed_perception_graph.copy_as_digraph()
        perception_as_graph = perception_as_digraph.to_undirected()

        meanings = []

        # 1) Take all of the obj perc that dont have part of relationships with anything else
        root_object_perception_nodes = []
        for node in perception_as_graph.nodes:
            if isinstance(node, ObjectPerception) and node.debug_handle != "the ground":
                if not any(
                    [
                        u == node and str(data["label"]) == "partOf"
                        for u, v, data in perception_as_digraph.edges.data()
                    ]
                ):
                    root_object_perception_nodes.append(node)

        # 2) for each of these, walk along the part of relationships backwards,
        # i.e find all of the subparts of the root object
        for root_object_perception_node in root_object_perception_nodes:
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

    def _compute_match_ratio(
        self, pattern: PerceptionGraphPattern, graph: PerceptionGraph
    ) -> "ProposeButVerifyLanguageLearner.PartialMatch":
        """
        Computes the fraction of pattern graph nodes of *pattern* which match *graph*.
        """
        hypothesis_pattern_common_subgraph = get_largest_matching_pattern(
            pattern,
            graph,
            debug_callback=self._debug_callback,
            graph_logger=self._graph_logger,
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
        return ProposeButVerifyLanguageLearner.PartialMatch(
            hypothesis_pattern_common_subgraph,
            num_nodes_matched=num_nodes_matched,
            num_nodes_in_pattern=leading_hypothesis_num_nodes,
        )

    @attrs(frozen=True)
    class PartialMatch:
        """
        *match_ratio* should be 1.0 exactly for a perfect match.
        """

        matching_subgraph: Optional[PerceptionGraphPattern] = attrib(
            validator=optional(instance_of(PerceptionGraphPattern))
        )
        num_nodes_matched: int = attrib(validator=instance_of(int), kw_only=True)
        num_nodes_in_pattern: int = attrib(validator=instance_of(int), kw_only=True)

        def matched_exactly(self) -> bool:
            return self.num_nodes_matched == self.num_nodes_in_pattern

        def match_ratio(self) -> float:
            return self.num_nodes_matched / self.num_nodes_in_pattern
