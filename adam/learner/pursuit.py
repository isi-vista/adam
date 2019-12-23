import logging
import random as r
from pathlib import Path
from typing import Dict, Generic, List, Mapping, Optional, Set, Tuple

from attr.validators import instance_of, optional
from immutablecollections import immutabledict
from more_itertools import first
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

r.seed(0)


@attrs
class PursuitLanguageLearner(
    Generic[PerceptionT, LinguisticDescriptionT],
    LanguageLearner[PerceptionT, LinguisticDescription],
):
    """
    An implementation of `LanguageLearner` for pursuit learning based approach for single object detection.
    """

    _words_to_hypotheses_and_scores: Dict[
        str, Dict[PerceptionGraphPattern, float]
    ] = attrib(init=False, default=Factory(dict))
    _lexicon: Dict[str, PerceptionGraphPattern] = attrib(
        init=False, default=Factory(dict)
    )
    _debug_callback: Optional[DebugCallableType] = attrib(default=None)

    # Learning factor (gamma) is the factor with which we update the hypotheses scores during reinforcement.
    _learning_factor: float = attrib(default=0.1, kw_only=True)
    # We use this threshold to measure whether a new perception sufficiently matches a previous hypothesis.
    _graph_match_confirmation_threshold: float = attrib(default=0.9, kw_only=True)
    # Threshold value for adding word to lexicon
    _lexicon_entry_threshold: float = attrib(default=0.8, kw_only=True)
    # Counter to be used to prevent prematurely lexicalizing novel words
    _words_to_number_of_observations: Dict[str, int] = attrib(
        init=False, default=Factory(dict)
    )
    _graph_logger: Optional[GraphLogger] = attrib(
        validator=optional(instance_of(GraphLogger)), default=None
    )
    debug_counter = 0

    # the following two fields are used if the user wishes the hypotheses for word meanings
    # to be logged at each step of learning for detailed debugging.
    _log_word_hypotheses_to: Optional[Path] = attrib(
        validator=optional(instance_of(Path)), default=None
    )
    _word_to_logger: Dict[str, GraphLogger] = attrib(init=False, default=Factory(dict))
    _rendered_word_hypothesis_pair_ids: Set[str] = attrib(
        init=False, default=Factory(set)
    )

    @staticmethod
    def from_parameters(
        params: Parameters, *, graph_logger: Optional[GraphLogger] = None
    ) -> "PursuitLanguageLearner":  # type: ignore
        log_word_hypotheses_dir = params.optional_creatable_directory(
            "log_word_hypotheses_dir"
        )
        if log_word_hypotheses_dir:
            logging.info("Hypotheses will be logged to %s", log_word_hypotheses_dir)
        return PursuitLanguageLearner(
            learning_factor=params.floating_point("learning_factor"),
            graph_match_confirmation_threshold=params.floating_point(
                "graph_match_confirmation_threshold"
            ),
            lexicon_entry_threshold=params.floating_point("lexicon_entry_threshold"),
            graph_logger=graph_logger,
            log_word_hypotheses_to=log_word_hypotheses_dir,
        )

    def observe(
        self, learning_example: LearningExample[PerceptionT, LinguisticDescription]
    ) -> None:
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

        self.learn_with_pursuit(
            observed_perception_graph, observed_linguistic_description
        )

    def learn_with_pursuit(
        self,
        observed_perception_graph: PerceptionGraph,
        observed_linguistic_description: Tuple[str, ...],
    ) -> None:
        logging.info(f"Pursuit learner observing {observed_linguistic_description}")
        # The learner’s words are W, meanings are M, their associations are A, and the new utterance is U = (W_U, M_U).
        # For every w in W_U
        for word in observed_linguistic_description:
            # TODO: pursuit learner hard-coded to ignore determiners
            # https://github.com/isi-vista/adam/issues/498
            if word in ("a", "the"):
                continue
            if word in self._words_to_number_of_observations:
                self._words_to_number_of_observations[word] += 1
            else:
                self._words_to_number_of_observations[word] = 1

            # If don't already know the meaning of the word, go through learning steps:
            if word not in self._lexicon:
                logging.info(f"Considering '{word}'")
                if word not in self._words_to_hypotheses_and_scores:
                    # a) Initialization step, if the word is a novel word
                    self.initialization_step(word, observed_perception_graph)
                else:
                    # b) If we already have a hypothesis, run the learning reinforcement step
                    is_hypothesis_confirmed = self.learning_step(
                        word, observed_perception_graph
                    )
                    # Try lexicon step if we confirmed a meaning
                    if is_hypothesis_confirmed:
                        self.lexicon_step(word)

                if self._log_word_hypotheses_to:
                    self._log_hypotheses(word)

    def initialization_step(self, word: str, observed_perception_graph: PerceptionGraph):
        # If it's a novel word, learn a new hypothesis/pattern,
        # generated as a pattern graph from the perception.
        # We want h_0 = arg_min_(m in M_U) max(A_m); i.e. h_0 is pattern_hypothesis
        meanings = self.get_meanings_from_perception(observed_perception_graph)
        pattern_hypothesis = first(meanings)
        min_score = float("inf")
        # Of the possible meanings for the word in this scene,
        # make our initial hypothesis the one with the least association
        # with any other word.
        for meaning in meanings:
            # TODO Try to make this more efficient?
            max_association_score = max(
                [
                    s
                    for w, h_to_s in self._words_to_hypotheses_and_scores.items()
                    for h, s in h_to_s.items()
                    if h.check_isomorphism(meaning)
                ]
                + [0]
            )
            if max_association_score < min_score:
                pattern_hypothesis = meaning
                min_score = max_association_score

        if self._graph_logger:
            self._graph_logger.log_graph(
                pattern_hypothesis,
                logging.INFO,
                "Initializing meaning for %s " "with score %s",
                word,
                self._learning_factor,
            )

        self._words_to_hypotheses_and_scores[word] = {
            pattern_hypothesis: self._learning_factor
        }

    def learning_step(
        self, word: str, observed_perception_graph: PerceptionGraph
    ) -> bool:
        # Select the most probable meaning h for w
        # I.e., if we already have hypotheses, get the leading hypothesis and compare it with the observed perception

        hypothesis_for_word = self._words_to_hypotheses_and_scores[word]
        previous_hypotheses_and_scores = hypothesis_for_word
        leading_hypothesis_pattern = max(
            previous_hypotheses_and_scores,
            key=lambda key: previous_hypotheses_and_scores[key],
        )
        # If the leading hypothesis sufficiently matches the observation, reinforce it
        # To do, we check how much of the leading pattern hypothesis matches the perception
        # TODO: Fix get_largest_matching_pattern - it is inconsistent for partial matches, depending on the start node
        hypothesis_pattern_common_subgraph = get_largest_matching_pattern(
            leading_hypothesis_pattern,
            observed_perception_graph,
            debug_callback=self._debug_callback,
            graph_logger=self._graph_logger,
        )
        current_hypothesis_score = hypothesis_for_word[leading_hypothesis_pattern]
        self.debug_counter += 1

        leading_hypothesis_num_nodes = len(
            leading_hypothesis_pattern.copy_as_digraph().nodes
        )
        num_nodes_matched = (
            len(hypothesis_pattern_common_subgraph.copy_as_digraph().nodes)
            if hypothesis_pattern_common_subgraph
            else 0
        )
        match_ratio = num_nodes_matched / leading_hypothesis_num_nodes

        # b.i) If the hypothesis is confirmed, we reinforce it.
        hypothesis_is_confirmed = num_nodes_matched == leading_hypothesis_num_nodes
        if hypothesis_is_confirmed and hypothesis_pattern_common_subgraph:
            logging.info("Current hypothesis is confirmed.")
            # Reinforce A(w,h)
            new_hypothesis_score = current_hypothesis_score + self._learning_factor * (
                1 - current_hypothesis_score
            )

            # Register the updated hypothesis score of A(w,h)
            hypothesis_for_word[leading_hypothesis_pattern] = new_hypothesis_score
            logging.info("Updating hypothesis score to %s", new_hypothesis_score)
        # b.ii) If the hypothesis is disconfirmed, so we weaken the previous score
        else:
            # Penalize A(w,h)
            new_hypothesis_score = current_hypothesis_score * (1 - self._learning_factor)
            # Register the updated hypothesis score of A(w,h)
            hypothesis_for_word[leading_hypothesis_pattern] = new_hypothesis_score
            logging.info(
                "Working hypothesis disconfirmed. Reducing score from %s -> %s",
                current_hypothesis_score,
                new_hypothesis_score,
            )

            # This is where we differ from the pursuit paper.
            # If a sufficiently close relaxed version of our pattern matches,
            # we used that relaxed version as the new hypothesis to introduce
            if (
                match_ratio >= self._graph_match_confirmation_threshold
                and hypothesis_pattern_common_subgraph
            ):
                logging.info(
                    "Introducing partial match as a new hypothesis; %s of %s nodes "
                    "matched.",
                    num_nodes_matched,
                    leading_hypothesis_num_nodes,
                )
                new_hypothesis = hypothesis_pattern_common_subgraph
            else:
                # otherwise, we follow the paper and hypothesize a randomly selected meaning
                # from the scene.
                logging.info("Choosing random meaning from the scene")
                meanings = self.get_meanings_from_perception(observed_perception_graph)
                new_hypothesis = r.choice(meanings)

            # TODO: there's a problem here - since all hypotheses are distinct,
            # we could accidentally introduce the same hypothesis multiple times here.
            hypothesis_for_word[new_hypothesis] = max(
                self._learning_factor, hypothesis_for_word.get(new_hypothesis, 0.0)
            )
            if self._graph_logger:
                self._graph_logger.log_graph(
                    new_hypothesis,
                    logging.INFO,
                    "Registered a new hypothesis with score %s",
                    self._learning_factor,
                )

        return hypothesis_is_confirmed

    def lexicon_step(self, word: str) -> None:
        # If any conditional probability P(h^|w) exceeds a certain threshold value (h), then file (w, h^) into the
        # lexicon
        # From Pursuit paper: P(h|w) = (A(w,h) + Gamma) / (Sum(A_w) + N x Gamma)
        leading_hypothesis_entry = self._leading_hypothesis_for(word)
        assert leading_hypothesis_entry
        (leading_hypothesis_pattern, leading_hypothesis_score) = leading_hypothesis_entry

        all_hypotheses_for_word = self._words_to_hypotheses_and_scores[word]
        sum_of_all_scores = sum(all_hypotheses_for_word.values())
        number_of_meanings = len(all_hypotheses_for_word)

        probability_of_meaning_given_word = (
            leading_hypothesis_score + self._learning_factor
        ) / (sum_of_all_scores + number_of_meanings * self._learning_factor)
        times_word_has_been_seen = self._words_to_number_of_observations[word]
        logging.info(
            "Prob of meaning given word: %s, Times seen: %s",
            probability_of_meaning_given_word,
            times_word_has_been_seen,
        )
        # file (w, h^) into the lexicon

        # TODO: We sometimes prematurely lexicalize words, so we use this arbitrary counter threshold
        if probability_of_meaning_given_word > self._lexicon_entry_threshold:
            if times_word_has_been_seen > 5:
                self._lexicon[word] = leading_hypothesis_pattern
                # Remove the word from hypotheses
                self._words_to_hypotheses_and_scores.pop(word)
                if self._graph_logger:
                    self._graph_logger.log_graph(
                        leading_hypothesis_pattern, logging.INFO, "Lexicalized %s", word
                    )
            else:
                logging.info("Would lexicalize, but haven't see the word often enough")

    @staticmethod
    def get_meanings_from_perception(
        observed_perception_graph: PerceptionGraph
    ) -> List[PerceptionGraphPattern]:
        # TODO: Return all possible meanings. We currently remove ground, and return complete objects
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
            # Iteratively get all other object perceptions that connect to a root with a part of relation
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
                    # TODO: We currently remove colors to achieve a match - otherwise finding patterns fails.
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
            meaning = PerceptionGraphPattern.from_graph(subgraph).perception_graph_pattern
            meanings.append(meaning)
        logging.info(f"Got {len(meanings)} candidate meanings")
        return meanings

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

        for word, meaning_pattern in self._lexicon.items():
            # Use PerceptionGraphPattern.matcher and matcher.matches() for a complete match
            matcher = meaning_pattern.matcher(observed_perception_graph)
            if any(
                matcher.matches(
                    use_lookahead_pruning=True, graph_logger=self._graph_logger
                )
            ):
                learned_description = TokenSequenceLinguisticDescription(("a", word))
                descriptions.append((learned_description, 1.0))

        if not descriptions:
            # no lexicalized word matched the perception,
            # but we can still try to match our leading hypotheses
            for word in self._words_to_hypotheses_and_scores.keys():
                # mypy doesn't know the leading hypothesis will always exist here,
                # but we do.
                leading_hypothesis_pair = self._leading_hypothesis_for(  # type: ignore
                    word
                )
                if leading_hypothesis_pair:
                    (leading_hypothesis, score) = leading_hypothesis_pair
                    matcher = leading_hypothesis.matcher(observed_perception_graph)
                    match = first(
                        matcher.matches(
                            use_lookahead_pruning=True, graph_logger=self._graph_logger
                        ),
                        default=None,
                    )
                    if match:
                        learned_description = TokenSequenceLinguisticDescription(
                            ("a", word)
                        )
                        descriptions.append((learned_description, score))

        return immutabledict(descriptions)

    def _leading_hypothesis_for(
        self, word: str
    ) -> Optional[Tuple[PerceptionGraphPattern, float]]:
        hypotheses_and_scores_for_word = self._words_to_hypotheses_and_scores.get(
            word, None
        )
        if hypotheses_and_scores_for_word:
            return max(hypotheses_and_scores_for_word.items(), key=lambda entry: entry[1])
        else:
            return None

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

        if word in self._lexicon:
            logging.info("The word %s has been lexicalized", word)
            lexicalized_meaning = self._lexicon[word]
            hypothesis_id = compute_hypothesis_id(lexicalized_meaning)
            if hypothesis_id not in self._rendered_word_hypothesis_pair_ids:
                graph_logger.log_graph(
                    lexicalized_meaning,
                    logging.INFO,
                    "Rendering lexicalized " "meaning %s " "for %s",
                    hypothesis_id,
                    word,
                    graph_name=str(hypothesis_id),
                )
                self._rendered_word_hypothesis_pair_ids.add(hypothesis_id)
        else:
            scored_hypotheses_for_word = self._words_to_hypotheses_and_scores[
                word
            ].items()
            # First, make sure all the hypotheses have been rendered.
            # We use the hash of this pair to generate a unique ID to match up logging messages
            # to the PDFs of hypothesized meaning graphs.
            for (hypothesis, _) in scored_hypotheses_for_word:
                hypothesis_id = compute_hypothesis_id(hypothesis)
                if hypothesis_id not in self._rendered_word_hypothesis_pair_ids:
                    graph_logger.log_graph(
                        hypothesis,
                        logging.INFO,
                        "Rendering  " "hypothesized " "meaning %s for %s",
                        hypothesis_id,
                        word,
                        graph_name=str(hypothesis_id),
                    )
                    self._rendered_word_hypothesis_pair_ids.add(hypothesis_id)

            logging.info(
                "After update, hypotheses for %s are %s",
                word,
                ", ".join(
                    f"{compute_hypothesis_id(hypothesis)}={score}"
                    for (hypothesis, score) in reversed(
                        sorted(scored_hypotheses_for_word, key=lambda x: x[1])
                    )
                ),
            )
