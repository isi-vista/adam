import logging
from pathlib import Path
from random import Random
from typing import Dict, Generic, Iterable, List, Mapping, Optional, Set, Tuple
from collections import defaultdict

from attr.validators import in_, instance_of, optional
from immutablecollections import immutabledict
from more_itertools import first
from vistautils.parameters import Parameters
from vistautils.range import Range

from adam.language import (
    LinguisticDescription,
    LinguisticDescriptionT,
    TokenSequenceLinguisticDescription,
)
from adam.learner import (
    LanguageLearner,
    LearningExample,
    graph_without_learner,
    get_largest_matching_pattern,
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


@attrs
class CrossSituationalLanguageLearner(
    Generic[PerceptionT, LinguisticDescriptionT],
    LanguageLearner[PerceptionT, LinguisticDescription],
):
    """
    An implementation of `LanguageLearner` for cross-situational learning based approach for single
    object detection.
    This learner aims to learn via storing all possible meanings and narrowing down to one meaning
    by caclulating association scores and probability based off those association scores for each
    utterance situation pair. It does so be associating all words to certain meanings. For new words
    meanings that are not associated strongly to another word already are associated evenly. For
    words encountered before, words are associated more strongly to meanings encountered with that
    word before and less strongly to newer meanings. Lastly, very familar word meaning pairs are
    associated together only, these would be words generally considered lexicalized. Once
    associations are made a probability for each word meaning pair being correct is caclulated.
    Finally if the proibability is high enough the word is lexicalized. More information can be
    found here: https://onlinelibrary.wiley.com/doi/full/10.1111/j.1551-6709.2010.01104.x
    """

    # Represents the association scores for each word meaning pair
    _words_to_hypotheses_and_scores: Dict[
        str, Dict[PerceptionGraphPattern, float]
    ] = attrib(init=False, default=defaultdict(dict))
    # Represents the probability for each word meaning pair being correct
    _words_to_hypotheses_and_probability: Dict[
        str, Dict[PerceptionGraphPattern, float]
    ] = attrib(init=False, default=defaultdict(dict))
    _lexicon: Dict[str, PerceptionGraphPattern] = attrib(
        init=False, default=Factory(dict)
    )
    _smoothing_parameter: float = attrib(
        validator=in_(Range.greater_than(0.0)), kw_only=True
    )
    """
    This smoothing factor is added to the scores of all hypotheses
    when forming a probability distribution over hypotheses.
    This should be a small value, at most 0.1 and possibly much less.
    See section 3.3 of the Cross-Situational paper.
    """
    _expected_number_of_meanings: float = attrib(
        validator=in_(Range.greater_than(0.0)), kw_only=True
    )

    _ontology: Ontology = attrib(validator=instance_of(Ontology), kw_only=True)

    _rng: Random = attrib(validator=instance_of(Random))
    _debug_callback: Optional[DebugCallableType] = attrib(default=None)

    # We use this threshold to measure whether a new perception sufficiently matches a previous
    # hypothesis.
    _graph_match_confirmation_threshold: float = attrib(default=0.9, kw_only=True)
    # Threshold value for adding word to lexicon
    _lexicon_entry_threshold: float = attrib(default=0.8, kw_only=True)
    # Counter to be used to prevent prematurely lexicalizing novel words
    _word_meaning_pairs_to_number_of_observations: Dict[
        Tuple[str, PerceptionGraphPattern], int
    ] = attrib(init=False, default=defaultdict(int))
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

    _observation_num = attrib(init=False, default=0)

    @staticmethod
    def from_parameters(
        params: Parameters, *, graph_logger: Optional[GraphLogger] = None
    ) -> "CrossSituationalLanguageLearner":  # type: ignore
        log_word_hypotheses_dir = params.optional_creatable_directory(
            "log_word_hypotheses_dir"
        )
        if log_word_hypotheses_dir:
            logging.info("Hypotheses will be logged to %s", log_word_hypotheses_dir)

        rng = Random()
        rng.seed(params.optional_integer("random_seed", default=0))

        return CrossSituationalLanguageLearner(
            learning_factor=params.floating_point("learning_factor"),
            graph_match_confirmation_threshold=params.floating_point(
                "graph_match_confirmation_threshold"
            ),
            lexicon_entry_threshold=params.floating_point("lexicon_entry_threshold"),
            smoothing_parameter=params.floating_point("smoothing_parameter"),
            expected_number_of_meanings=params.floating_point(
                "expected_number_of_meanings"
            ),
            graph_logger=graph_logger,
            log_word_hypotheses_to=log_word_hypotheses_dir,
            rng=rng,
            ontology=GAILA_PHASE_1_ONTOLOGY,
        )

    def observe(
        self, learning_example: LearningExample[PerceptionT, LinguisticDescription]
    ) -> None:
        logging.info("Observation %s", self._observation_num)
        self._observation_num += 1

        perception = learning_example.perception
        if len(perception.frames) != 1:
            raise RuntimeError(
                "Cross-situational learner can only handle single frames for now"
            )
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

        self.learn_with_cross_situational(
            observed_perception_graph, observed_linguistic_description
        )

    def learn_with_cross_situational(
        self,
        observed_perception_graph: PerceptionGraph,
        observed_linguistic_description: Tuple[str, ...],
    ) -> None:
        logging.info(
            f"Cross-situational learner observing {observed_linguistic_description}"
        )
        # The learner’s words are W, meanings are M, their associations are A, and the new
        # utterance is U = (W_U, M_U).
        # For every w in W_U
        # Adding "dummy word" that smoothes out learning for situations where a word shows up
        # without it's meaning or vice versa. Also smoothes out meanings that show up without a word
        # in general.
        words = observed_linguistic_description + tuple("d")
        meanings = [
            object_
            for object_ in self.get_objects_from_perception(observed_perception_graph)
        ]

        probabilities_and_hypotheses_for_words: Dict[
            str, Dict[PerceptionGraphPattern, float]
        ] = dict()
        if self._lexicon:
            for word in words:
                # Remove meanings strongly associated with words in the lexicon
                if word in self._lexicon:
                    match: Optional[
                        "CrossSituationalLanguageLearner.PartialMatch"
                    ] = self._compute_match_ratio(
                        self._lexicon[word], observed_perception_graph
                    )
                    if match and match.matching_subgraph:
                        for meaning in meanings:
                            hypothesis = PerceptionGraphPattern.from_graph(
                                meaning
                            ).perception_graph_pattern
                            if match.matching_subgraph.check_isomorphism(hypothesis):
                                meanings.remove(meaning)

        for word in words:
            probabilities_and_hypotheses_for_words[word] = self.get_probabilities(
                word, meanings
            )

        for word in words:
            if word not in self._lexicon:
                logging.info(f"Considering '{word}'")
                # For each meaning, update the association score for the word meaning pair
                for meaning in meanings:
                    match, matching_hypothesis = self._find_similar_hypothesis(
                        meaning, self._words_to_hypotheses_and_scores[word].keys()
                    )
                    # Updating association score as described by the paper below
                    # assoc(w, m) = pass_assoc(w, m) + a(w|m, U)
                    # https://onlinelibrary.wiley.com/doi/full/10.1111/j.1551-6709.2010.01104.x (2)
                    if match and matching_hypothesis:
                        # We check that match is not None before using it
                        self._words_to_hypotheses_and_scores[word][  # type: ignore
                            match.matching_subgraph
                        ] = (
                            self.alignment_probability(
                                word,
                                words,
                                meaning,
                                probabilities_and_hypotheses_for_words,
                            )
                            + self._words_to_hypotheses_and_scores[word][
                                matching_hypothesis
                            ]
                        )
                        # match.matching_subgraph is checked for None type
                        self._word_meaning_pairs_to_number_of_observations[  # type: ignore
                            (word, match.matching_subgraph)
                        ] += 1
                        if matching_hypothesis.check_isomorphism(  # type: ignore
                            match.matching_subgraph
                        ):
                            pass
                        else:
                            self._words_to_hypotheses_and_scores[word].pop(
                                matching_hypothesis
                            )
                    else:
                        new_hypothesis = PerceptionGraphPattern.from_graph(
                            meaning
                        ).perception_graph_pattern
                        self._words_to_hypotheses_and_scores[word][
                            new_hypothesis
                        ] = self.alignment_probability(
                            word, words, meaning, probabilities_and_hypotheses_for_words
                        )
                        self._word_meaning_pairs_to_number_of_observations[
                            (word, new_hypothesis)
                        ] += 1

                self.lexicon_step(word)

                if self._log_word_hypotheses_to:
                    self._log_hypotheses(word)

    def alignment_probability(
        self,
        word: str,
        words: Tuple[str, ...],
        meaning: PerceptionGraph,
        probabilities: Dict[str, Dict[PerceptionGraphPattern, float]],
    ) -> float:
        # Returns alignment probability as defined by the paper below
        # a^(t)(m|w, U) = p^(t-1)(m|w) / sum(for w' in U or {d})(p^(t-1)(m|w')),
        # where m|w is the word meaning pair, U is the utterance being analyzed, t is the time when
        # the occurence happened (simply using the probabilities present before we started analyzing
        # this utterance) {d} is a dummy word (added earlier in the program) which smoothes when the
        # meaning and word is present without it's counterpart.
        # https://onlinelibrary.wiley.com/doi/full/10.1111/j.1551-6709.2010.01104.x (1)
        normalizing_factor = 0.0
        for other_word in words:
            if other_word is not word:
                _, hypothesis = self._find_similar_hypothesis(
                    meaning, probabilities[other_word].keys()
                )
                if hypothesis:
                    normalizing_factor += probabilities[other_word][hypothesis]
        _, hypothesis = self._find_similar_hypothesis(meaning, probabilities[word].keys())
        # hypothesis should never be none since it is user to populate probabilities earlier
        return probabilities[word][hypothesis] / normalizing_factor  # type: ignore

    def get_probabilities(
        self, word: str, meanings: Iterable[PerceptionGraph]
    ) -> Dict[PerceptionGraphPattern, float]:
        # Update all word meaning probabilities for given word as defined by the paper below
        # p(m|w) = assoc(m, w) + lambda / sum(for m' in M)(assoc(m', w)) + (beta * lambda)
        # where w and m are given words and meanings, lambda is a smoothing factor, M is all
        # meanings encountered, beta is the expected number of meaning types.
        # https://onlinelibrary.wiley.com/doi/full/10.1111/j.1551-6709.2010.01104.x (3)
        probabilities: Dict[PerceptionGraphPattern, float] = dict()
        for meaning in meanings:
            if self._find_similar_hypothesis(
                meaning, self._words_to_hypotheses_and_scores[word].keys()
            ) == (None, None):
                new_hypothesis = PerceptionGraphPattern.from_graph(
                    meaning
                ).perception_graph_pattern
                normalizing_factor = (
                    self._expected_number_of_meanings * self._smoothing_parameter
                )
                for other_meaning in self._words_to_hypotheses_and_scores[word]:
                    normalizing_factor += self._words_to_hypotheses_and_scores[word][
                        other_meaning
                    ]
                probabilities[new_hypothesis] = (
                    self._smoothing_parameter / normalizing_factor
                )
        for hypothesis in self._words_to_hypotheses_and_scores[word]:
            normalizing_factor = 0
            for other_hypothesis in self._words_to_hypotheses_and_scores[word]:
                normalizing_factor += self._words_to_hypotheses_and_scores[word][
                    other_hypothesis
                ]
            normalizing_factor -= self._words_to_hypotheses_and_scores[word][hypothesis]
            probabilities[hypothesis] = (
                self._words_to_hypotheses_and_scores[word][hypothesis]
                + self._smoothing_parameter
            ) / (
                normalizing_factor
                + (self._expected_number_of_meanings * self._smoothing_parameter)
            )
        return probabilities

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

    def _compute_match_ratio(
        self, pattern: PerceptionGraphPattern, graph: PerceptionGraph
    ) -> "CrossSituationalLanguageLearner.PartialMatch":
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
        return CrossSituationalLanguageLearner.PartialMatch(
            hypothesis_pattern_common_subgraph,
            num_nodes_matched=num_nodes_matched,
            num_nodes_in_pattern=leading_hypothesis_num_nodes,
        )

    def lexicon_step(self, word: str) -> None:
        # If any conditional probability P(h^|w) exceeds a certain threshold value (h), then file
        # (w, h^) into the lexicon
        leading_hypothesis_entry = self._leading_hypothesis_for(word)
        if leading_hypothesis_entry:
            (
                leading_hypothesis_pattern,
                probability_of_meaning_given_word,
            ) = leading_hypothesis_entry

            times_word_meaning_pair_has_been_seen = self._word_meaning_pairs_to_number_of_observations[
                (word, leading_hypothesis_pattern)
            ]
            logging.info(
                "Prob of meaning given word: %s, Times seen: %s",
                probability_of_meaning_given_word,
                times_word_meaning_pair_has_been_seen,
            )
            # file (w, h^) into the lexicon

            # Check this as it may not be an issue for cross-situational
            # TODO: We sometimes prematurely lexicalize words, so we use this arbitrary counter
            #  threshold
            if probability_of_meaning_given_word > self._lexicon_entry_threshold:
                self._lexicon[word] = leading_hypothesis_pattern
                # Remove the word from hypotheses
                self._words_to_hypotheses_and_scores.pop(word)
                if self._graph_logger:
                    self._graph_logger.log_graph(
                        leading_hypothesis_pattern, logging.INFO, "Lexicalized %s", word
                    )

    @staticmethod
    def get_objects_from_perception(
        observed_perception_graph: PerceptionGraph
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
                    (leading_hypothesis, probability) = leading_hypothesis_pair
                    matcher = leading_hypothesis.matcher(
                        observed_perception_graph, matching_objects=True
                    )
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
                        descriptions.append((learned_description, probability))
        return immutabledict(descriptions)

    def _leading_hypothesis_for(
        self, word: str
    ) -> Optional[Tuple[PerceptionGraphPattern, float]]:
        hypotheses_and_probabilities_for_word = self.get_probabilities(word, [])
        while hypotheses_and_probabilities_for_word:
            leading = max(
                hypotheses_and_probabilities_for_word.items(), key=lambda entry: entry[1]
            )
            if self._word_meaning_pairs_to_number_of_observations[(word, leading[0])] < 5:
                logging.info(f"{word} | {leading[0]} pair not seen enough")
                hypotheses_and_probabilities_for_word.pop(leading[0])
            else:
                return leading
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

    def _find_similar_hypothesis(
        self,
        new_hypothesis: PerceptionGraph,
        candidates: Iterable[PerceptionGraphPattern],
    ) -> Tuple[
        Optional["CrossSituationalLanguageLearner.PartialMatch"],
        Optional[PerceptionGraphPattern],
    ]:
        """
        Finds the object in candidates most similar to new_hypothesis and returns it with the ratio.
        """
        candidates = iter(candidates)
        match = None
        while match is None:
            try:
                old_hypothesis = next(candidates)
            except StopIteration:
                return None, None
            try:
                match = self._compute_match_ratio(old_hypothesis, new_hypothesis)
            except RuntimeError:
                # Occurs when no matches of the pattern are found in the graph. This seems to
                # to indicate some full matches and some matches with no intersection at all
                pass
        for candidate in candidates:
            try:
                new_match = self._compute_match_ratio(candidate, new_hypothesis)
            except RuntimeError:
                # Occurs when no matches of the pattern are found in the graph. This seems to
                # to indicate some full matches and some matches with no intersection at all
                new_match = None
            if new_match and new_match.match_ratio() > match.match_ratio():
                match = new_match
                old_hypothesis = candidate
        if (
            match.match_ratio() >= self._graph_match_confirmation_threshold
            and match.matching_subgraph
        ):
            return match, old_hypothesis
        else:
            return None, None
