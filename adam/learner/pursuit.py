import logging
from abc import abstractmethod, ABC
from pathlib import Path
from random import Random
from typing import (
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Sequence,
    Any,
)

from attr import Factory, attrib, attrs
from attr.validators import in_, instance_of, optional
from immutablecollections import immutabledict, ImmutableSet, immutableset
from more_itertools import first, flatten
from networkx import all_shortest_paths, DiGraph
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
    get_largest_matching_pattern,
    graph_without_learner,
)
from adam.learner.object_recognizer import ObjectRecognizer
from adam.learner.preposition_pattern import PrepositionPattern, _MODIFIED, _GROUND
from adam.learner.preposition_subset import PrepositionSurfaceTemplate
from adam.learner.verb_pattern import VerbPattern, _AGENT, _PATIENT, VerbSurfaceTemplate
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
    MatchedObjectNode,
    PerceptionGraphNode,
    _graph_node_order,
    MatchedObjectPerceptionPredicate,
    NodePredicate,
    TemporalScope,
)
from adam.utils import networkx_utils
from adam.utils.networkx_utils import digraph_with_nodes_sorted_by

# Abstract type to represent learned items such as words (str) and preposition phrases (PrepositionSurfaceTemplate)
LearnedItemT = TypeVar("LearnedItemT")
# Abstract type to represent hypothesis types
HypothesisT = TypeVar("HypothesisT")
HypothesisT2 = TypeVar("HypothesisT2")


@attrs(frozen=True, slots=True, eq=False)
class ObjectPattern:
    """
    Class representing an object hypothesis
    """

    graph_pattern: PerceptionGraphPattern = attrib(
        validator=instance_of(PerceptionGraphPattern)
    )


@attrs
class HypothesisLogger(GraphLogger):
    """
    Subclass of graph logger to generalize hypothesis logging
    """

    def log_hypothesis_graph(
        self,
        hypothesis: HypothesisT,
        level,
        msg: str,
        *args,
        match_correspondence_ids: Mapping[Any, str] = immutabledict(),
        graph_name: Optional[str] = None,
    ) -> None:
        if isinstance(hypothesis, ObjectPattern) or isinstance(
            hypothesis, PrepositionPattern
        ):
            graph = hypothesis.graph_pattern
        else:
            raise RuntimeError("Logging unknown hypothesis type")
        GraphLogger.log_graph(
            self,
            graph,
            level,
            msg,
            *args,
            match_correspondence_ids=match_correspondence_ids,
            graph_name=graph_name,
        )


@attrs
class AbstractPursuitLearner(
    Generic[LearnedItemT, HypothesisT, PerceptionT, LinguisticDescriptionT],
    LanguageLearner[PerceptionT, LinguisticDescription],
    ABC,
):
    """
    An implementation of `LanguageLearner` for pursuit learning as a base for different pursuit based
    learners. Paper on Pursuit Learning Algorithm: https://www.ling.upenn.edu/~ycharles/papers/pursuit-final.pdf
    The Pursuit Learner operatoes on two abstract types: 'LearnedItemT' represents learned items such as words (str),
    and 'HypothesisT' which represents hypothesis types such as PrepositionPatterns.
    """

    _learned_item_to_hypotheses_and_scores: Dict[
        LearnedItemT, Dict[HypothesisT, float]
    ] = attrib(init=False, default=Factory(dict))
    _lexicon: Dict[LearnedItemT, HypothesisT] = attrib(init=False, default=Factory(dict))
    _smoothing_parameter: float = attrib(
        validator=in_(Range.greater_than(0.0)), kw_only=True
    )
    """
    This smoothing factor is added to the scores of all hypotheses
    when forming a probability distribution over hypotheses.
    This should be a small value, at most 0.1 and possibly much less.
    See section 2.2 of the Pursuit paper.
    """
    _ontology: Ontology = attrib(validator=instance_of(Ontology), kw_only=True)

    _rng: Random = attrib(validator=instance_of(Random))
    _debug_callback: Optional[DebugCallableType] = attrib(default=None)

    # Learning factor (gamma) is the factor with which we update the hypotheses scores during
    # reinforcement.
    _learning_factor: float = attrib(default=0.1, kw_only=True)
    # We use this threshold to measure whether a new perception sufficiently matches a previous
    # hypothesis.
    _graph_match_confirmation_threshold: float = attrib(default=0.9, kw_only=True)
    # Threshold value for adding word to lexicon
    _lexicon_entry_threshold: float = attrib(default=0.8, kw_only=True)
    # Counter to be used to prevent prematurely lexicalizing novel words
    _learned_item_to_number_of_observations: Dict[LearnedItemT, int] = attrib(
        init=False, default=Factory(dict)
    )
    _hypothesis_logger: Optional[HypothesisLogger] = attrib(
        validator=optional(instance_of(HypothesisLogger)), default=None
    )
    debug_counter = 0

    # the following two fields are used if the user wishes the hypotheses for word meanings
    # to be logged at each step of learning for detailed debugging.
    _log_learned_item_hypotheses_to: Optional[Path] = attrib(
        validator=optional(instance_of(Path)), default=None
    )
    _learned_item_to_logger: Dict[LearnedItemT, HypothesisLogger] = attrib(
        init=False, default=Factory(dict)
    )
    # Set of ids to track hypothesis that are rendered for graph logging
    _rendered_learned_item_hypothesis_pair_ids: Set[str] = attrib(
        init=False, default=Factory(set)
    )

    _observation_num = attrib(init=False, default=0)

    def learn_with_pursuit(
        self,
        observed_perception_graph: PerceptionGraph,
        items_to_learn: Tuple[LearnedItemT, ...],
    ) -> None:
        logging.info(f"Pursuit learner observing {items_to_learn}")
        # The learner’s words are W, meanings are M, their associations are A, and the new
        # utterance is U = (W_U, M_U).
        # For every w in W_U
        for item in items_to_learn:
            # TODO: pursuit learner hard-coded to ignore determiners
            # https://github.com/isi-vista/adam/issues/498
            if item in ("a", "the"):
                continue
            if item in self._learned_item_to_number_of_observations:
                self._learned_item_to_number_of_observations[item] += 1
            else:
                self._learned_item_to_number_of_observations[item] = 1

            # If don't already know the meaning of the word, go through learning steps:
            if item not in self._lexicon:
                logging.info(f"Considering '{item}'")
                if item not in self._learned_item_to_hypotheses_and_scores:
                    # a) Initialization step, if the word is a novel word
                    self.initialization_step(item, observed_perception_graph)
                else:
                    # b) If we already have a hypothesis, run the learning reinforcement step
                    is_hypothesis_confirmed = self.learning_step(
                        item, observed_perception_graph
                    )
                    # Try lexicon step if we confirmed a meaning
                    if is_hypothesis_confirmed:
                        self.lexicon_step(item)

                if self._log_learned_item_hypotheses_to:
                    self._log_hypotheses(item)

    def initialization_step(
        self, item: LearnedItemT, observed_perception_graph: PerceptionGraph
    ):
        # If it's a novel word, learn a new hypothesis/pattern,
        # generated as a pattern graph from the perception.
        # We want h_0 = arg_min_(m in M_U) max(A_m); i.e. h_0 is pattern_hypothesis
        hypotheses: Sequence[HypothesisT] = self._candidate_hypotheses(
            observed_perception_graph
        )

        pattern_hypothesis = first(hypotheses)
        min_score = float("inf")
        # Of the possible meanings for the word in this scene,
        # make our initial hypothesis the one with the least association
        # with any other word.
        for hypothesis in hypotheses:
            # TODO Try to make this more efficient?
            max_association_score = max(
                [
                    s
                    for w, h_to_s in self._learned_item_to_hypotheses_and_scores.items()
                    for h, s in h_to_s.items()
                    if self._are_isomorphic(h, hypothesis)
                ]
                + [0]
            )
            if max_association_score < min_score:
                pattern_hypothesis = hypothesis
                min_score = max_association_score

        if self._hypothesis_logger:
            self._hypothesis_logger.log_hypothesis_graph(
                pattern_hypothesis,
                logging.INFO,
                "Initializing meaning for %s " "with score %s",
                item,
                self._learning_factor,
            )

        self._learned_item_to_hypotheses_and_scores[item] = {
            pattern_hypothesis: self._learning_factor
        }

    def learning_step(
        self, item: LearnedItemT, observed_perception_graph: PerceptionGraph
    ) -> bool:
        # Select the most probable meaning h for w
        # I.e., if we already have hypotheses, get the leading hypothesis and compare it with the
        # observed perception

        hypotheses_for_item = self._learned_item_to_hypotheses_and_scores[item]
        previous_hypotheses_and_scores = hypotheses_for_item
        leading_hypothesis_pattern = max(
            previous_hypotheses_and_scores,
            key=lambda key: previous_hypotheses_and_scores[key],
        )

        logging.info(
            "Current leading hypothesis is %s",
            abs(hash((item, leading_hypothesis_pattern))),
        )

        current_hypothesis_score = hypotheses_for_item[leading_hypothesis_pattern]
        self.debug_counter += 1

        # If the leading hypothesis sufficiently matches the observation, reinforce it
        # To do, we check how much of the leading pattern hypothesis matches the perception
        partial_match = self._find_partial_match(
            leading_hypothesis_pattern, observed_perception_graph
        )

        # b.i) If the hypothesis is confirmed, we reinforce it.
        hypothesis_is_confirmed = partial_match.matched_exactly()
        if hypothesis_is_confirmed and partial_match.partial_match_hypothesis:
            logging.info("Current hypothesis is confirmed.")
            # Reinforce A(w,h)
            new_hypothesis_score = current_hypothesis_score + self._learning_factor * (
                1 - current_hypothesis_score
            )

            # Register the updated hypothesis score of A(w,h)
            hypotheses_for_item[leading_hypothesis_pattern] = new_hypothesis_score
            logging.info("Updating hypothesis score to %s", new_hypothesis_score)
        # b.ii) If the hypothesis is disconfirmed, so we weaken the previous score
        else:
            # Penalize A(w,h)
            penalized_hypothesis_score = current_hypothesis_score * (
                1.0 - self._learning_factor
            )
            # Register the updated hypothesis score of A(w,h)
            hypotheses_for_item[leading_hypothesis_pattern] = penalized_hypothesis_score
            logging.info(
                "Working hypothesis disconfirmed. Reducing score from %s -> %s",
                current_hypothesis_score,
                penalized_hypothesis_score,
            )

            # This is where we differ from the pursuit paper.
            # If a sufficiently close relaxed version of our pattern matches,
            # we used that relaxed version as the new hypothesis to introduce
            hypotheses_to_reward: List[HypothesisT] = []
            if (
                partial_match.match_score() >= self._graph_match_confirmation_threshold
                and partial_match.partial_match_hypothesis
            ):
                logging.info(
                    "Introducing partial match as a new hypothesis; %s of %s nodes "
                    "matched.",
                    partial_match.num_nodes_matched,
                    partial_match.num_nodes_in_pattern,
                )
                # we know if partial_match_hypothesis is non-None above, it still will be.
                # we know if partial_match_hypothesis is non-None above, it still will be.
                hypotheses_to_reward.append(  # type: ignore
                    partial_match.partial_match_hypothesis
                )

            else:
                # Here's where it gets complicated.
                # In the Pursuit paper, at this point they choose a random meaning from the scene.
                # But if you do this it becomes difficult to learn object meanings
                # which are generalizations from the direct object observations.
                # Therefore, in addition to rewarding the hypothesis
                # which directly encodes the randomly selected object's perception,
                # we also reward all other non-leading hypotheses which would match it.
                logging.info(
                    "Choosing a random object from the scene to use as the word meaning"
                )

                perceptions = self._candidate_perceptions(observed_perception_graph)
                chosen_perception = self._rng.choice(perceptions)
                hypotheses_to_reward.append(
                    self._hypothesis_from_perception(chosen_perception)
                )

                for hypothesis in hypotheses_for_item:
                    non_leading_hypothesis_partial_match = self._find_partial_match(
                        hypothesis, chosen_perception
                    )
                    if (
                        non_leading_hypothesis_partial_match.match_score()
                        > self._graph_match_confirmation_threshold
                    ):
                        hypotheses_to_reward.append(hypothesis)
                        if self._hypothesis_logger:
                            self._hypothesis_logger.log_hypothesis_graph(
                                hypothesis,
                                logging.INFO,
                                "Boosting existing non-leading hypothesis",
                            )

            # Guard against the same object being rewarded more than once on the same update step.
            hypothesis_objects_boosted_on_this_update: Set[HypothesisT] = set()
            for hypothesis_to_reward in hypotheses_to_reward:
                # Because equality can't be computed straightforwardly between DiGraphs,
                # we can't just lookup the new_hypothesis in hypotheses_for_word
                # to determine if we've seen it before.
                # Instead, we need to do a more complicated check.
                if hypothesis_to_reward in hypotheses_for_item:
                    hypothesis_object_to_reward = hypothesis_to_reward
                else:
                    existing_hypothesis_matching_new_hypothesis = self._find_identical_hypothesis(
                        hypothesis_to_reward, candidates=hypotheses_for_item
                    )
                    if existing_hypothesis_matching_new_hypothesis:
                        hypothesis_object_to_reward = (
                            existing_hypothesis_matching_new_hypothesis
                        )
                        logging.info("Found existing matching hypothesis for new meaning")
                    else:
                        hypothesis_object_to_reward = hypothesis_to_reward

                if (
                    hypothesis_object_to_reward
                    not in hypothesis_objects_boosted_on_this_update
                ):
                    cur_score_for_new_hypothesis = hypotheses_for_item.get(
                        hypothesis_object_to_reward, 0.0
                    )
                    hypotheses_for_item[hypothesis_object_to_reward] = (
                        cur_score_for_new_hypothesis
                        + self._learning_factor * (1.0 - cur_score_for_new_hypothesis)
                    )
                    hypothesis_objects_boosted_on_this_update.add(
                        hypothesis_object_to_reward
                    )

        return hypothesis_is_confirmed

    @attrs(frozen=True)
    class PartialMatch(Generic[HypothesisT2]):
        """
        A class to hold hypothesis match information, such as the partial_match_hypothesis.
        *match_ratio* should be 1.0 exactly for a perfect match.
        """

        partial_match_hypothesis: Optional[HypothesisT2] = attrib(kw_only=True)
        num_nodes_matched: int = attrib(validator=instance_of(int), kw_only=True)
        num_nodes_in_pattern: int = attrib(validator=instance_of(int), kw_only=True)

        @abstractmethod
        def matched_exactly(self) -> bool:
            """
            Returns a boolean indicating whether the matching was an exact match
            """

        @abstractmethod
        def match_score(self) -> float:
            """
            Returns a score in [0.0, 1.0] where 0.0 indicates no match at all and 1.0 indicates a perfect match
            """

    def lexicon_step(self, item: LearnedItemT) -> None:
        # If any conditional probability P(h^|w) exceeds a certain threshold value (h), then file
        # (w, h^) into the
        # lexicon
        # From Pursuit paper: P(h|w) = (A(w,h) + Gamma) / (Sum(A_w) + N x Gamma)
        leading_hypothesis_entry = self._leading_hypothesis_for(item)
        assert leading_hypothesis_entry
        (leading_hypothesis_pattern, leading_hypothesis_score) = leading_hypothesis_entry

        all_hypotheses_for_word = self._learned_item_to_hypotheses_and_scores[item]
        sum_of_all_scores = sum(all_hypotheses_for_word.values())
        number_of_meanings = len(all_hypotheses_for_word)

        probability_of_meaning_given_word = (
            leading_hypothesis_score + self._smoothing_parameter
        ) / (sum_of_all_scores + number_of_meanings * self._smoothing_parameter)
        times_word_has_been_seen = self._learned_item_to_number_of_observations[item]
        logging.info(
            "Prob of meaning given word: %s, Times seen: %s",
            probability_of_meaning_given_word,
            times_word_has_been_seen,
        )
        # file (w, h^) into the lexicon

        # TODO: We sometimes prematurely lexicalize words, so we use this arbitrary counter
        #  threshold
        if probability_of_meaning_given_word > self._lexicon_entry_threshold:
            if times_word_has_been_seen > 5:
                self._lexicon[item] = leading_hypothesis_pattern
                # Remove the word from hypotheses
                self._learned_item_to_hypotheses_and_scores.pop(item)
                if self._hypothesis_logger:
                    self._hypothesis_logger.log_hypothesis_graph(
                        leading_hypothesis_pattern, logging.INFO, "Lexicalized %s", item
                    )
            else:
                logging.info("Would lexicalize, but haven't see the word often enough")

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

    @staticmethod
    def replace_template_variables_with_object_names(
        surface_template: Tuple[str, ...],
        object_variable_name_to_object_match_pattern_node: Mapping[
            str, MatchedObjectPerceptionPredicate
        ],
        pattern_node_to_aligned_perception_node: Mapping[
            NodePredicate, PerceptionGraphNode
        ],
        object_match_node_to_object_handle: Mapping[PerceptionGraphNode, str],
    ) -> Tuple[str, ...]:
        rtnr: List[str] = []
        # each entry in a verb surface object_match_node is either a token
        # (typically a preposition) or one of the two placeholders
        # e.g. AGENT and PATIENT
        for token_or_surface_template_variable in surface_template:
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

    def _leading_hypothesis_for(
        self, item: LearnedItemT
    ) -> Optional[Tuple[HypothesisT, float]]:
        hypotheses_and_scores_for_word = self._learned_item_to_hypotheses_and_scores.get(
            item, None
        )
        if hypotheses_and_scores_for_word:
            return max(hypotheses_and_scores_for_word.items(), key=lambda entry: entry[1])
        else:
            return None

    def _log_hypotheses(self, item: LearnedItemT) -> None:
        assert self._log_learned_item_hypotheses_to

        # if the user has asked us
        # to log the progress of the learner's hypotheses about word meanings,
        # then we use a GraphLogger per-word to write diagram's
        # of each word's hypotheses into their own sub-directory
        if item in self._learned_item_to_logger:
            graph_logger = self._learned_item_to_logger[item]
        else:
            log_directory_for_word = self._log_learned_item_hypotheses_to / str(item)

            graph_logger = HypothesisLogger(
                log_directory=log_directory_for_word, enable_graph_rendering=True
            )
            self._learned_item_to_logger[item] = graph_logger

        def compute_hypothesis_id(h: HypothesisT) -> str:
            # negative hashes cause the dot renderer to crash
            return str(abs(hash((item, h))))

        if item in self._lexicon:
            logging.info("The word %s has been lexicalized", item)
            lexicalized_meaning = self._lexicon[item]
            hypothesis_id = compute_hypothesis_id(lexicalized_meaning)
            if hypothesis_id not in self._rendered_learned_item_hypothesis_pair_ids:
                graph_logger.log_hypothesis_graph(
                    lexicalized_meaning,
                    logging.INFO,
                    "Rendering lexicalized " "meaning %s " "for %s",
                    hypothesis_id,
                    item,
                    graph_name=str(hypothesis_id),
                )
                self._rendered_learned_item_hypothesis_pair_ids.add(hypothesis_id)
        else:
            scored_hypotheses_for_word = self._learned_item_to_hypotheses_and_scores[
                item
            ].items()
            # First, make sure all the hypotheses have been rendered.
            # We use the hash of this pair to generate a unique ID to match up logging messages
            # to the PDFs of hypothesized meaning graphs.
            for (hypothesis, _) in scored_hypotheses_for_word:
                hypothesis_id = compute_hypothesis_id(hypothesis)
                if hypothesis_id not in self._rendered_learned_item_hypothesis_pair_ids:
                    graph_logger.log_hypothesis_graph(
                        hypothesis,
                        logging.INFO,
                        "Rendering  " "hypothesized " "meaning %s for %s",
                        hypothesis_id,
                        item,
                        graph_name=str(hypothesis_id),
                    )
                    self._rendered_learned_item_hypothesis_pair_ids.add(hypothesis_id)

            logging.info(
                "After update, hypotheses for %s are %s",
                item,
                ", ".join(
                    f"{compute_hypothesis_id(hypothesis)}={score}"
                    for (hypothesis, score) in reversed(
                        sorted(scored_hypotheses_for_word, key=lambda x: x[1])
                    )
                ),
            )

    def _candidate_hypotheses(
        self, observed_perception_graph: PerceptionGraph
    ) -> Sequence[HypothesisT]:
        """
        Given a perception graph, returns all possible meaning hypotheses of type HypothesisT in that graph.
        """
        return [
            self._hypothesis_from_perception(object_)
            for object_ in self._candidate_perceptions(observed_perception_graph)
        ]

    @abstractmethod
    def observe(
        self, learning_example: LearningExample[PerceptionT, LinguisticDescription]
    ) -> None:
        """
        A pursuit based implementation of the Language Learner observe method.
        """

    @abstractmethod
    def describe(
        self, perception: PerceptualRepresentation[PerceptionT]
    ) -> Mapping[LinguisticDescription, float]:
        """
        A pursuit based implementation of the Language Learner describe method.
        """

    @abstractmethod
    def _find_identical_hypothesis(
        self, new_hypothesis: HypothesisT, candidates: Iterable[HypothesisT]
    ) -> Optional[HypothesisT]:
        """
        Finds the first hypothesis object, if any, in *candidates*
        which is isomorphic to *new_hypothesis*.
        """

    @abstractmethod
    def _candidate_perceptions(
        self, observed_perception_graph: PerceptionGraph
    ) -> List[PerceptionGraph]:
        """
        Returns candidate perception graphs from a scene which might correspond to words of the type being learned.
        """

    @abstractmethod
    def _hypothesis_from_perception(self, perception: PerceptionGraph) -> HypothesisT:
        """
        Returns a meaning hypothesis representation of type *HypothesisT* for a given *perception*.
        """

    @abstractmethod
    def _find_partial_match(
        self, hypothesis: HypothesisT, graph: PerceptionGraph
    ) -> "AbstractPursuitLearner.PartialMatch[HypothesisT]":
        """
        Compute the degree to which a meaning matches a perception.
        The resulting score should be between 0.0 (no match) and 1.0 (a perfect match)
        """

    @abstractmethod
    def _are_isomorphic(self, h: HypothesisT, hypothesis: HypothesisT) -> bool:
        """
        Checks if two hypotheses are isomorphic.
        """


class ObjectPursuitLearner(
    Generic[PerceptionT, LinguisticDescriptionT],
    AbstractPursuitLearner[str, ObjectPattern, PerceptionT, LinguisticDescriptionT],
):
    """
    An implementation of pursuit learner for object recognition
    """

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

        self.learn_with_pursuit(
            observed_perception_graph, observed_linguistic_description
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

        descriptions = []

        for word, meaning_pattern in self._lexicon.items():
            # Use PerceptionGraphPattern.matcher and matcher.matches() for a complete match
            if self._matches(
                hypothesis=meaning_pattern,
                observed_perception_graph=observed_perception_graph,
            ):
                learned_description = TokenSequenceLinguisticDescription(("a", word))
                descriptions.append((learned_description, 1.0))

        if not descriptions:
            # no lexicalized word matched the perception,
            # but we can still try to match our leading hypotheses
            for word in self._learned_item_to_hypotheses_and_scores.keys():
                # mypy doesn't know the leading hypothesis will always exist here,
                # but we do.
                leading_hypothesis_pair = self._leading_hypothesis_for(  # type: ignore
                    word
                )
                if leading_hypothesis_pair:
                    (leading_hypothesis, score) = leading_hypothesis_pair
                    if self._matches(
                        hypothesis=leading_hypothesis,
                        observed_perception_graph=observed_perception_graph,
                    ):
                        learned_description = TokenSequenceLinguisticDescription(
                            ("a", word)
                        )
                        descriptions.append((learned_description, score))

        return immutabledict(descriptions)

    def _hypothesis_from_perception(self, perception: PerceptionGraph) -> ObjectPattern:
        return ObjectPattern(
            PerceptionGraphPattern.from_graph(perception).perception_graph_pattern
        )

    def _candidate_perceptions(self, observed_perception_graph) -> List[PerceptionGraph]:
        return self.get_objects_from_perception(observed_perception_graph)

    def _matches(
        self, *, hypothesis: ObjectPattern, observed_perception_graph: PerceptionGraph
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
    class ObjectHypothesisPartialMatch(
        AbstractPursuitLearner.PartialMatch[ObjectPattern]
    ):
        partial_match_hypothesis: Optional[ObjectPattern] = attrib(
            validator=optional(instance_of(ObjectPattern))
        )
        num_nodes_matched: int = attrib(validator=instance_of(int), kw_only=True)
        num_nodes_in_pattern: int = attrib(validator=instance_of(int), kw_only=True)

        def matched_exactly(self) -> bool:
            return self.num_nodes_matched == self.num_nodes_in_pattern

        def match_score(self) -> float:
            return self.num_nodes_matched / self.num_nodes_in_pattern

    def _find_partial_match(
        self, hypothesis: ObjectPattern, graph: PerceptionGraph
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
            ObjectPattern(hypothesis_pattern_common_subgraph)
            if hypothesis_pattern_common_subgraph
            else None,
            num_nodes_matched=num_nodes_matched,
            num_nodes_in_pattern=leading_hypothesis_num_nodes,
        )

    def _find_identical_hypothesis(
        self, new_hypothesis: ObjectPattern, candidates: Iterable[ObjectPattern]
    ) -> Optional[ObjectPattern]:
        for candidate in candidates:
            if new_hypothesis.graph_pattern.check_isomorphism(candidate.graph_pattern):
                return candidate
        return None

    def _are_isomorphic(self, h: ObjectPattern, hypothesis: ObjectPattern) -> bool:
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


class PrepositionPursuitLearner(
    Generic[PerceptionT, LinguisticDescriptionT],
    AbstractPursuitLearner[
        PrepositionSurfaceTemplate,
        PrepositionPattern,
        PerceptionT,
        LinguisticDescriptionT,
    ],
):
    """
    An implementation of pursuit learner for preposition leaning
    """

    # Variables for tracking preposition phrase information. These are filled in observe.
    object_match_node_for_ground: Optional[MatchedObjectNode] = None
    object_match_node_for_modified: Optional[MatchedObjectNode] = None
    template_variables_to_object_match_nodes: Optional[
        Iterable[Tuple[str, MatchedObjectNode]]
    ] = None

    def observe(
        self,
        learning_example: LearningExample[PerceptionT, LinguisticDescription],
        object_recognizer: Optional[ObjectRecognizer] = None,
    ) -> None:
        # We are also given the list of matched nodes in the perception
        # We can't learn prepositions without already knowing some objects in the learning instance
        perception = learning_example.perception
        if len(perception.frames) != 1:
            raise RuntimeError(
                "Preposition learner can only handle single frames for now"
            )
        if isinstance(perception.frames[0], DevelopmentalPrimitivePerceptionFrame):
            original_perception = PerceptionGraph.from_frame(perception.frames[0])
        else:
            raise RuntimeError("Cannot process perception type.")
        if not object_recognizer:
            raise RuntimeError("Preposition learner is missing object recognizer")

        observed_linguistic_description = (
            learning_example.linguistic_description.as_token_sequence()
        )

        # Convert the observed perception to a version with recognized objects
        recognized_object_perception = object_recognizer.match_objects(
            original_perception
        )

        # Get the match nodes and their word indices
        object_match_nodes = []
        token_indices_of_matched_object_words = []
        for (idx, token) in enumerate(observed_linguistic_description):
            if (
                token
                in recognized_object_perception.description_to_matched_object_node.keys()
            ):
                token_indices_of_matched_object_words.append(idx)
                object_match_nodes.append(
                    recognized_object_perception.description_to_matched_object_node[token]
                )
        if (
            len(object_match_nodes) != 2
            and len(token_indices_of_matched_object_words) != 2
        ):
            raise RuntimeError(
                f"Learning a preposition with more or less than two recognized objects is not currently supported. "
                f"Found {len(object_match_nodes)} from "
                f"{observed_linguistic_description}."
            )

        # TODO: Simplify this. It's currently copied from Subset for conveniece
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

        self.object_match_node_for_ground = object_match_node_for_ground
        self.object_match_node_for_modified = object_match_node_for_modified
        # This is the template_variables_to_object_match_nodes of sentence locations to pattern nodes
        self.template_variables_to_object_match_nodes = immutableset(
            [
                (_MODIFIED, object_match_node_for_modified),
                (_GROUND, object_match_node_for_ground),
            ]
        )

        self.learn_with_pursuit(
            observed_perception_graph=recognized_object_perception.perception_graph,
            items_to_learn=(preposition_surface_template,),
        )

    def describe(
        self,
        perception: PerceptualRepresentation[PerceptionT],
        object_recognizer: Optional[ObjectRecognizer] = None,
    ) -> Mapping[LinguisticDescription, float]:
        if len(perception.frames) != 1:
            raise RuntimeError(
                "Preposition learner can only handle single frames for now"
            )
        if isinstance(perception.frames[0], DevelopmentalPrimitivePerceptionFrame):
            original_perception = PerceptionGraph.from_frame(perception.frames[0])
        else:
            raise RuntimeError("Cannot process perception type.")
        if not object_recognizer:
            raise RuntimeError("Preposition learner is missing object recognizer")

        recognized_object_perception = object_recognizer.match_objects(
            original_perception
        )

        object_match_node_to_object_handle: Mapping[
            PerceptionGraphNode, str
        ] = immutabledict(
            (node, description)
            for description, node in recognized_object_perception.description_to_matched_object_node.items()
        )

        # this will be our output
        description_to_score: List[Tuple[TokenSequenceLinguisticDescription, float]] = []

        # For each preposition we've learned
        for (preposition_surface_template, preposition_pattern) in self._lexicon.items():
            # try to see if (our model of) its semantics is present in the situation.
            matcher = preposition_pattern.graph_pattern.matcher(
                recognized_object_perception.perception_graph, matching_objects=False
            )
            for match in matcher.matches(use_lookahead_pruning=True):
                # if it is, use that preposition to describe the situation.
                description_to_score.append(
                    (
                        TokenSequenceLinguisticDescription(
                            # we generate the description by taking the preposition surface template
                            # which has MODIFIER and GROUND variables,
                            # and replacing those variables by the actual names
                            # of the matched objects.
                            PrepositionPursuitLearner.replace_template_variables_with_object_names(
                                preposition_surface_template,
                                preposition_pattern.object_variable_name_to_pattern_node,
                                match.pattern_node_to_matched_graph_node,
                                object_match_node_to_object_handle,
                            )
                        ),
                        1.0,
                    )
                )

        return immutabledict(description_to_score)

    def _hypothesis_from_perception(
        self, perception: PerceptionGraph
    ) -> PrepositionPattern:
        if self.template_variables_to_object_match_nodes:
            return PrepositionPattern.from_graph(
                perception.copy_as_digraph(),
                self.template_variables_to_object_match_nodes,
            )
        else:
            raise RuntimeError(
                "Empty template variables:", self.template_variables_to_object_match_nodes
            )

    def _candidate_perceptions(self, observed_perception_graph) -> List[PerceptionGraph]:
        # The directions of edges in the perception graph are not necessarily meaningful
        # from the point-of-view of hypothesis generation, so we need an undirected copy
        # of the graph.
        perception_graph = observed_perception_graph.copy_as_digraph()
        # as_view=True loses s
        perception_graph_as_undirected = observed_perception_graph.copy_as_digraph().to_undirected(
            as_view=False
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
                    self.object_match_node_for_ground,
                    self.object_match_node_for_modified,
                )
            )
        )

        # Along the core of our hypothesis we also want to collect the predecessors and successors
        hypothesis_nodes_mutable = []
        for node in hypothesis_spine_nodes:
            if node not in [
                self.object_match_node_for_ground,
                self.object_match_node_for_modified,
            ]:
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
            networkx_utils.subgraph(
                observed_perception_graph.copy_as_digraph(), nodes=hypothesis_nodes
            ),
            _graph_node_order,
        )
        return [PerceptionGraph(preposition_pattern_graph)]

    @attrs(frozen=True)
    class PrepositionHypothesisPartialMatch(
        AbstractPursuitLearner.PartialMatch[PrepositionPattern]
    ):
        partial_match_hypothesis: Optional[PrepositionPattern] = attrib(
            validator=optional(instance_of(PrepositionPattern))
        )
        num_nodes_matched: int = attrib(validator=instance_of(int), kw_only=True)
        num_nodes_in_pattern: int = attrib(validator=instance_of(int), kw_only=True)

        def matched_exactly(self) -> bool:
            return self.num_nodes_matched == self.num_nodes_in_pattern

        def match_score(self) -> float:
            return self.num_nodes_matched / self.num_nodes_in_pattern

    def _find_partial_match(
        self, hypothesis: PrepositionPattern, graph: PerceptionGraph
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
            partial_hypothesis: Optional[PrepositionPattern] = PrepositionPattern(
                graph_pattern=hypothesis_pattern_common_subgraph,
                object_variable_name_to_pattern_node=hypothesis.object_variable_name_to_pattern_node,
            )
        else:
            partial_hypothesis = None

        return PrepositionPursuitLearner.PrepositionHypothesisPartialMatch(
            partial_hypothesis,
            num_nodes_matched=num_nodes_matched,
            num_nodes_in_pattern=leading_hypothesis_num_nodes,
        )

    def _find_identical_hypothesis(
        self, new_hypothesis: PrepositionPattern, candidates: Iterable[PrepositionPattern]
    ) -> Optional[PrepositionPattern]:
        """
        Finds the first hypothesis object, if any, in *candidates*
        which is isomorphic to *new_hypothesis*.
        """
        for candidate in candidates:
            if self._are_isomorphic(new_hypothesis, candidate):
                return candidate
        return None

    def _are_isomorphic(
        self, h: PrepositionPattern, hypothesis: PrepositionPattern
    ) -> bool:
        # Check mapping equality of preposition patterns
        first_mapping = h.object_variable_name_to_pattern_node
        second_mapping = hypothesis.object_variable_name_to_pattern_node
        are_equal_mappings = len(first_mapping) == len(second_mapping) and all(
            k in second_mapping and second_mapping[k].is_equivalent(v)
            for k, v in first_mapping.items()
        )
        return are_equal_mappings and h.graph_pattern.check_isomorphism(
            hypothesis.graph_pattern
        )


class VerbPursuitLearner(
    Generic[PerceptionT, LinguisticDescriptionT],
    AbstractPursuitLearner[
        VerbSurfaceTemplate, VerbPattern, PerceptionT, LinguisticDescriptionT
    ],
):
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

        observed_linguistic_description = (
            learning_example.linguistic_description.as_token_sequence()
        )

        # Convert the observed perception to a version with recognized objects
        recognized_object_perception = object_recognizer.match_objects(
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

    def describe(
        self,
        perception: PerceptualRepresentation[PerceptionT],
        object_recognizer: Optional[ObjectRecognizer] = None,
    ) -> Mapping[LinguisticDescription, float]:
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

        recognized_object_perception = object_recognizer.match_objects(
            original_perception
        )

        object_match_node_to_object_handle: Mapping[
            PerceptionGraphNode, str
        ] = immutabledict(
            (node, description)
            for description, node in recognized_object_perception.description_to_matched_object_node.items()
        )

        # this will be our output
        description_to_score: List[Tuple[TokenSequenceLinguisticDescription, float]] = []

        # For each verb we've learned
        for (verb_surface_template, verb_pattern) in self._lexicon.items():
            # try to see if (our model of) its semantics is present in the situation.
            matcher = verb_pattern.graph_pattern.matcher(
                recognized_object_perception.perception_graph, matching_objects=False
            )
            for match in matcher.matches(use_lookahead_pruning=True):
                # if it is, use that verb to describe the situation.
                description_to_score.append(
                    (
                        TokenSequenceLinguisticDescription(
                            # we generate the description by taking the preposition surface template
                            # which has AGENT and PATIENT variables,
                            # and replacing those variables by the actual names
                            # of the matched objects.
                            VerbPursuitLearner.replace_template_variables_with_object_names(
                                verb_surface_template,
                                verb_pattern.object_variable_name_to_pattern_node,
                                match.pattern_node_to_matched_graph_node,
                                object_match_node_to_object_handle,
                            )
                        ),
                        1.0,
                    )
                )

        return immutabledict(description_to_score)

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
    class VerbHypothesisPartialMatch(AbstractPursuitLearner.PartialMatch[VerbPattern]):
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
