import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from random import Random
from adam.perception.perception_graph import IsOntologyNodePredicate
from adam.ontology.phase1_ontology import GAZED_AT
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    AbstractSet,
)

from attr import Factory, attrib, attrs
from attr.validators import in_, instance_of, optional
from immutablecollections import immutabledict, immutableset
from more_itertools import first
from vistautils.range import Range

from adam.learner import LanguagePerceptionSemanticAlignment
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.surface_templates import (
    SurfaceTemplate,
    SurfaceTemplateBoundToSemanticNodes,
)
from adam.learner.template_learner import (
    AbstractTemplateLearner,
    AbstractTemplateLearnerNew,
)
from adam.ontology.ontology import Ontology
from adam.perception.deprecated import LanguageAlignedPerception
from adam.perception.perception_graph import (
    DebugCallableType,
    GraphLogger,
    PerceptionGraph,
)
from adam.semantics import Concept


@attrs
class HypothesisLogger(GraphLogger):
    """
    Subclass of graph logger to generalize hypothesis logging
    """

    def log_hypothesis_graph(
        self,
        hypothesis: PerceptionGraphTemplate,
        level,
        msg: str,
        *args,
        match_correspondence_ids: Mapping[Any, str] = immutabledict(),
        graph_name: Optional[str] = None,
    ) -> None:
        if isinstance(hypothesis, PerceptionGraphTemplate):
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
class AbstractPursuitLearner(AbstractTemplateLearner, ABC):
    """
    An implementation of `LanguageLearner` for pursuit learning as a base for different pursuit based
    learners. Paper on Pursuit Learning Algorithm: https://www.ling.upenn.edu/~ycharles/papers/pursuit-final.pdf
    """

    _learned_item_to_hypotheses_and_scores: Dict[
        SurfaceTemplate, Dict[PerceptionGraphTemplate, float]
    ] = attrib(init=False, default=Factory(dict))
    _lexicon: Dict[SurfaceTemplate, PerceptionGraphTemplate] = attrib(
        init=False, default=Factory(dict)
    )
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

    _rng: Random = attrib(validator=instance_of(Random), kw_only=True)
    _debug_callback: Optional[DebugCallableType] = attrib(default=None, kw_only=True)

    # Learning factor (gamma) is the factor with which we update the hypotheses scores during
    # reinforcement.
    _learning_factor: float = attrib(default=0.1, kw_only=True)
    # We use this threshold to measure whether a new perception sufficiently matches a previous
    # hypothesis.
    _graph_match_confirmation_threshold: float = attrib(default=0.9, kw_only=True)
    # Threshold value for adding word to lexicon
    _lexicon_entry_threshold: float = attrib(default=0.8, kw_only=True)
    # Counter to be used to prevent prematurely lexicalizing novel words
    _learned_item_to_number_of_observations: Dict[SurfaceTemplate, int] = attrib(
        init=False, default=Factory(lambda: defaultdict(int))
    )
    _hypothesis_logger: Optional[HypothesisLogger] = attrib(
        validator=optional(instance_of(HypothesisLogger)), default=None, kw_only=True
    )
    debug_counter = 0

    # the following two fields are used if the user wishes the hypotheses for word meanings
    # to be logged at each step of learning for detailed debugging.
    _log_learned_item_hypotheses_to: Optional[Path] = attrib(
        validator=optional(instance_of(Path)), default=None, kw_only=True
    )
    _learned_item_to_logger: Dict[SurfaceTemplate, HypothesisLogger] = attrib(
        init=False, default=Factory(dict)
    )
    # Set of ids to track hypothesis that are rendered for graph logging
    _rendered_learned_item_hypothesis_pair_ids: Set[str] = attrib(
        init=False, default=Factory(set)
    )

    _observation_num = attrib(init=False, default=0)

    def _learning_step(
        self,
        preprocessed_input: LanguageAlignedPerception,
        surface_template: SurfaceTemplate,
    ) -> None:
        # We track this to prevent overly aggressive lexicalization.
        self._learned_item_to_number_of_observations[surface_template] += 1

        # If we are not already committed to the meaning of the word, go through learning steps:
        if surface_template not in self._lexicon:
            logging.info(f"Considering '{surface_template}'")
            if surface_template not in self._learned_item_to_hypotheses_and_scores:
                # This is the first time we have seen this word/phrase.
                self.initialization_step(surface_template, preprocessed_input)
            else:
                # We have seen this word/phrase before, so run the learning reinforcement step
                is_hypothesis_confirmed = self.learning_step(
                    surface_template, preprocessed_input
                )
                if is_hypothesis_confirmed:
                    self.maybe_lexicalize(surface_template)

            if self._log_learned_item_hypotheses_to:
                self._log_hypotheses(surface_template)

    def _primary_templates(
        self
    ) -> Iterable[Tuple[SurfaceTemplate, PerceptionGraphTemplate, float]]:
        return [
            (surface_template, graph_pattern, 1.0)
            for (surface_template, graph_pattern) in self._lexicon.items()
        ]

    def _fallback_templates(
        self
    ) -> Iterable[Tuple[SurfaceTemplate, PerceptionGraphTemplate, float]]:
        for (
            surface_template,
            graph_patterns_to_scores,
        ) in self._learned_item_to_hypotheses_and_scores.items():
            for (graph_pattern, score) in graph_patterns_to_scores.items():
                yield (surface_template, graph_pattern, score)

    def initialization_step(
        self,
        surface_template: SurfaceTemplate,
        aligned_perception: LanguageAlignedPerception,
    ):
        # If it's a novel word, learn a new hypothesis/pattern,
        # generated as a pattern graph from the perception.
        # We want h_0 = arg_min_(m in M_U) max(A_m); i.e. h_0 is pattern_hypothesis

        # exclude graphs for which we already have a mapping from our possibilities
        hypotheses: Sequence[PerceptionGraphTemplate] = [
            hypothesis
            for hypothesis in self._candidate_hypotheses(aligned_perception)
            if hypothesis not in self._lexicon.values()
        ]

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
                surface_template,
                self._learning_factor,
            )

        self._learned_item_to_hypotheses_and_scores[surface_template] = {
            pattern_hypothesis: self._learning_factor
        }

    def learning_step(
        self,
        surface_template: SurfaceTemplate,
        language_aligned_perception: LanguageAlignedPerception,
    ) -> bool:
        # Select the most probable meaning h for w
        # I.e., if we already have hypotheses, get the leading hypothesis and compare it with the
        # observed perception

        hypotheses_for_item = self._learned_item_to_hypotheses_and_scores[
            surface_template
        ]
        previous_hypotheses_and_scores = hypotheses_for_item
        leading_hypothesis_pattern = max(
            previous_hypotheses_and_scores,
            key=lambda key: previous_hypotheses_and_scores[key],
        )

        logging.info(
            "Current leading hypothesis is %s",
            abs(hash((surface_template, leading_hypothesis_pattern))),
        )

        current_hypothesis_score = hypotheses_for_item[leading_hypothesis_pattern]
        self.debug_counter += 1

        # If the leading hypothesis sufficiently matches the observation, reinforce it
        # To do, we check how much of the leading pattern hypothesis matches the perception
        partial_match = self._find_partial_match(
            leading_hypothesis_pattern, language_aligned_perception.perception_graph
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
            hypotheses_to_reward: List[PerceptionGraphTemplate] = []
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

                chosen_hypothesis = self._rng.choice(
                    [
                        hypothesis
                        for hypothesis in self._candidate_hypotheses(
                            language_aligned_perception
                        )
                        if hypothesis not in self._lexicon.values()
                    ]
                )
                hypotheses_to_reward.append(chosen_hypothesis)

                for hypothesis in hypotheses_for_item:
                    non_leading_hypothesis_partial_match = self._find_partial_match(
                        hypothesis, language_aligned_perception.perception_graph
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
            hypothesis_objects_boosted_on_this_update: Set[
                PerceptionGraphTemplate
            ] = set()
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
    class PartialMatch:
        """
        A class to hold hypothesis match information, such as the partial_match_hypothesis.
        *match_ratio* should be 1.0 exactly for a perfect match.
        """

        partial_match_hypothesis: Optional[PerceptionGraphTemplate] = attrib(kw_only=True)
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

    def maybe_lexicalize(self, surface_template: SurfaceTemplate) -> None:
        # If any conditional probability P(h^|w) exceeds a certain threshold value (h), then file
        # (w, h^) into the
        # lexicon
        # From Pursuit paper: P(h|w) = (A(w,h) + Gamma) / (Sum(A_w) + N x Gamma)
        leading_hypothesis_entry = self._leading_hypothesis_for(surface_template)
        assert leading_hypothesis_entry
        (leading_hypothesis_pattern, leading_hypothesis_score) = leading_hypothesis_entry

        all_hypotheses_for_word = self._learned_item_to_hypotheses_and_scores[
            surface_template
        ]
        sum_of_all_scores = sum(all_hypotheses_for_word.values())
        number_of_meanings = len(all_hypotheses_for_word)

        probability_of_meaning_given_word = (
            leading_hypothesis_score + self._smoothing_parameter
        ) / (sum_of_all_scores + number_of_meanings * self._smoothing_parameter)
        times_word_has_been_seen = self._learned_item_to_number_of_observations[
            surface_template
        ]
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
                self._lexicon[surface_template] = leading_hypothesis_pattern
                # Remove the word from hypotheses
                self._learned_item_to_hypotheses_and_scores.pop(surface_template)
                if self._hypothesis_logger:
                    self._hypothesis_logger.log_hypothesis_graph(
                        leading_hypothesis_pattern,
                        logging.INFO,
                        "Lexicalized %s",
                        surface_template,
                    )
            else:
                logging.info("Would lexicalize, but haven't see the word often enough")

    def _leading_hypothesis_for(
        self, item: SurfaceTemplate
    ) -> Optional[Tuple[PerceptionGraphTemplate, float]]:
        hypotheses_and_scores_for_word = self._learned_item_to_hypotheses_and_scores.get(
            item, None
        )
        if hypotheses_and_scores_for_word:
            return max(hypotheses_and_scores_for_word.items(), key=lambda entry: entry[1])
        else:
            return None

    def _log_hypotheses(self, item: SurfaceTemplate) -> None:
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

        def compute_hypothesis_id(h: PerceptionGraphTemplate) -> str:
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

    @abstractmethod
    def _candidate_hypotheses(
        self, language_aligned_perception: LanguageAlignedPerception
    ) -> Sequence[PerceptionGraphTemplate]:
        """
        Given a learning input, returns all possible meaning hypotheses.
        """

    @abstractmethod
    def _find_identical_hypothesis(
        self,
        new_hypothesis: PerceptionGraphTemplate,
        candidates: Iterable[PerceptionGraphTemplate],
    ) -> Optional[PerceptionGraphTemplate]:
        """
        Finds the first hypothesis object, if any, in *candidates*
        which is isomorphic to *new_hypothesis*.
        """

    @abstractmethod
    def _find_partial_match(
        self, hypothesis: PerceptionGraphTemplate, graph: PerceptionGraph
    ) -> "AbstractPursuitLearner.PartialMatch":
        """
        Compute the degree to which a meaning matches a perception.
        The resulting score should be between 0.0 (no match) and 1.0 (a perfect match)
        """

    @abstractmethod
    def _are_isomorphic(
        self, h: PerceptionGraphTemplate, hypothesis: PerceptionGraphTemplate
    ) -> bool:
        """
        Checks if two hypotheses are isomorphic.
        """


@attrs
class AbstractPursuitLearnerNew(AbstractTemplateLearnerNew, ABC):
    """
    An implementation of `TemplateLearnerNew` for pursuit learning as a base for different pursuit based
    learners. Paper on Pursuit Learning Algorithm: https://www.ling.upenn.edu/~ycharles/papers/pursuit-final.pdf
    """

    _concept_to_surface_template: Dict[Concept, SurfaceTemplate] = attrib(
        init=False, default=Factory(dict)
    )
    _surface_template_to_concept: Dict[SurfaceTemplate, Concept] = attrib(
        init=False, default=Factory(dict)
    )
    _known_bad_patterns: Set[SurfaceTemplate] = attrib(init=False, default=Factory(set))

    _concept_to_hypotheses_and_scores: Dict[
        Concept, Dict[PerceptionGraphTemplate, float]
    ] = attrib(init=False, default=Factory(dict))
    _lexicon: Dict[SurfaceTemplate, PerceptionGraphTemplate] = attrib(
        init=False, default=Factory(dict)
    )

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
    _learned_item_to_number_of_observations: Dict[SurfaceTemplate, int] = attrib(
        init=False, default=Factory(lambda: defaultdict(int))
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
    _learned_item_to_logger: Dict[SurfaceTemplate, HypothesisLogger] = attrib(
        init=False, default=Factory(dict)
    )
    # Set of ids to track hypothesis that are rendered for graph logging
    _rendered_learned_item_hypothesis_pair_ids: Set[str] = attrib(
        init=False, default=Factory(set)
    )

    _observation_num = attrib(init=False, default=0)

    rank_gaze_higher: bool = attrib(default=False)

    def _learning_step(
        self,
        language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,
    ) -> None:
        if bound_surface_template.surface_template in self._known_bad_patterns:
            # We tried to learn an alignment for this surface template previously
            # and it didn't work out.
            # For example, early on, we might think 'the' could be an object,
            # but eventually we will become quite sure it isn't one.
            return

        # We track this to prevent overly aggressive lexicalization.
        self._learned_item_to_number_of_observations[
            bound_surface_template.surface_template
        ] += 1

        # If we are not already committed to the meaning of the word, go through learning steps:
        if bound_surface_template.surface_template not in self._lexicon:
            logging.info(f"Considering '{bound_surface_template}'")
            if (
                bound_surface_template.surface_template
                not in self._surface_template_to_concept
            ):
                # This is the first time we have seen this word/phrase.
                self.initialization_step(
                    language_perception_semantic_alignment, bound_surface_template
                )
            else:
                # We have seen this word/phrase before, so run the learning reinforcement step
                is_hypothesis_confirmed = self.learning_step(
                    language_perception_semantic_alignment, bound_surface_template
                )
                if is_hypothesis_confirmed:
                    self.maybe_lexicalize(bound_surface_template.surface_template)

            if self._log_learned_item_hypotheses_to:
                self._log_hypotheses(bound_surface_template.surface_template)

    def remove_gaze_from_hypothesis(self, hypothesis: PerceptionGraphTemplate):
        """Removes any nodes that are gazed-at from a given hypothesis to help prevent this from affecting predictions"""
        nodes_to_remove = []
        for node in hypothesis.graph_pattern.copy_as_digraph().node:
            if (
                isinstance(node, IsOntologyNodePredicate)
                and node.property_value == GAZED_AT
            ):
                nodes_to_remove.append(node)
        if not nodes_to_remove:
            return hypothesis
        new_hypothesis = hypothesis
        for node in nodes_to_remove:
            new_hypothesis.graph_pattern._graph.remove_node(  # pylint: disable=protected-access
                node
            )
        return new_hypothesis

    def initialization_step(
        self,
        language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,
    ):
        # If it's a novel word, learn a new hypothesis/pattern,
        # generated as a pattern graph from the perception.
        # We want h_0 = arg_min_(m in M_U) max(A_m); i.e. h_0 is pattern_hypothesis

        # only consider those hypotheses for which we don't already have a mapping
        hypotheses: List[PerceptionGraphTemplate] = [
            hypothesis
            for hypothesis in self._hypotheses_from_perception(
                language_perception_semantic_alignment, bound_surface_template
            )
            if self.remove_gaze_from_hypothesis(hypothesis) not in self._lexicon.values()
        ]

        pattern_hypothesis = first(hypotheses)
        min_score = float("inf")
        # get all objects that have gaze
        gazed_at_hypotheses: List[PerceptionGraphTemplate] = []
        if self.rank_gaze_higher:
            for hypothesis in hypotheses:
                if GAZED_AT in [
                    node.property_value
                    for node in hypothesis.graph_pattern.copy_as_digraph().node
                    if isinstance(node, IsOntologyNodePredicate)
                ]:
                    gazed_at_hypotheses.append(hypothesis)
        # if there is only one object that has gaze, then this is the one that we consider -- we prioritize gaze above all else
        if len(gazed_at_hypotheses) == 1:
            pattern_hypothesis = first(gazed_at_hypotheses)
            min_score = max(
                [
                    s
                    for w, h_to_s in self._concept_to_hypotheses_and_scores.items()
                    for h, s in h_to_s.items()
                    if h.graph_pattern.check_isomorphism(pattern_hypothesis.graph_pattern)
                ]
                + [0]
            )
        # otherwise, we make our initial hypothesis the one with the least association with any other word
        else:
            hypotheses_to_consider: List[PerceptionGraphTemplate] = []
            # if there were no gazed objects, we consider all hypotheses
            if gazed_at_hypotheses:
                hypotheses_to_consider = gazed_at_hypotheses
            # if there were multiple gazed objects, we consider all objects with gaze
            else:
                hypotheses_to_consider = list(hypotheses)

            # Of the possible meanings for the word in this scene,
            # make our initial hypothesis the one with the least association
            # with any other word.
            for hypothesis in hypotheses_to_consider:
                # TODO Try to make this more efficient?
                max_association_score = max(
                    [
                        s
                        for w, h_to_s in self._concept_to_hypotheses_and_scores.items()
                        for h, s in h_to_s.items()
                        if h.graph_pattern.check_isomorphism(hypothesis.graph_pattern)
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
                bound_surface_template,
                self._learning_factor,
            )

        pattern_hypothesis = self.remove_gaze_from_hypothesis(pattern_hypothesis)

        # Create the mappings from concept to template and the new hypothesis
        concept = self._new_concept(
            debug_string=bound_surface_template.surface_template.to_short_string()
        )
        self._surface_template_to_concept[
            bound_surface_template.surface_template
        ] = concept
        self._concept_to_surface_template[
            concept
        ] = bound_surface_template.surface_template

        self._concept_to_hypotheses_and_scores[concept] = {
            pattern_hypothesis: self._learning_factor
        }

    def learning_step(
        self,
        language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,
    ) -> bool:
        # Select the most probable meaning h for w
        # I.e., if we already have hypotheses, get the leading hypothesis and compare it with the
        # observed perception

        hypotheses_for_item = self._concept_to_hypotheses_and_scores[
            self._surface_template_to_concept[bound_surface_template.surface_template]
        ]
        previous_hypotheses_and_scores = hypotheses_for_item
        leading_hypothesis_pattern = max(
            previous_hypotheses_and_scores,
            key=lambda key: previous_hypotheses_and_scores[key],
        )

        logging.info(
            "Current leading hypothesis is %s",
            abs(
                hash(
                    (bound_surface_template.surface_template, leading_hypothesis_pattern)
                )
            ),
        )

        current_hypothesis_score = hypotheses_for_item[leading_hypothesis_pattern]
        self.debug_counter += 1

        # If the leading hypothesis sufficiently matches the observation, reinforce it
        # To do, we check how much of the leading pattern hypothesis matches the perception
        partial_match = self._find_partial_match(
            leading_hypothesis_pattern,
            language_perception_semantic_alignment.perception_semantic_alignment.perception_graph,
        )

        # b.i) If the hypothesis is confirmed, we reinforce it.
        hypothesis_is_confirmed = partial_match.matched_exactly()
        if hypothesis_is_confirmed and partial_match.partial_match_hypothesis:
            logging.info("Current hypothesis is confirmed.")
            # Reinforce A(w,h)

            new_hypothesis_score = current_hypothesis_score + self._learning_factor * (
                1 - current_hypothesis_score
            )

            # if the hypothesis is confirmed and there is gaze, increase the reinforcement factor
            # TODO: tune how much this is reinforced https://github.com/isi-vista/adam/issues/734
            if self.rank_gaze_higher and GAZED_AT in [
                node.property_value
                for node in leading_hypothesis_pattern.graph_pattern.copy_as_digraph().node
                if isinstance(node, IsOntologyNodePredicate)
            ]:
                new_hypothesis_score = (
                    current_hypothesis_score
                    + self._learning_factor * (2 - current_hypothesis_score)
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

            def get_partial_match() -> Optional[PerceptionGraphTemplate]:
                if (
                    partial_match.match_score()
                    >= self._graph_match_confirmation_threshold
                    and partial_match.partial_match_hypothesis
                ):
                    logging.info(
                        "Introducing partial match as a new hypothesis; %s of %s nodes "
                        "matched.",
                        partial_match.num_nodes_matched,
                        partial_match.num_nodes_in_pattern,
                    )
                    return partial_match.partial_match_hypothesis
                else:
                    return None

            # This is where we differ from the pursuit paper.
            # First, we look for a node sufficiently close to our node as a new hypothesis
            # If a sufficiently close relaxed version of our pattern matches,
            # we used that relaxed version as the new hypothesis to introduce.
            # Otherwise, if we are prioritizing gazed-at object, we pick one of them, and if not, then we pick another
            # random object.

            hypotheses_to_reward: List[PerceptionGraphTemplate] = []
            # get all hypotheses which do not already have a mapping
            hypotheses = [
                hypothesis
                for hypothesis in self._hypotheses_from_perception(
                    language_perception_semantic_alignment, bound_surface_template
                )
                if self.remove_gaze_from_hypothesis(hypothesis)
                not in self._lexicon.values()
            ]

            # if there is an object that partially matches the object we're trying to learn, reward this one
            partial_possibility: Optional[PerceptionGraphTemplate] = get_partial_match()
            if partial_possibility:
                hypotheses_to_reward.append(partial_possibility)
            else:
                # if we are considering gaze, get the objects that are gazed at, then pick one of these at random
                gazed_at_possibilities = []
                if self.rank_gaze_higher:
                    for hypothesis in hypotheses:
                        if GAZED_AT in [
                            node.property_value
                            for node in hypothesis.graph_pattern.copy_as_digraph().node
                            if isinstance(node, IsOntologyNodePredicate)
                        ]:
                            gazed_at_possibilities.append(hypothesis)
                if gazed_at_possibilities:
                    hypotheses_to_reward.append(self._rng.choice(gazed_at_possibilities))
                # if there is no gaze or we aren't considering it, then pick any hypothesis at random
                else:
                    hypotheses_to_reward.append(self._rng.choice(list(hypotheses)))

            # Here's where it gets complicated.
            # In the Pursuit paper, at this point they choose a random meaning from the scene.
            # But if you do this it becomes difficult to learn object meanings
            # which are generalizations from the direct object observations.
            # Therefore, in addition to rewarding the hypothesis
            # which directly encodes the randomly selected object's perception,
            # we also reward all other non-leading hypotheses which would match it.

            for hypothesis in hypotheses_for_item:
                non_leading_hypothesis_partial_match = self._find_partial_match(
                    hypothesis,
                    language_perception_semantic_alignment.perception_semantic_alignment.perception_graph,
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
            hypothesis_objects_boosted_on_this_update: Set[
                PerceptionGraphTemplate
            ] = set()
            for hypothesis_to_reward in hypotheses_to_reward:
                # Because equality can't be computed straightforwardly between DiGraphs,
                # we can't just lookup the new_hypothesis in hypotheses_for_word
                # to determine if we've seen it before.
                # Instead, we need to do a more complicated check.
                if hypothesis_to_reward in hypotheses_for_item:
                    hypothesis_object_to_reward = hypothesis_to_reward
                else:
                    hypothesis_to_reward_without_gaze = self.remove_gaze_from_hypothesis(
                        hypothesis_to_reward
                    )
                    existing_hypothesis_matching_new_hypothesis = self._find_identical_hypothesis(
                        hypothesis_to_reward_without_gaze, candidates=hypotheses_for_item
                    )
                    if existing_hypothesis_matching_new_hypothesis:
                        hypothesis_object_to_reward = (
                            existing_hypothesis_matching_new_hypothesis
                        )
                        logging.info("Found existing matching hypothesis for new meaning")
                    else:
                        hypothesis_object_to_reward = hypothesis_to_reward

                hypothesis_object_to_reward_without_gaze = self.remove_gaze_from_hypothesis(
                    hypothesis_object_to_reward
                )
                if (
                    hypothesis_object_to_reward_without_gaze
                    not in hypothesis_objects_boosted_on_this_update
                ):

                    cur_score_for_new_hypothesis = hypotheses_for_item.get(
                        hypothesis_object_to_reward_without_gaze, 0.0
                    )
                    # if the object has gaze, we want to reinforce it more strongly than if it doesn't
                    # TODO: tune how much this is reinforced https://github.com/isi-vista/adam/issues/734
                    if self.rank_gaze_higher and GAZED_AT in [
                        node
                        for node in hypothesis_object_to_reward.graph_pattern.copy_as_digraph().node
                        if isinstance(node, IsOntologyNodePredicate)
                    ]:

                        hypotheses_for_item[hypothesis_object_to_reward_without_gaze] = (
                            cur_score_for_new_hypothesis
                            + self._learning_factor * (2.0 - cur_score_for_new_hypothesis)
                        )
                    else:
                        hypotheses_for_item[hypothesis_object_to_reward_without_gaze] = (
                            cur_score_for_new_hypothesis
                            + self._learning_factor * (1.0 - cur_score_for_new_hypothesis)
                        )
                    hypothesis_objects_boosted_on_this_update.add(
                        hypothesis_object_to_reward_without_gaze
                    )

        return hypothesis_is_confirmed

    @attrs(frozen=True)
    class PartialMatch:
        """
        A class to hold hypothesis match information, such as the partial_match_hypothesis.
        *match_ratio* should be 1.0 exactly for a perfect match.
        """

        partial_match_hypothesis: Optional[PerceptionGraphTemplate] = attrib(kw_only=True)
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

    def maybe_lexicalize(self, surface_template: SurfaceTemplate) -> None:
        # If any conditional probability P(h^|w) exceeds a certain threshold value (h), then file
        # (w, h^) into the
        # lexicon
        # From Pursuit paper: P(h|w) = (A(w,h) + Gamma) / (Sum(A_w) + N x Gamma)
        leading_hypothesis_entry = self._leading_hypothesis_for(surface_template)
        assert leading_hypothesis_entry
        (leading_hypothesis_pattern, leading_hypothesis_score) = leading_hypothesis_entry

        all_hypotheses_for_word = self._concept_to_hypotheses_and_scores[
            self._surface_template_to_concept[surface_template]
        ]
        sum_of_all_scores = sum(all_hypotheses_for_word.values())
        number_of_meanings = len(all_hypotheses_for_word)

        probability_of_meaning_given_word = (
            leading_hypothesis_score + self._smoothing_parameter
        ) / (sum_of_all_scores + number_of_meanings * self._smoothing_parameter)
        times_word_has_been_seen = self._learned_item_to_number_of_observations[
            surface_template
        ]
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
                self._lexicon[surface_template] = leading_hypothesis_pattern
                # Remove the word from hypotheses
                self._concept_to_hypotheses_and_scores.pop(
                    self._surface_template_to_concept[surface_template]
                )
                if self._hypothesis_logger:
                    self._hypothesis_logger.log_hypothesis_graph(
                        leading_hypothesis_pattern,
                        logging.INFO,
                        "Lexicalized %s",
                        surface_template,
                    )
            else:
                logging.info("Would lexicalize, but haven't see the word often enough")

    def _leading_hypothesis_for(
        self, surface_template: SurfaceTemplate
    ) -> Optional[Tuple[PerceptionGraphTemplate, float]]:
        hypotheses_and_scores_for_word = self._concept_to_hypotheses_and_scores.get(
            self._surface_template_to_concept[surface_template], None
        )
        if hypotheses_and_scores_for_word:
            return max(hypotheses_and_scores_for_word.items(), key=lambda entry: entry[1])
        else:
            return None

    def _log_hypotheses(self, surface_template: SurfaceTemplate) -> None:
        assert self._log_learned_item_hypotheses_to

        # if the user has asked us
        # to log the progress of the learner's hypotheses about word meanings,
        # then we use a GraphLogger per-word to write diagram's
        # of each word's hypotheses into their own sub-directory
        if surface_template in self._learned_item_to_logger:
            graph_logger = self._learned_item_to_logger[surface_template]
        else:
            log_directory_for_word = self._log_learned_item_hypotheses_to / str(
                surface_template
            )

            graph_logger = HypothesisLogger(
                log_directory=log_directory_for_word, enable_graph_rendering=True
            )
            self._learned_item_to_logger[surface_template] = graph_logger

        def compute_hypothesis_id(h: PerceptionGraphTemplate) -> str:
            # negative hashes cause the dot renderer to crash
            return str(abs(hash((surface_template, h))))

        if surface_template in self._lexicon:
            logging.info("The word %s has been lexicalized", surface_template)
            lexicalized_meaning = self._lexicon[surface_template]
            hypothesis_id = compute_hypothesis_id(lexicalized_meaning)
            if hypothesis_id not in self._rendered_learned_item_hypothesis_pair_ids:
                graph_logger.log_hypothesis_graph(
                    lexicalized_meaning,
                    logging.INFO,
                    "Rendering lexicalized " "meaning %s " "for %s",
                    hypothesis_id,
                    surface_template,
                    graph_name=str(hypothesis_id),
                )
                self._rendered_learned_item_hypothesis_pair_ids.add(hypothesis_id)
        else:
            scored_hypotheses_for_word = self._concept_to_hypotheses_and_scores[
                self._surface_template_to_concept[surface_template]
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
                        surface_template,
                        graph_name=str(hypothesis_id),
                    )
                    self._rendered_learned_item_hypothesis_pair_ids.add(hypothesis_id)

            logging.info(
                "After update, hypotheses for %s are %s",
                surface_template,
                ", ".join(
                    f"{compute_hypothesis_id(hypothesis)}={score}"
                    for (hypothesis, score) in reversed(
                        sorted(scored_hypotheses_for_word, key=lambda x: x[1])
                    )
                ),
            )

    @abstractmethod
    def _find_identical_hypothesis(
        self,
        new_hypothesis: PerceptionGraphTemplate,
        candidates: Iterable[PerceptionGraphTemplate],
    ) -> Optional[PerceptionGraphTemplate]:
        """
        Finds the first hypothesis object, if any, in *candidates*
        which is isomorphic to *new_hypothesis*.
        """

    @abstractmethod
    def _find_partial_match(
        self, hypothesis: PerceptionGraphTemplate, graph: PerceptionGraph
    ) -> "AbstractPursuitLearner.PartialMatch":
        """
        Compute the degree to which a meaning matches a perception.
        The resulting score should be between 0.0 (no match) and 1.0 (a perfect match)
        """

    def templates_for_concept(self, concept: Concept) -> AbstractSet[SurfaceTemplate]:
        if concept in self._concept_to_surface_template:
            return immutableset([self._concept_to_surface_template[concept]])
        else:
            return immutableset()

    @abstractmethod
    def _new_concept(self, debug_string: str) -> Concept:
        """
        Create a new `Concept` of the appropriate type with the given *debug_string*.
        """

    @abstractmethod
    def _keep_hypothesis(
        self,
        *,
        hypothesis: PerceptionGraphTemplate,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,
    ) -> bool:
        """
        Should a candidate hypothesis for the meaning of *bound_surface_template* be kept,
        or should it be discarded.

        Typically this is checking things like whether the hypothesis has gotten too small
        to plausibly encode the semantics of the template.

        This method may assume that the slots of *hypothesis* match
        the slots of the *bound_surface_template*;
        if not, the hypothesis would be automatically rejected.
        """

    @abstractmethod
    def _hypotheses_from_perception(
        self,
        learning_state: LanguagePerceptionSemanticAlignment,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,
    ) -> AbstractSet[PerceptionGraphTemplate]:
        """
        Get a hypothesis for the meaning of *surface_template* from a given *learning_state*.
        """

    def _primary_templates(
        self
    ) -> Iterable[Tuple[Concept, PerceptionGraphTemplate, float]]:
        return [
            (self._surface_template_to_concept[surface_template], graph_pattern, 1.0)
            for (surface_template, graph_pattern) in self._lexicon.items()
        ]

    def _fallback_templates(
        self
    ) -> Iterable[Tuple[Concept, PerceptionGraphTemplate, float]]:
        for (
            concept,
            graph_patterns_to_scores,
        ) in self._concept_to_hypotheses_and_scores.items():
            for (graph_pattern, score) in graph_patterns_to_scores.items():
                yield (concept, graph_pattern, score)
