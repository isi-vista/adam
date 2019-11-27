import logging
import random as r
from typing import Dict, Generic, Mapping, Tuple, Optional, List

import networkx
from attr import Factory, attrib, attrs
from immutablecollections import immutabledict
from more_itertools import first

from adam.language import (
    LinguisticDescriptionT,
    TokenSequenceLinguisticDescription,
    LinguisticDescription,
)
from adam.learner import LanguageLearner, LearningExample, graph_without_learner, get_largest_matching_pattern
from adam.perception import PerceptionT, PerceptualRepresentation, ObjectPerception
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception.perception_graph import (
    PerceptionGraph,
    PerceptionGraphPattern,
    DebugCallableType,
)

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
    _lexicon: Dict[str, PerceptionGraphPattern] = attrib(init=False, default=Factory(dict))
    _debug_callback: Optional[DebugCallableType] = attrib(default=None)

    # Learning factor (gamma) is the factor with which we update the hypotheses scores during reinforcement.
    _learning_factor: float = attrib(default=0.1, kw_only=True)
    # We use this threshold to measure whether a new perception sufficiently matches a previous hypothesis.
    _graph_match_confirmation_threshold: float = attrib(default=0.9, kw_only=True)
    # We use this threshold to check whether we should describe a scene using lexicpn word
    _describe_from_lexicon_threshold: float = attrib(default=0.9, kw_only=True)
    # Threshold value for adding word to lexicon
    _lexicon_entry_threshold: float = attrib(default=0.8, kw_only=True)

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

        self.learn_with_pursuit(observed_perception_graph, observed_linguistic_description)

    def learn_with_pursuit(self, observed_perception_graph: PerceptionGraph,
                           observed_linguistic_description: Tuple[str, ...]) -> None:
        logging.info(f"Pursuit learner observing {observed_linguistic_description}")
        # The learner’s words are W, meanings are M, their associations are A, and the new utterance is U = (W_U, M_U).
        # For every w in W_U
        for word in observed_linguistic_description:
            # If don't already know the meaning of the word, go through learning steps:
            if word not in self._lexicon:
                logging.info(f"Considering '{word}'")
                if word not in self._words_to_hypotheses_and_scores:
                    # a) Initialization step, if the word is a novel word
                    self.initialization_step(word, observed_perception_graph)
                else:
                    # b) If we already have a hypothesis, run the learning reinforcement step
                    is_hypothesis_confirmed = self.learning_step(word, observed_perception_graph)
                    # Try lexicon step if we confirmed a meaning
                    if is_hypothesis_confirmed:
                        self.lexicon_step(word)

    def initialization_step(self, word: str, observed_perception_graph: PerceptionGraph):
        # If it's a novel word, learn a new hypothesis/pattern,
        # generated as a pattern graph from the perception.
        # We want h_0 = arg_min_(m in M_U) max(A_m); i.e. h_0 is pattern_hypothesis
        meanings = self.get_meanings_from_perception(observed_perception_graph)
        pattern_hypothesis = first(meanings)
        min_score = float('inf')
        # TODO @Deniz: The association score makes no reference to meaning that I see
        # So won’t the hypothesis always just be either the first or last thing in meanings,
        # depending on the (unchanging) value  of max_association_score?
        for meaning in meanings:
            # Get the maximum association score for that meaning
            max_association_score = max([s for w, h_to_s in self._words_to_hypotheses_and_scores.items()
                                         for h, s in h_to_s.items()] + [0])
            if max_association_score < min_score:
                pattern_hypothesis = meaning

        self._words_to_hypotheses_and_scores[
            word
        ] = {pattern_hypothesis: self._learning_factor}

    def learning_step(self, word: str, observed_perception_graph: PerceptionGraph) -> bool:
        # Select the most probable meaning h for w
        # I.e., if we already have hypotheses, get the leading hypothesis and compare it with the observed perception
        previous_hypotheses_and_scores = self._words_to_hypotheses_and_scores[
            word
        ]
        leading_hypothesis_pattern = max(previous_hypotheses_and_scores,
                                         key=lambda key: previous_hypotheses_and_scores[key])
        # If the leading hypothesis sufficiently matches the observation, reinforce it
        # To do, we check how much of the leading pattern hypothesis matches the perception
        # TODO: It's worth double checking whether get_largest_matching_pattern works as intended - should we be getting
        #  at least 1 node match every time because there is guaranteed to be object-perception nodes?
        hypothesis_pattern_common_subgraph = get_largest_matching_pattern(
            leading_hypothesis_pattern,
            observed_perception_graph,
            debug_callback=self._debug_callback,
        )
        current_hypothesis_score = self._words_to_hypotheses_and_scores[
            word
        ][leading_hypothesis_pattern]
        print('Match:', len(hypothesis_pattern_common_subgraph.copy_as_digraph().nodes), '/',
              len(leading_hypothesis_pattern.copy_as_digraph().nodes))
        match_ratio = len(hypothesis_pattern_common_subgraph.copy_as_digraph().nodes) / \
                      len(leading_hypothesis_pattern.copy_as_digraph().nodes)

        # b.i) If the hypothesis is confirmed, we reinforce it.
        is_hypothesis_confirmed = match_ratio >= self._graph_match_confirmation_threshold
        if is_hypothesis_confirmed:
            # Reinforce A(w,h)
            new_hypothesis_score = current_hypothesis_score + self._learning_factor * (
                    1 - current_hypothesis_score)
            print('Reinforcing ', word, match_ratio, new_hypothesis_score, leading_hypothesis_pattern.copy_as_digraph().nodes)
        # b.ii) If the hypothesis is disconfirmed, so we weaken the previous score
        else:
            # Penalize A(w,h)
            new_hypothesis_score = current_hypothesis_score * (1 - self._learning_factor)
        # Register the updated hypothesis score of A(w,h)
        self._words_to_hypotheses_and_scores[
            word
        ][leading_hypothesis_pattern] = new_hypothesis_score

        # Reward A(w, h’) for a randomly selected h’ in M_U
        # TODO: Can we replace this with the initialization step?
        meanings = self.get_meanings_from_perception(observed_perception_graph)
        random_new_hypothesis = r.choice(meanings)
        self._words_to_hypotheses_and_scores[
            word
        ][random_new_hypothesis] = self._learning_factor
        return is_hypothesis_confirmed

    def lexicon_step(self, word: str) -> None:
        # If any conditional probability P(h^|w) exceeds a certain threshold value (h), then file (w, h^) into the
        # lexicon
        # From Pursuit paper: P(h|w) = (A(w,h) + Gamma) / (Sum(A_w) + N x Gamma)
        previous_hypotheses_and_scores = self._words_to_hypotheses_and_scores[
            word
        ]
        leading_hypothesis_pattern = max(previous_hypotheses_and_scores,
                                         key=lambda key: previous_hypotheses_and_scores[key])
        leading_hypothesis_score = previous_hypotheses_and_scores[leading_hypothesis_pattern]
        sum_of_all_scores = sum(previous_hypotheses_and_scores.values())
        number_of_meanings = len(previous_hypotheses_and_scores)

        probability_of_meaning_given_word = (leading_hypothesis_score + self._learning_factor) / \
                                            (sum_of_all_scores + number_of_meanings * self._learning_factor)
        # TODO: We sometimes prematurely lexicalize words, because the denominator is low in first rounds of training
        # file (w, h^) into the lexicon
        print('Lexicon prob:', probability_of_meaning_given_word, leading_hypothesis_pattern.copy_as_digraph().nodes)
        if probability_of_meaning_given_word > self._lexicon_entry_threshold:
            self._lexicon[word] = leading_hypothesis_pattern
            # Remove the word from hypotheses
            self._words_to_hypotheses_and_scores.pop(word)
            print('Lexicalized', word)

    @staticmethod
    def get_meanings_from_perception(observed_perception_graph: PerceptionGraph) -> List[PerceptionGraphPattern]:
        # TODO: Return all possible meanings. We currently remove ground,
        #  and return 2nd degree neighbor subgraphs of object perception nodes
        perception_as_digraph = observed_perception_graph.copy_as_digraph()
        perception_as_graph = perception_as_digraph.to_undirected()

        meanings = []
        for object_perception_root_node in [node for node in perception_as_graph.nodes
                                     if isinstance(node, ObjectPerception) and node.debug_handle != 'the ground']:
            # TODO: RMG - need different meaning generation strategy
            # we want whole sub-trees rooted at object perceptions
            if any([u == object_perception_root_node and data['label'] == 'partOf'
                    for u,v, data in perception_as_digraph.edges.data()]):
                continue
            first_neigbors = list(perception_as_graph.neighbors(object_perception_root_node))
            second_neighbors = []
            for node in first_neigbors:
                second_neighbors.extend(list(perception_as_graph.neighbors(node)))
            neighbors = [node for node in first_neigbors+second_neighbors if not isinstance(node, ObjectPerception)]
            subgraph = networkx.DiGraph(perception_as_digraph.subgraph([object_perception_root_node]+neighbors))
            subgraph.remove_nodes_from(list(networkx.isolates(subgraph)))
            meaning = PerceptionGraphPattern.from_graph(
                subgraph
            )
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

        # TODO: Discuss how to generate a description. Currently we pick the highest match, works with single objects
        learned_description = None
        largest_match_ratio = 0
        for word, meaning_pattern in self._lexicon.items():
            # get the largest common match
            common_pattern = get_largest_matching_pattern(
                meaning_pattern,
                observed_perception_graph,
                debug_callback=self._debug_callback,
            )
            match_ratio = len(common_pattern.copy_as_digraph().nodes) / \
                          len(meaning_pattern.copy_as_digraph().nodes)
            print(word, match_ratio, meaning_pattern.copy_as_digraph().nodes)
            if match_ratio > largest_match_ratio:
                learned_description = ('a', word)
                print(learned_description)
                largest_match_ratio = match_ratio
        if learned_description:
            return immutabledict(
                ((TokenSequenceLinguisticDescription(learned_description), 1.0),)
            )
        else:
            return immutabledict()
