from pathlib import Path
from typing import Counter, NamedTuple, Protocol, Tuple, Type, TypeVar

from attr import attrs, attrib, evolve
from attr.validators import instance_of
from immutablecollections import ImmutableSet, immutableset
from more_itertools import first
from networkx import DiGraph

from vistautils.iter_utils import only

from adam.learner import ApprenticeLearner, LanguagePerceptionSemanticAlignment, LanguageConceptAlignment
from adam.learner.objects import SubsetObjectLearner
from adam.ontology import OntologyNode
from adam.ontology.ontology import Ontology
from adam.perception import MatchMode
from adam.perception.perception_graph import (
    PerceptionGraphPattern, PerceptionGraphPatternFromGraph, PerceptionGraphPatternMatch, PerceptionGraph,
)
from adam.perception.perception_graph_nodes import PerceptionGraphNode
from adam.perception.perception_graph_predicates import CategoricalPredicate, NodePredicate
from adam.semantics import ObjectSemanticNode, Concept, SemanticNode


@attrs
class LanguagePerceptionSemanticContrast(NamedTuple):
    """
    Defines a contrasting pair of `LanguagePerceptionSemanticAlignment` s that a contrastive learner
    can learn from.

    By contrasting we mean the observations are of different *perceptions* and have different
    associated concepts or *language*.
    """

    first_alignment: LanguagePerceptionSemanticAlignment
    second_alignment: LanguagePerceptionSemanticAlignment


class ContrastiveLearner(Protocol):
    """
    A learner that can learn from a `LanguagePerceptionSemanticContrast`.

    Note that such learners are not expected to contribute to description at all. Rather they are
    meant to modify the behavior of other learners.
    """

    def learn_from(self, matching: LanguagePerceptionSemanticContrast) -> None:
        """
        Learn from the given pair of semantically-aligned inputs.
        """

    def log_hypotheses(self, log_output_path: Path) -> None:
        """
        Log some representation of the contrastive learner's hypotheses to the given log directory.
        """


@attrs
class TeachingContrastiveObjectLearner(Protocol):
    """
    A learner that learns contrasts between objects.

    This learner revolves around an apprentice. This learner expects to learn from the contrasting
    alignments produced by its apprenticed object learner. It learns in the sense that it pushes its
    understanding down into the apprentice.

    This implementation works by matching the perception graphs to the graphs and the graphs to each
    other, and counting (for each concept separately) the number of times we observe various nodes
    in a difference of two graphs vs. overall.

    (The difference here is the difference of node "sets" N(A) - N(B) where A is the perception
    graph the concept is present in and B is the other one.)

    We match the relevant patterns to each graph and we match the perceptions
    TODO fix this docstring it's unfinished
    Any time we observe a pattern node that matches to a graph node in the
    TODO fix this docstring it's unfinished

    We calculate weights for a semantic node as follows. First, count the number of times we've seen
    a
    TODO fix this docstring it's unfinished
    """

    apprentice: ApprenticeLearner = attrib(validator=instance_of(SubsetObjectLearner))
    _ontology: Ontology = attrib(validator=instance_of(Ontology))
    # These count the number of
    _ontology_node_present: Counter[Tuple[Concept, OntologyNode]] = attrib(validator=instance_of(Counter))
    _ontology_node_present_in_difference: Counter[Tuple[Concept, OntologyNode]] = attrib(validator=instance_of(Counter))
    _categorical_values_present: Counter[Tuple[Concept, str]] = attrib(validator=instance_of(Counter))
    _categorical_values_present_in_difference: Counter[Tuple[Concept, str]] = attrib(validator=instance_of(Counter))

    def learn_from(self, matching: LanguagePerceptionSemanticContrast) -> None:
        """
        Learn from the given pair of semantically-aligned inputs.
        """
        concept1, concept2 = _get_relevant_concepts(matching, node_type=ObjectSemanticNode)
        graph1_difference_nodes, graph2_difference_nodes = _get_difference_nodes(
            matching, ontology=self._ontology
        )
        for concept, difference_nodes in zip(
            [concept1, concept2], [graph1_difference_nodes, graph2_difference_nodes]
        ):
            pattern_to_graph_match = self._match_concept_pattern_to_graph(concept)
            self._update_counts(concept, pattern_to_graph_match, difference_nodes)

        for concept in [concept1, concept2]:
            self._propose_updated_hypothesis_to_apprentice(concept)

    def _match_concept_pattern_to_graph(
        self, concept: Concept, graph: PerceptionGraph
    ) -> PerceptionGraphPatternMatch:
        # Arbitrarily take the first match.
        #
        # Because there may be more than one match, this results in non-deterministic behavior in
        # general. We would like to do something smarter. See the GitHub issue:
        # TODO issue
        return first(
            self._get_apprentice_pattern(concept).matcher(
                graph,
                match_mode=MatchMode.OBJECT,
            ).matches(use_lookahead_pruning=True)
        )

    def log_hypotheses(self, log_output_path: Path) -> None:
        """
        Log some representation of the contrastive learner's hypotheses to the given log directory.
        """
        raise NotImplementedError()

    def _update_counts(
        self,
        concept: Concept,
        pattern_to_graph_match: PerceptionGraphPatternMatch,
        difference_nodes: ImmutableSet[PerceptionGraphNode],
    ) -> None:
        for pattern_node in pattern_to_graph_match.matched_pattern:
            # If it has an ontology node, do X
            # TODO figure out which things have ontology nodes
            if isinstance(pattern_node, ...):
                ontology_node = ...

                self._ontology_node_present[concept, ontology_node] += 1

                if pattern_to_graph_match.pattern_node_to_matched_graph_node[pattern_node] in difference_nodes:
                    self._ontology_node_present_in_difference[concept, ontology_node] += 1
            # Otherwise, if it's categorical, count the value observed
            elif isinstance(pattern_node, CategoricalPredicate):
                self._categorical_values_present[concept, pattern_node.value] += 1

                if pattern_to_graph_match.pattern_node_to_matched_graph_node[pattern_node] in difference_nodes:
                    self._categorical_values_present_in_difference[concept, pattern_node.value] += 1

    def _propose_updated_hypothesis_to_apprentice(self, concept: Concept) -> None:
        pattern = self._get_apprentice_pattern(concept)
        old_to_new_node = {}
        old_pattern_digraph = pattern.copy_as_digraph()
        new_pattern_digraph = DiGraph()

        for node, data in old_pattern_digraph.nodes(data=True):
            old_to_new_node[node] = evolve(node, weight=self._calculate_weight_for(concept, node))
            new_pattern_digraph.add_node(node, **data)

        for old_u, old_v, data in old_pattern_digraph.edges(data=True):
            new_pattern_digraph.add_edge(
                old_to_new_node[old_u], old_to_new_node[old_v], **data
            )

    def _calculate_weight_for(self, concept: Concept, node: NodePredicate) -> float:
        present_count = 1
        in_difference_count = 1

        # TODO make sure to ignore slot nodes
        # If it involves an ontology node, count that way
        # TODO implement
        if isinstance(node, ...):
            ontology_node = ...
            present_count = self._ontology_node_present.get((concept, ontology_node), 1)
            in_difference_count = self._ontology_node_present_in_difference.get(
                (concept, ontology_node), 1
            )
        # Otherwise, if it's categorical, count the value observed
        elif isinstance(node, CategoricalPredicate):
            present_count = self._categorical_values_present.get((concept, node.value), 1)
            in_difference_count = self._categorical_values_present_in_difference.get(
                (concept, node.value), 1
            )

        return in_difference_count / present_count

    def _get_apprentice_pattern(self, concept: Concept) -> PerceptionGraphPattern:
        concepts_to_patterns = self.apprentice.concepts_to_patterns()
        return concepts_to_patterns[concept]


NodeType = TypeVar("NodeType", bound=SemanticNode)


def _get_relevant_concepts(
    contrastive_pair: LanguagePerceptionSemanticContrast,
    *,
    node_type: Type[NodeType],
) -> Tuple[Concept, Concept]:
    return (
        # TODO these might be incorrect if they come from the learner. 🙃
        _get_only_semantic_node(
            contrastive_pair.first_alignment.language_concept_alignment, node_type=node_type
        ).concept,
        _get_only_semantic_node(
            contrastive_pair.second_alignment.language_concept_alignment, node_type=node_type
        ).concept
    )


def _get_only_semantic_node(
    alignment: LanguageConceptAlignment, *, node_type: Type[NodeType]
) -> NodeType:
    return only(
        aligned_node for aligned_node in iter(alignment.aligned_nodes)
        if isinstance(aligned_node, node_type)
    )


def _get_difference_nodes(
    matching: LanguagePerceptionSemanticContrast, *, ontology: Ontology
) -> Tuple[ImmutableSet[PerceptionGraphNode], ImmutableSet[PerceptionGraphNode]]:
    """
    Get the set of perception nodes in graph 1 but not graph 2, and the set in graph 2 but not 1.
    """
    graph1_as_pattern, perception_graph_match = _match_graphs_to_each_other(matching, ontology=ontology)

    return (
        immutableset([
            graph1_node
            for graph1_node in matching.first_alignment.perception_semantic_alignment.perception_graph
            if (
                graph1_as_pattern.perception_graph_node_to_pattern_node[graph1_node]
                not in perception_graph_match.matched_pattern
            )
        ]),
        immutableset([
            graph2_node
            for graph2_node in matching.second_alignment.perception_semantic_alignment.perception_graph
            if graph2_node not in perception_graph_match.matched_sub_graph
        ]),
    )


def _match_graphs_to_each_other(
    matching: LanguagePerceptionSemanticContrast, *, ontology: Ontology
) -> Tuple[PerceptionGraphPatternFromGraph, PerceptionGraphPatternMatch]:
    graph1_as_pattern = PerceptionGraphPattern.from_graph(
        matching.first_alignment.perception_semantic_alignment.perception_graph
    )
    perception_graph_match = graph1_as_pattern.perception_graph_pattern.matcher(
        matching.second_alignment.perception_semantic_alignment.perception_graph, match_mode=MatchMode.OBJECT
    ).relax_pattern_until_it_matches_getting_match(ontology=ontology, trim_after_match=None)
    return graph1_as_pattern, perception_graph_match
