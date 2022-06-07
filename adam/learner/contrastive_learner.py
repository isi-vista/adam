import json
import logging
from abc import abstractmethod
from pathlib import Path
from typing import (
    AbstractSet,
    Any,
    Counter,
    Mapping,
    MutableMapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

from more_itertools import first
from typing_extensions import Protocol

from attr import attrs, attrib, evolve
from attr.validators import instance_of
from immutablecollections import ImmutableSet, immutableset
from networkx import DiGraph

from adam.language import LinguisticDescriptionT
from adam.learner import (
    ApprenticeLearner,
    LanguagePerceptionSemanticAlignment,
    SurfaceTemplate,
)
from adam.learner.objects import SubsetObjectLearner
from adam.ontology import OntologyNode
from adam.ontology.ontology import Ontology
from adam.perception import MatchMode
from adam.perception.perception_graph import (
    PerceptionGraphPattern,
    PerceptionGraphPatternFromGraph,
    PerceptionGraphPatternMatch,
    PerceptionGraph,
)
from adam.perception.perception_graph_nodes import PerceptionGraphNode
from adam.perception.perception_graph_predicates import (
    CategoricalPredicate,
    IsOntologyNodePredicate,
    NodePredicate,
    StrokeGNNRecognitionPredicate,
)
from adam.semantics import Concept, SemanticNode


logger = logging.getLogger(__name__)


class LanguagePerceptionSemanticContrast(NamedTuple):
    """
    Defines a contrasting pair of `LanguagePerceptionSemanticAlignment` s that a contrastive learner
    can learn from.

    By contrasting we mean the observations are of different *perceptions* and have different
    associated concepts or *language*.
    """

    first_alignment: LanguagePerceptionSemanticAlignment
    second_alignment: LanguagePerceptionSemanticAlignment

    def perception_graphs(self) -> Tuple[PerceptionGraph, PerceptionGraph]:
        return (
            self.first_alignment.perception_semantic_alignment.perception_graph,
            self.second_alignment.perception_semantic_alignment.perception_graph,
        )


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


class ApprenticeTemplateLearner(ApprenticeLearner, Protocol):
    @abstractmethod
    def templates_for_concept(self, concept: Concept) -> AbstractSet[SurfaceTemplate]:
        """
        Return the set of templates for the given concept.
        """


@attrs
class TeachingContrastiveObjectLearner(ContrastiveLearner):
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

    We weight only nodes with ontology nodes, and categoricals. We calculate weights for these in
    similar ways: Counting by the assoicated ontology node (nodes with ontology nodes), and counting
    by value (categoricals). The weight for each node is

        #(times we've seen this in the difference) / #(times we've seen this)
    """

    apprentice: ApprenticeTemplateLearner = attrib(
        validator=instance_of(SubsetObjectLearner)
    )
    _ontology: Ontology = attrib(validator=instance_of(Ontology))
    observations: int = attrib(validator=instance_of(int), default=0, init=False)
    # These count the number of
    _stroke_gnn_recognition_nodes_present: Counter[Tuple[Concept, str]] = attrib(
        validator=instance_of(Counter), factory=Counter, init=False
    )
    _stroke_gnn_recognition_nodes_present_in_difference: Counter[
        Tuple[Concept, str]
    ] = attrib(validator=instance_of(Counter), factory=Counter, init=False)
    _ontology_node_present: Counter[Tuple[Concept, OntologyNode]] = attrib(
        validator=instance_of(Counter), factory=Counter, init=False
    )
    _ontology_node_present_in_difference: Counter[Tuple[Concept, OntologyNode]] = attrib(
        validator=instance_of(Counter), factory=Counter, init=False
    )
    _categorical_values_present: Counter[Tuple[Concept, str]] = attrib(
        validator=instance_of(Counter), factory=Counter, init=False
    )
    _categorical_values_present_in_difference: Counter[Tuple[Concept, str]] = attrib(
        validator=instance_of(Counter), factory=Counter, init=False
    )

    def learn_from(self, matching: LanguagePerceptionSemanticContrast) -> None:
        """
        Learn from the given pair of semantically-aligned inputs.
        """
        self.observations += 1
        concept1, concept2 = _get_relevant_concepts_from_learner(
            matching, learner=self.apprentice
        )
        logger.info(
            "Learning from contrastive observation %d (%s, %s)",
            self.observations,
            concept1,
            concept2,
        )
        graph1_difference_nodes, graph2_difference_nodes = _get_difference_nodes(
            matching, ontology=self._ontology
        )
        for concept, graph, difference_nodes in zip(
            [concept1, concept2],
            matching.perception_graphs(),
            [graph1_difference_nodes, graph2_difference_nodes],
        ):
            temp = self._match_concept_pattern_to_multiple_graphs(concept, graph)
            pattern_to_graph_match = self._match_concept_pattern_to_primary_graph(
                concept, graph
            )
            if pattern_to_graph_match:
                self._update_counts(concept, pattern_to_graph_match, difference_nodes)
            else:
                logger.debug(
                    "Could not update counts for concept %s: Failed to match pattern to perception "
                    "graph.",
                    concept,
                )

        for concept in [concept1, concept2]:
            self._propose_updated_hypothesis_to_apprentice(concept)

    def _match_concept_pattern_to_multiple_graphs(
        self, concept: Concept, graph: PerceptionGraph, top_n: Optional[int] = None
    ) -> Optional[Mapping[PerceptionGraphPatternMatch, PerceptionGraphPatternMatch]]:
        # Arbitrarily take the first match per hypothesis.
        #
        # Because there may be more than one match per hypothesis, this results in non-deterministic behavior in
        # general. We would like to do something smarter. See the GitHub issue:
        # https://github.com/isi-vista/adam/issues/1140
        # TODO issue
        return {
            apprentice_concept.graph_pattern: apprentice_concept.graph_pattern.matcher(
                graph,
                match_mode=MatchMode.OBJECT,
            ).relax_pattern_until_it_matches_getting_match(
                ontology=self._ontology, trim_after_match=None
            )
            for apprentice_concept in self.apprentice.concept_to_hypotheses(
                concept, top_n
            )
        }

    def _match_concept_pattern_to_primary_graph(
        self, concept: Concept, graph: PerceptionGraph
    ) -> Optional[PerceptionGraphPatternMatch]:
        # Arbitrarily take the first match.
        #
        # Because there may be more than one match, this results in non-deterministic behavior in
        # general. We would like to do something smarter. See the GitHub issue:
        # TODO issue
        return (
            self._get_apprentice_pattern(concept)
            .matcher(
                graph,
                match_mode=MatchMode.OBJECT,
            )
            .relax_pattern_until_it_matches_getting_match(
                ontology=self._ontology, trim_after_match=None
            )
        )

    def log_hypotheses(self, log_output_path: Path) -> None:
        """
        Log some representation of the contrastive learner's hypotheses to the given log directory.
        """
        logging.info(
            "Logging %d concept counts to %s",
            max(
                len(self._ontology_node_present),
                len(self._stroke_gnn_recognition_nodes_present),
                len(self._categorical_values_present),
            ),
            log_output_path,
        )
        counts = {
            "stroke_gnn_recognition_node_present": _make_counter_json_ready(
                self._stroke_gnn_recognition_nodes_present
            ),
            "stroke_gnn_recognition_node_present_in_difference": _make_counter_json_ready(
                self._stroke_gnn_recognition_nodes_present_in_difference
            ),
            "ontology_node_present": _make_counter_json_ready(
                self._ontology_node_present
            ),
            "ontology_node_present_in_difference": _make_counter_json_ready(
                self._ontology_node_present_in_difference
            ),
            "categorical_present": _make_counter_json_ready(
                self._categorical_values_present
            ),
            "categorical_present_in_difference": _make_counter_json_ready(
                self._categorical_values_present_in_difference
            ),
        }
        with open(
            log_output_path / "contrastive_object_learning_counts.json",
            encoding="utf-8",
            mode="w",
        ) as json_out:
            json.dump(counts, json_out)
            json_out.write("\n")

    def _update_counts(
        self,
        concept: Concept,
        pattern_to_graph_match: PerceptionGraphPatternMatch,
        difference_nodes: ImmutableSet[PerceptionGraphNode],
    ) -> None:
        for pattern_node in pattern_to_graph_match.matched_pattern:
            # If it's a stroke GNN recognition node, count the "recognized object" string observed
            if isinstance(pattern_node, StrokeGNNRecognitionPredicate):
                self._stroke_gnn_recognition_nodes_present[
                    concept, pattern_node.recognized_object
                ] += 1

                if (
                    pattern_to_graph_match.pattern_node_to_matched_graph_node[
                        pattern_node
                    ]
                    in difference_nodes
                ):
                    self._stroke_gnn_recognition_nodes_present_in_difference[
                        concept, pattern_node.recognized_object
                    ] += 1
            # If it has an ontology node, do X
            elif isinstance(pattern_node, IsOntologyNodePredicate):
                self._ontology_node_present[concept, pattern_node.property_value] += 1

                if (
                    pattern_to_graph_match.pattern_node_to_matched_graph_node[
                        pattern_node
                    ]
                    in difference_nodes
                ):
                    self._ontology_node_present_in_difference[
                        concept, pattern_node.property_value
                    ] += 1
            # Otherwise, if it's categorical, count the value observed
            elif isinstance(pattern_node, CategoricalPredicate):
                self._categorical_values_present[concept, pattern_node.value] += 1

                if (
                    pattern_to_graph_match.pattern_node_to_matched_graph_node[
                        pattern_node
                    ]
                    in difference_nodes
                ):
                    self._categorical_values_present_in_difference[
                        concept, pattern_node.value
                    ] += 1

    def _propose_updated_hypothesis_to_apprentice(self, concept: Concept) -> None:
        pattern = self._get_apprentice_pattern(concept)
        old_to_new_node = {}
        old_pattern_digraph = pattern.copy_as_digraph()
        new_pattern_digraph = DiGraph()

        for node, data in old_pattern_digraph.nodes(data=True):
            old_to_new_node[node] = self._re_weighted_node(concept, node)
            new_pattern_digraph.add_node(old_to_new_node[node], **data)

        for old_u, old_v, data in old_pattern_digraph.edges(data=True):
            new_pattern_digraph.add_edge(
                old_to_new_node[old_u], old_to_new_node[old_v], **data
            )

        new_pattern = PerceptionGraphPattern(new_pattern_digraph, dynamic=pattern.dynamic)
        self.apprentice.propose_updated_hypotheses({concept: new_pattern})

    def _re_weighted_node(self, concept: Concept, node: NodePredicate) -> NodePredicate:
        """
        Return a re-weighted version of the given node assuming its weight changed, or the node
        itself otherwise.
        """
        new_weight = self._calculate_weight_for(concept, node)
        return evolve(node, weight=new_weight) if node.weight != new_weight else node  # type: ignore

    def _calculate_weight_for(self, concept: Concept, node: NodePredicate) -> float:
        if isinstance(
            node,
            (
                StrokeGNNRecognitionPredicate,
                IsOntologyNodePredicate,
                CategoricalPredicate,
            ),
        ):
            present_count = 1
            in_difference_count = 1

            # If it involves an ontology node, count that way
            if isinstance(node, StrokeGNNRecognitionPredicate):
                present_count = self._stroke_gnn_recognition_nodes_present.get(
                    (concept, node.recognized_object), 1
                )
                in_difference_count = (
                    self._stroke_gnn_recognition_nodes_present_in_difference.get(
                        (concept, node.recognized_object), 1
                    )
                )
            # If it involves an ontology node, count that way
            elif isinstance(node, IsOntologyNodePredicate):
                present_count = self._ontology_node_present.get(
                    (concept, node.property_value), 1
                )
                in_difference_count = self._ontology_node_present_in_difference.get(
                    (concept, node.property_value), 1
                )
            # Otherwise, if it's categorical, count the value observed
            elif isinstance(node, CategoricalPredicate):
                present_count = self._categorical_values_present.get(
                    (concept, node.value), 1
                )
                in_difference_count = self._categorical_values_present_in_difference.get(
                    (concept, node.value), 1
                )

            return 0.5 + in_difference_count / present_count
        else:
            return node.weight

    def _get_apprentice_pattern(self, concept: Concept) -> PerceptionGraphPattern:
        concepts_to_patterns = self.apprentice.concepts_to_patterns()
        return concepts_to_patterns[concept]


NodeType = TypeVar("NodeType", bound=SemanticNode)  # pylint:disable=invalid-name


def _get_relevant_concepts_from_learner(
    contrastive_pair: LanguagePerceptionSemanticContrast,
    *,
    learner: ApprenticeTemplateLearner,
) -> Tuple[Concept, Concept]:
    return (
        _get_relevant_concept_from_learner(
            contrastive_pair.first_alignment.language_concept_alignment.language,
            learner=learner,
        ),
        _get_relevant_concept_from_learner(
            contrastive_pair.second_alignment.language_concept_alignment.language,
            learner=learner,
        ),
    )


def _get_relevant_concept_from_learner(
    linguistic_description: LinguisticDescriptionT, *, learner: ApprenticeTemplateLearner
) -> Concept:
    return first(
        _get_matching_concepts(linguistic_description, _concepts_to_templates(learner))
    )


def _concepts_to_templates(
    learner: ApprenticeTemplateLearner,
) -> Mapping[Concept, AbstractSet[SurfaceTemplate]]:
    return {
        concept: learner.templates_for_concept(concept)
        for concept, _ in learner.concept_to_surface_template.items()
    }


def _get_matching_concepts(
    linguistic_description: LinguisticDescriptionT,
    concept_to_templates: Mapping[Concept, AbstractSet[SurfaceTemplate]],
) -> Sequence[Concept]:
    """
    Get the concepts whose templates match the linguistic description, ordered by match length.

    We use the maximum match length here.
    """
    concepts_with_max_match_lengths = []
    for concept, templates in concept_to_templates.items():
        matches = []
        for template in templates:
            maybe_span = template.match_against_tokens(
                linguistic_description.as_token_sequence(), slots_to_filler_spans={}
            )
            if maybe_span:
                matches.append(maybe_span)

        if matches:
            concepts_with_max_match_lengths.append(
                (concept, max(len(match) for match in matches))
            )
    concepts_with_max_match_lengths = sorted(
        concepts_with_max_match_lengths, key=lambda pair: pair[1]
    )
    return tuple(
        concept for concept, _max_match_length in concepts_with_max_match_lengths
    )


def _get_difference_nodes(
    matching: LanguagePerceptionSemanticContrast, *, ontology: Ontology
) -> Tuple[ImmutableSet[PerceptionGraphNode], ImmutableSet[PerceptionGraphNode]]:
    """
    Get the set of perception nodes in graph 1 but not graph 2, and the set in graph 2 but not 1.
    """
    graph1_as_pattern, perception_graph_match = _match_graphs_to_each_other(
        matching, ontology=ontology
    )
    if perception_graph_match is None:
        return (
            immutableset(
                matching.first_alignment.perception_semantic_alignment.perception_graph
            ),
            immutableset(
                matching.second_alignment.perception_semantic_alignment.perception_graph
            ),
        )

    return (
        immutableset(
            [
                graph1_node
                for graph1_node in matching.first_alignment.perception_semantic_alignment.perception_graph
                if (
                    graph1_as_pattern.perception_graph_node_to_pattern_node[graph1_node]
                    not in perception_graph_match.matched_pattern
                )
            ]
        ),
        immutableset(
            [
                graph2_node
                for graph2_node in matching.second_alignment.perception_semantic_alignment.perception_graph
                if graph2_node not in perception_graph_match.matched_sub_graph
            ]
        ),
    )


def _match_graphs_to_each_other(
    matching: LanguagePerceptionSemanticContrast, *, ontology: Ontology
) -> Tuple[PerceptionGraphPatternFromGraph, Optional[PerceptionGraphPatternMatch]]:
    graph1_as_pattern = PerceptionGraphPattern.from_graph(
        matching.first_alignment.perception_semantic_alignment.perception_graph,
        # The min match score doesn't matter here. We're creating patterns from perception graphs.
        # Any continuous nodes in those graphs must necessarily have only one observation. That
        # means they will fall back to fixed-tolerance interval matching, which ignores the match
        # scoring threshold entirely.
        min_continuous_feature_match_score=0.05,
    )
    perception_graph_match = graph1_as_pattern.perception_graph_pattern.matcher(
        matching.second_alignment.perception_semantic_alignment.perception_graph,
        match_mode=MatchMode.OBJECT,
    ).relax_pattern_until_it_matches_getting_match(
        ontology=ontology, trim_after_match=None
    )
    return graph1_as_pattern, perception_graph_match


CountedType = TypeVar("CountedType")  # pylint:disable=invalid-name


def _make_counter_json_ready(counter: Counter[CountedType]) -> Mapping[str, Any]:
    result: MutableMapping[str, Any] = {}
    for key, count in counter.items():
        # If the key is iterable, represent that in the output as a nesting of dicts count
        if isinstance(key, Sequence) and not isinstance(key, str):
            ptr = result
            for part in key[:-1]:
                ptr = ptr.setdefault(str(part), {})
            ptr[str(key[-1])] = count
        # If it's not iterable
        else:
            result[str(key)] = count
    return result
