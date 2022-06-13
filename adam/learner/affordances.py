import logging
from abc import ABC
from pathlib import Path
from typing import (
    AbstractSet,
    Tuple,
    Optional,
    Iterable,
    Dict,
    MutableMapping,
    Sequence,
    MutableSet,
)

import yaml
from attr import attrs, attrib
from attr.validators import deep_iterable, instance_of, deep_mapping
from immutablecollections import immutablesetmultidict, immutableset

from adam.learner import (
    LanguagePerceptionSemanticAlignment,
    PerceptionSemanticAlignment,
    SurfaceTemplateBoundToSemanticNodes,
)
from adam.learner.learner_utils import add_node_connected_to_perception_graph
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.subset import AbstractTemplateSubsetLearner
from adam.learner.surface_templates import SLOT1, SurfaceTemplate
from adam.learner.template_learner import AbstractTemplateLearner
from adam.perception import MatchMode
from adam.perception.perception_graph import (
    PerceptionGraph,
    HAS_PROPERTY_LABEL,
    TemporalScope,
    LABEL,
    edge_equals_ignoring_temporal_scope,
    HAS_STROKE_LABEL,
    PerceptionGraphPattern,
    PerceptionGraphPatternMatch,
)
from adam.perception.perception_graph_nodes import CategoricalNode
from adam.perception.perception_graph_predicates import (
    AnyObjectPredicate,
    ObjectSemanticNodePerceptionPredicate,
)
from adam.semantics import (
    SemanticNode,
    Concept,
    AffordanceConcept,
    AffordanceSemanticNode,
    ActionSemanticNode,
)


class AbstractAffordanceTemplateLearner(AbstractTemplateLearner, ABC):
    def _can_learn_from(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> bool:
        return any(
            node
            for node in language_perception_semantic_alignment.perception_semantic_alignment.semantic_nodes
            if isinstance(node, ActionSemanticNode)
        )

    def _preprocess_scene(
        self, perception_semantic_alignment: PerceptionSemanticAlignment
    ) -> PerceptionSemanticAlignment:
        digraph = perception_semantic_alignment.perception_graph.copy_as_digraph()
        filtered_semantic_nodes = []
        for node in perception_semantic_alignment.semantic_nodes:
            if isinstance(node, AffordanceSemanticNode):
                digraph.remove_node(node)
                continue
            filtered_semantic_nodes.append(node)
        return PerceptionSemanticAlignment(
            perception_graph=PerceptionGraph(
                digraph, dynamic=perception_semantic_alignment.perception_graph.dynamic
            ),
            semantic_nodes=filtered_semantic_nodes,
        )

    def _candidate_templates(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> AbstractSet[SurfaceTemplateBoundToSemanticNodes]:
        candidates = []
        semantic_nodes = (
            language_perception_semantic_alignment.perception_semantic_alignment.semantic_nodes
        )
        for action_node in semantic_nodes:
            if isinstance(action_node, ActionSemanticNode):
                for slot, object_node in action_node.slot_fillings.items():
                    candidates.append(
                        # We acknowledge this is an unusual form for this alignment as normally this would align to
                        # some input string. Given affordances are unnamed however we generate a surface template
                        # via this hack rather than a large refactoring effort needed to learn unnamed concepts
                        # We also don't include the `SLOT1` enum value as we don't actually ever want to
                        # Include the name of the object we're adding this affordance to instead we just want the
                        # name of the slot this object was in.
                        SurfaceTemplateBoundToSemanticNodes(
                            surface_template=SurfaceTemplate(
                                elements=(slot.name, action_node.concept.debug_string),
                                language_mode=self._language_mode,
                            ),
                            slot_to_semantic_node={SLOT1: object_node},
                        )
                    )

        return immutableset(candidates)

    def _enrich_post_process(
        self,
        perception_graph_after_matching: PerceptionGraph,
        immutable_new_nodes: AbstractSet[SemanticNode],
    ) -> Tuple[PerceptionGraph, AbstractSet[SemanticNode]]:

        digraph = perception_graph_after_matching.copy_as_digraph()
        for node in immutable_new_nodes:
            if (
                isinstance(node, AffordanceSemanticNode)
                and len(node.slot_fillings.values()) == 1
            ):
                # Ideally I'd like this weight to be the graph match percentage, but we don't have that information
                # Available to us here and refactoring to have that information is out of scope
                affordance_property = CategoricalNode(
                    label="affordance", value=node.concept.debug_string, weight=1.0
                )
                perception_graph_after_matching = add_node_connected_to_perception_graph(
                    perception_graph_after_matching,
                    list(node.slot_fillings.values())[0],
                    affordance_property,
                    temporal_scope=TemporalScope.BEFORE
                    if perception_graph_after_matching.dynamic
                    else None,
                )

        return (
            PerceptionGraph(
                graph=digraph, dynamic=perception_graph_after_matching.dynamic
            ),
            immutable_new_nodes,
        )


@attrs(slots=True)
class SubsetAffordanceLearner(
    AbstractAffordanceTemplateLearner, AbstractTemplateSubsetLearner
):
    """
    An implementation of `TopLevelLanguageLearner` for subset learning based approach for affordance learning.
    """

    def _update_hypothesis(
        self,
        previous_pattern_hypothesis: PerceptionGraphTemplate,
        current_pattern_hypothesis: PerceptionGraphTemplate,
    ) -> Optional[PerceptionGraphTemplate]:
        match = previous_pattern_hypothesis.intersection_getting_match(
            current_pattern_hypothesis,
            ontology=self._ontology,
            match_mode=MatchMode.NON_OBJECT,
            allowed_matches=immutablesetmultidict(
                [
                    (node2, node1)
                    for previous_slot, node1 in previous_pattern_hypothesis.template_variable_to_pattern_node.items()
                    for new_slot, node2 in current_pattern_hypothesis.template_variable_to_pattern_node.items()
                    if previous_slot == new_slot
                ]
            ),
        )
        if match:
            match.confirm_match()
            return match.intersection
        return None

    def _new_concept(self, debug_string: str) -> Concept:
        return AffordanceConcept(debug_string)

    def _keep_hypothesis(
        self,
        *,
        hypothesis: PerceptionGraphTemplate,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,
    ) -> bool:
        if len(bound_surface_template.slot_to_semantic_node) > 1:
            return False
        return len(hypothesis.graph_pattern) >= 2

    def _hypotheses_from_perception(
        self,
        learning_state: LanguagePerceptionSemanticAlignment,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,
    ) -> AbstractSet[PerceptionGraphTemplate]:
        if len(bound_surface_template.slot_to_semantic_node) > 1:
            raise ValueError(
                f"Affordance learner can not extract features when more than one object node is a "
                f"slot filler. Offending template: {bound_surface_template}"
            )
        root_object_node = bound_surface_template.slot_to_semantic_node[SLOT1]
        candidate_affordance_subgraph_nodes = [root_object_node]
        candidate_affordance_subgraph_nodes.extend(
            node
            for _, node, label in learning_state.perception_semantic_alignment.perception_graph._graph.out_edges(  # pylint: disable=protected-access
                root_object_node, data=LABEL
            )
            if edge_equals_ignoring_temporal_scope(
                label, {HAS_PROPERTY_LABEL, HAS_STROKE_LABEL}
            )
        )
        return immutableset(
            [
                PerceptionGraphTemplate.from_graph(
                    perception_graph=learning_state.perception_semantic_alignment.perception_graph.subgraph_by_nodes(
                        immutableset(candidate_affordance_subgraph_nodes)
                    ),
                    template_variable_to_matched_object_node=bound_surface_template.slot_to_semantic_node,
                    min_continuous_feature_match_score=self._min_continuous_feature_match_score,
                )
            ]
        )

    def _primary_templates(
        self,
    ) -> Iterable[Tuple[Concept, PerceptionGraphTemplate, float]]:
        for concept, hypotheses in self._concept_to_hypotheses.items():
            if len(hypotheses) > 1:
                continue
            hypothesis = hypotheses[0]
            yield concept, hypothesis.copy_replacing_nodes(
                {
                    node: AnyObjectPredicate()
                    for node in hypothesis.graph_pattern
                    if isinstance(node, ObjectSemanticNodePerceptionPredicate)
                }
            ), 1.0

    def _fallback_templates(
        self,
    ) -> Iterable[Tuple[Concept, PerceptionGraphTemplate, float]]:
        # Alternate hypotheses stored in the beam.
        for concept, hypotheses in self._concept_to_hypotheses.items():
            if len(hypotheses) == 1:
                continue
            for hypothesis in hypotheses[1:]:
                yield concept, hypothesis.copy_replacing_nodes(
                    {
                        node: AnyObjectPredicate()
                        for node in hypothesis.graph_pattern
                        if isinstance(node, ObjectSemanticNodePerceptionPredicate)
                    }
                ), 1.0


@attrs(slots=True)
class MappingAffordanceLearner(AbstractAffordanceTemplateLearner):

    concept_to_affordances: MutableMapping[Concept, MutableSet[Concept]] = attrib(
        validator=deep_mapping(instance_of(Concept), deep_iterable(instance_of(Concept))),
        factory=dict,
    )
    affordance_debug_string_to_concept: MutableMapping[str, Concept] = attrib(
        validator=deep_mapping(instance_of(str), instance_of(Concept)), factory=dict
    )

    def _match_template(
        self,
        *,
        concept: Concept,
        pattern: PerceptionGraphTemplate,
        perception_graph: PerceptionGraph,
        confidence: float,
    ) -> Iterable[Tuple[PerceptionGraphPatternMatch, SemanticNode]]:
        return immutableset()

    def _new_concept(self, debug_string: str) -> Concept:
        return AffordanceConcept(debug_string)

    def _learning_step(
        self,
        language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,
    ) -> None:
        # This loop should only execute once in standard cases
        # As a normal affordance only has one slot filler,
        # the object the affordance affects.
        for (
            _,
            object_semantic_node,
        ) in bound_surface_template.slot_to_semantic_node.items():
            debug_name = bound_surface_template.surface_template.to_short_string()
            if debug_name in self.affordance_debug_string_to_concept:
                concept = self.affordance_debug_string_to_concept[debug_name]
            else:
                concept = self._new_concept(debug_name)
                self.affordance_debug_string_to_concept[debug_name] = concept

            self.concept_to_affordances.setdefault(
                object_semantic_node.concept, set()
            ).add(concept)

    def _primary_templates(
        self,
    ) -> Iterable[Tuple[Concept, PerceptionGraphTemplate, float]]:
        return immutableset()

    def _fallback_templates(
        self,
    ) -> Iterable[Tuple[Concept, PerceptionGraphTemplate, float]]:
        return immutableset()

    def templates_for_concept(self, concept: Concept) -> AbstractSet[SurfaceTemplate]:
        return immutableset()

    def process_affordance(
        self, concept: Concept, affordance_node: AffordanceSemanticNode
    ) -> None:
        logging.debug(
            "MappingAffordanceLearner doesn't process incoming affordance nodes."
        )

    def log_hypotheses(self, log_output_path: Path) -> None:
        with open(
            log_output_path / "mapping_affordance.yaml", "w", encoding="utf-8"
        ) as log:
            yaml.dump(
                {
                    f"ObjectConcept({concept.debug_string})": [
                        f"{affordance.debug_string}"
                        for affordance in sorted(
                            affordances, key=lambda x: x.debug_string
                        )
                    ]
                    for concept, affordances in self.concept_to_affordances.items()
                },
                log,
            )

    def concepts_to_patterns(self) -> Dict[Concept, PerceptionGraphPattern]:
        return {}

    def affordances_of_concept(self, concept: Concept) -> Sequence[Concept]:
        return sorted(
            self.concept_to_affordances.get(concept, []),
            key=lambda x: x.debug_string,
        )
