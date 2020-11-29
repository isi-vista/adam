from abc import ABC

import logging

# from itertools import chain
from pathlib import Path
from typing import AbstractSet, Union, Optional, Tuple, Iterable, Sequence, Mapping

# from more_itertools import powerset
from adam.language import LinguisticDescription
from adam.language_specific.english import DETERMINERS
from adam.learner import LearningExample, get_largest_matching_pattern
from adam.learner.language_mode import LanguageMode
from adam.learner.alignments import (
    LanguagePerceptionSemanticAlignment,
    PerceptionSemanticAlignment,
)
from adam.learner.learner_utils import (
    assert_static_situation,
    pattern_remove_incomplete_region_or_spatial_path,
    covers_entire_utterance,
)
from adam.learner.object_recognizer import (
    ObjectRecognizer,
    PerceptionGraphFromObjectRecognizer,
)
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.pursuit import AbstractPursuitLearnerNew
from adam.learner.subset import (
    AbstractTemplateSubsetLearner,
    AbstractTemplateSubsetLearnerNew,
)
from adam.learner.surface_templates import (
    SLOT1,
    STANDARD_SLOT_VARIABLES,
    SurfaceTemplate,
    SurfaceTemplateBoundToSemanticNodes,
)
from adam.learner.template_learner import (
    AbstractTemplateLearner,
    AbstractTemplateLearnerNew,
)
from adam.perception import PerceptualRepresentation, MatchMode
from adam.perception.deprecated import LanguageAlignedPerception
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception.perception_graph import PerceptionGraph, HAS_PROPERTY_LABEL
from adam.semantics import AttributeConcept, ObjectSemanticNode, SemanticNode
from attr import attrib, attrs
from attr.validators import instance_of, optional
from immutablecollections import immutabledict, immutableset, immutablesetmultidict
from vistautils.span import Span
from adam.learner.learner_utils import SyntaxSemanticsVariable


@attrs
class AbstractAttributeTemplateLearnerNew(AbstractTemplateLearnerNew, ABC):
    # pylint:disable=abstract-method
    def _candidate_templates(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> AbstractSet[SurfaceTemplateBoundToSemanticNodes]:
        ret = []
        language_concept_alignment = (
            language_perception_semantic_alignment.language_concept_alignment
        )
        # Find all objects we have recognized...
        for (
            object_node,
            span_for_object,
        ) in language_concept_alignment.node_to_language_span.items():
            if isinstance(object_node, ObjectSemanticNode):
                try:
                    # Any words immediately before them or after them are candidate attributes.
                    # See https://github.com/isi-vista/adam/issues/791 .
                    preceding_token_index = span_for_object.start - 1
                    if (
                        preceding_token_index >= 0
                        and not language_concept_alignment.token_index_is_aligned(
                            preceding_token_index
                        )
                    ):
                        ret.append(
                            SurfaceTemplateBoundToSemanticNodes(
                                language_concept_alignment.to_surface_template(
                                    {object_node: SLOT1},
                                    restrict_to_span=Span(
                                        preceding_token_index, span_for_object.end
                                    ),
                                    language_mode=self._language_mode,
                                ),
                                {SLOT1: object_node},
                            )
                        )
                    following_token_index = span_for_object.end + 1
                    if following_token_index < len(
                        language_concept_alignment.language.as_token_sequence()
                    ) and not language_concept_alignment.token_index_is_aligned(
                        following_token_index
                    ):
                        ret.append(
                            SurfaceTemplateBoundToSemanticNodes(
                                language_concept_alignment.to_surface_template(
                                    {object_node: SLOT1},
                                    restrict_to_span=Span(
                                        span_for_object.start, following_token_index
                                    ),
                                    language_mode=self._language_mode,
                                ),
                                {SLOT1: object_node},
                            )
                        )
                # Catches errors in to_surface_template() - we skip this case to prevent the learning from breaking.
                except RuntimeError:
                    continue

        return immutableset(
            bound_surface_template
            for bound_surface_template in ret
            # For now, we require templates to account for the entire utterance.
            # See https://github.com/isi-vista/adam/issues/789
            if covers_entire_utterance(
                bound_surface_template,
                language_concept_alignment,
                # We need to explicitly ignore determiners here for some reason
                # See: https://github.com/isi-vista/adam/issues/871
                ignore_determiners=True,
            )
            # this keeps the relation learner from learning things such as "a_slot1" which will pose an issue for
            # later learning of attributes since the learner may consider both the attribute and the object to be objects initially,
            # leading it to try to match two objects with a template that only has one slot
            and not all(
                (e in DETERMINERS or isinstance(e, SyntaxSemanticsVariable))
                for e in bound_surface_template.surface_template.elements
            )
            #separate set for English and all determiners
        )

    def _enrich_post_process(
        self,
        perception_graph_after_matching: PerceptionGraph,
        immutable_new_nodes: AbstractSet[SemanticNode],
    ) -> Tuple[PerceptionGraph, AbstractSet[SemanticNode]]:
        return perception_graph_after_matching, immutable_new_nodes


@attrs
class AbstractAttributeTemplateLearner(AbstractTemplateLearner, ABC):
    # mypy doesn't realize that fields without defaults can come after those with defaults
    # if they are keyword-only.
    _object_recognizer: ObjectRecognizer = attrib(  # type: ignore
        validator=instance_of(ObjectRecognizer), kw_only=True
    )

    def _assert_valid_input(
        self,
        to_check: Union[
            PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame],
            LearningExample[DevelopmentalPrimitivePerceptionFrame, LinguisticDescription],
        ],
    ) -> None:
        assert_static_situation(to_check)

    def _extract_perception_graph(
        self, perception: PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame]
    ) -> PerceptionGraph:
        return PerceptionGraph.from_frame(perception.frames[0])

    def _preprocess_scene_for_learning(
        self, language_concept_alignment: LanguageAlignedPerception
    ) -> LanguageAlignedPerception:
        post_recognition_object_perception_alignment = self._object_recognizer.match_objects_with_language_old(
            language_concept_alignment
        )
        return post_recognition_object_perception_alignment

    def _preprocess_scene_for_description(
        self, perception_graph: PerceptionGraph
    ) -> PerceptionGraphFromObjectRecognizer:
        return self._object_recognizer.match_objects_old(perception_graph)

    def _extract_surface_template(
        self,
        language_concept_alignment: LanguageAlignedPerception,
        language_mode: LanguageMode = LanguageMode.ENGLISH,
    ) -> SurfaceTemplate:
        if len(language_concept_alignment.aligned_nodes) > 1:
            raise RuntimeError("Input has too many aligned nodes for us to handle.")

        return language_concept_alignment.to_surface_template(
            object_node_to_template_variable=immutabledict(
                zip(language_concept_alignment.aligned_nodes, STANDARD_SLOT_VARIABLES)
            ),
            # This is a hack to handle determiners.
            # For attributes at the moment we learn the determiner together with the
            # attribute, which is not ideal.
            determiner_prefix_slots=immutableset(),
        )


@attrs
class SubsetAttributeLearner(
    AbstractTemplateSubsetLearner, AbstractAttributeTemplateLearner
):
    def _hypothesis_from_perception(
        self, preprocessed_input: LanguageAlignedPerception
    ) -> PerceptionGraphTemplate:
        num_nodes_aligned_to_language = len(preprocessed_input.aligned_nodes)
        if num_nodes_aligned_to_language != 1:
            raise RuntimeError(
                f"Attribute learner can work only with a single aligned node,"
                f"but got {num_nodes_aligned_to_language}. Language is "
                f"{preprocessed_input.language.as_token_string()}"
            )

        return PerceptionGraphTemplate.from_graph(
            preprocessed_input.perception_graph,
            template_variable_to_matched_object_node=immutabledict(
                zip(STANDARD_SLOT_VARIABLES, preprocessed_input.aligned_nodes)
            ),
        )

    def _update_hypothesis(
        self,
        previous_pattern_hypothesis: PerceptionGraphTemplate,
        current_pattern_hypothesis: PerceptionGraphTemplate,
    ) -> Optional[PerceptionGraphTemplate]:
        return previous_pattern_hypothesis.intersection(
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


@attrs
class SubsetAttributeLearnerNew(
    AbstractTemplateSubsetLearnerNew, AbstractAttributeTemplateLearnerNew
):
    def _can_learn_from(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> bool:
        return (
            len(
                language_perception_semantic_alignment.perception_semantic_alignment.semantic_nodes
            )
            > 1
        )

    def _preprocess_scene(
        self, perception_semantic_alignment: PerceptionSemanticAlignment
    ) -> PerceptionSemanticAlignment:
        return perception_semantic_alignment

    def _new_concept(self, debug_string: str) -> AttributeConcept:
        return AttributeConcept(debug_string)

    def _hypotheses_from_perception(
        self,
        learning_state: LanguagePerceptionSemanticAlignment,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,
    ) -> AbstractSet[PerceptionGraphTemplate]:
        # This makes a hypothesis for the whole graph, with the wildcard slot
        # at each recognized object.
        return immutableset(
            [
                PerceptionGraphTemplate.from_graph(
                    learning_state.perception_semantic_alignment.perception_graph,
                    template_variable_to_matched_object_node=bound_surface_template.slot_to_semantic_node,
                )
            ]
        )

    def _keep_hypothesis(
        self,
        *,
        hypothesis: PerceptionGraphTemplate,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,  # pylint:disable=unused-argument
    ) -> bool:
        #TODO: update this for classifier experiments 
        if len(hypothesis.graph_pattern) < 2:
            # We need at least two nodes - a wildcard and a property -
            # for meaningful attribute semantics.
            return False
        return True

    def _update_hypothesis(
        self,
        previous_pattern_hypothesis: PerceptionGraphTemplate,
        current_pattern_hypothesis: PerceptionGraphTemplate,
    ) -> Optional[PerceptionGraphTemplate]:
        return previous_pattern_hypothesis.intersection(
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
            trim_after_match=pattern_remove_incomplete_region_or_spatial_path,
        )


@attrs
class PursuitAttributeLearnerNew(
    AbstractPursuitLearnerNew, AbstractAttributeTemplateLearnerNew
):
    def _find_identical_hypothesis(
        self,
        new_hypothesis: PerceptionGraphTemplate,
        candidates: Iterable[PerceptionGraphTemplate],
    ) -> Optional[PerceptionGraphTemplate]:
        for candidate in candidates:
            if new_hypothesis.graph_pattern.check_isomorphism(candidate.graph_pattern):
                return candidate
        return None

    @attrs(frozen=True)
    class AttributeHypothesisPartialMatch(AbstractPursuitLearnerNew.PartialMatch):
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
        self,
        hypothesis: PerceptionGraphTemplate,
        graph: PerceptionGraph,
        *,
        required_alignments: Mapping[SyntaxSemanticsVariable, ObjectSemanticNode],
    ) -> "AbstractPursuitLearnerNew.PartialMatch":
        pattern = hypothesis.graph_pattern
        hypothesis_pattern_common_subgraph = get_largest_matching_pattern(
            pattern,
            graph,
            debug_callback=self._debug_callback,
            graph_logger=self._hypothesis_logger,
            ontology=self._ontology,
            match_mode=MatchMode.NON_OBJECT,
            allowed_matches=immutablesetmultidict(
                [
                    (hypothesis.template_variable_to_pattern_node[variable], object_node)
                    for variable, object_node in required_alignments.items()
                ]
            ),
        )
        self.debug_counter += 1

        leading_hypothesis_num_nodes = len(pattern)
        num_nodes_matched = (
            len(hypothesis_pattern_common_subgraph.copy_as_digraph().nodes)
            if hypothesis_pattern_common_subgraph
            else 0
        )
        if hypothesis_pattern_common_subgraph:
            partial_hypothesis: Optional[
                PerceptionGraphTemplate
            ] = PerceptionGraphTemplate(
                graph_pattern=hypothesis_pattern_common_subgraph,
                template_variable_to_pattern_node=hypothesis.template_variable_to_pattern_node,
            )
        else:
            partial_hypothesis = None

        return PursuitAttributeLearnerNew.AttributeHypothesisPartialMatch(
            partial_hypothesis,
            num_nodes_matched=num_nodes_matched,
            num_nodes_in_pattern=leading_hypothesis_num_nodes,
        )

    def _new_concept(self, debug_string: str) -> AttributeConcept:
        return AttributeConcept(debug_string)

    def _keep_hypothesis(
        self,
        *,
        hypothesis: PerceptionGraphTemplate,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,  # pylint:disable=unused-argument
    ) -> bool:
        if len(hypothesis.graph_pattern) < 2:
            # We need at least two nodes - a wildcard and a property -
            # for meaningful attribute semantics.
            return False
        return True

    def _hypotheses_from_perception(
        self,
        learning_state: LanguagePerceptionSemanticAlignment,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,
    ) -> AbstractSet[PerceptionGraphTemplate]:
        return immutableset(
            PerceptionGraphTemplate.from_graph(
                perception_graph=candidate_attribute_meaning,
                template_variable_to_matched_object_node=bound_surface_template.slot_to_semantic_node,
            )
            for candidate_attribute_meaning in _extract_candidate_attributes(
                learning_state.perception_semantic_alignment.perception_graph,
                bound_surface_template.slot_to_semantic_node[SLOT1],
            )
        )

    def _can_learn_from(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> bool:
        # We can only learn attributes if there are aligned object semantic nodes,
        # and for there to be any aligned semantic nodes at all
        # there must be aligned object semantic nodes,
        # since relations, verbs, etc. build on object semantic nodes.
        # Furthermore we only need one node, because attributes only involve one object.
        return (
            len(
                language_perception_semantic_alignment.perception_semantic_alignment.semantic_nodes
            )
            > 1
        )

    def _preprocess_scene(
        self, perception_semantic_alignment: PerceptionSemanticAlignment
    ) -> PerceptionSemanticAlignment:
        return perception_semantic_alignment

    def log_hypotheses(self, log_output_path: Path) -> None:
        logging.info(
            "Logging %s hypotheses to %s",
            len(self._concept_to_hypotheses_and_scores),
            log_output_path,
        )
        for (
            concept,
            hypotheses_to_scores,
        ) in self._concept_to_hypotheses_and_scores.items():
            for (i, (hypothesis, score)) in enumerate(hypotheses_to_scores.items()):
                hypothesis.render_to_file(
                    concept.debug_string,
                    log_output_path / f"{concept.debug_string}.{i}.score_{score}",
                )

        logging.info(
            "Logging %s lexicon items to %s", len(self._lexicon), log_output_path
        )
        for (
            surface_template,
            meaning_as_perception_graph_template,
        ) in self._lexicon.items():
            meaning_as_perception_graph_template.render_to_file(
                surface_template.to_short_string(),
                log_output_path / f"lexicon-{surface_template.to_short_string()}",
            )


def _extract_candidate_attributes(
    whole_scene_perception_graph: PerceptionGraph,
    object_with_attribute: ObjectSemanticNode,
) -> Sequence[PerceptionGraph]:
    perception_digraph = whole_scene_perception_graph.copy_as_digraph()
    # For now, we assume all attributes are based on properties.
    properties = immutableset(
        [
            node
            for _, node, label in perception_digraph.out_edges(
                object_with_attribute, data="label"
            )
            if label == HAS_PROPERTY_LABEL
        ]
    )
    # Furthermore, we limit the search space to the even smaller set of hypotheses
    # where we consider only single properties as possible attributes.
    # Otherwise there are too many hypotheses for the pursuit learner to search through
    # and it's unlikely to converge on the correct hypothesis
    # in any reasonable amount of time or number of examples.
    candidate_attribute_subgraph_node_sets = [
        immutableset([object_with_attribute, property]) for property in properties
    ]
    return immutableset(
        [
            whole_scene_perception_graph.subgraph_by_nodes(
                candidate_attribute_subgraph_nodes
            )
            for candidate_attribute_subgraph_nodes in candidate_attribute_subgraph_node_sets
        ]
    )
