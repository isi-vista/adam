import logging
from itertools import chain
from pathlib import Path

from attr import attrib, attrs
from attr.validators import instance_of, optional
from immutablecollections import (
    ImmutableSet,
    ImmutableSetMultiDict,
    immutabledict,
    immutableset,
    immutablesetmultidict,
)
from vistautils.parameters import Parameters

from typing import (
    AbstractSet,
    Iterable,
    Optional,
    Tuple,
    Dict,
    Mapping,
)

from more_itertools import first

from adam.language_specific.chinese.chinese_phase_1_lexicon import (
    GAILA_PHASE_1_CHINESE_LEXICON,
)
from adam.language_specific.english import DETERMINERS
from adam.learner import get_largest_matching_pattern, graph_without_learner
from adam.learner.alignments import (
    LanguagePerceptionSemanticAlignment,
    PerceptionSemanticAlignment,
)
from adam.learner.cross_situational_learner import AbstractCrossSituationalLearner
from adam.learner.language_mode import LanguageMode
from adam.learner.learner_utils import (
    candidate_object_hypotheses,
    covers_entire_utterance,
    get_objects_from_perception,
)
from adam.learner.object_recognizer import (
    ObjectRecognizer,
    extract_candidate_objects,
    replace_match_with_object_graph_node,
)
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.propose_but_verify import AbstractProposeButVerifyLearner
from adam.learner.pursuit import AbstractPursuitLearner
from adam.learner.subset import AbstractTemplateSubsetLearner
from adam.learner.surface_templates import (
    SurfaceTemplate,
    SurfaceTemplateBoundToSemanticNodes,
)
from adam.learner.template_learner import AbstractTemplateLearner, TemplateLearner
from adam.ontology.ontology import Ontology
from adam.ontology.phase1_ontology import GAILA_PHASE_1_ONTOLOGY
from adam.perception import ObjectPerception, MatchMode
from adam.perception.perception_graph import (
    PerceptionGraph,
    PerceptionGraphPattern,
    PerceptionGraphPatternMatch,
    GraphLogger,
)
from adam.random_utils import RandomChooser
from adam.semantics import (
    Concept,
    ObjectConcept,
    GROUND_OBJECT_CONCEPT,
    SemanticNode,
    ObjectSemanticNode,
    FunctionalObjectConcept,
    SyntaxSemanticsVariable,
)
from adam.utils.networkx_utils import subgraph


class AbstractObjectTemplateLearner(AbstractTemplateLearner):
    # pylint:disable=abstract-method
    def _can_learn_from(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> bool:
        # We can try to learn objects from anything, as long as the scene isn't already
        # completely understood.
        return (
            not language_perception_semantic_alignment.language_concept_alignment.is_entirely_aligned
        )

    def _preprocess_scene(
        self, perception_semantic_alignment: PerceptionSemanticAlignment
    ) -> PerceptionSemanticAlignment:
        # Avoid accidentally identifying a word with the learner itself.
        return perception_semantic_alignment.copy_with_updated_graph_and_added_nodes(
            new_graph=graph_without_learner(
                perception_semantic_alignment.perception_graph
            ),
            new_nodes=[],
        )

    def _candidate_templates(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> AbstractSet[SurfaceTemplateBoundToSemanticNodes]:
        # We can only learn single words for objects at the moment.
        # See https://github.com/isi-vista/adam/issues/793 .

        # Attempt to align every unaligned token to some object in the scene.
        language_alignment = (
            language_perception_semantic_alignment.language_concept_alignment
        )
        ret = immutableset(
            SurfaceTemplateBoundToSemanticNodes(
                SurfaceTemplate.for_object_name(token, language_mode=self._language_mode),
                slot_to_semantic_node={},
            )
            for (tok_idx, token) in enumerate(
                language_alignment.language.as_token_sequence()
            )
            if not language_alignment.token_index_is_aligned(tok_idx)
            # ignore determiners
            and token not in DETERMINERS
        )

        return immutableset(
            bound_surface_template
            for bound_surface_template in ret
            # For now, we require templates to account for the entire utterance.
            # See https://github.com/isi-vista/adam/issues/789
            if covers_entire_utterance(
                bound_surface_template, language_alignment, ignore_determiners=True
            )
        )

    def _enrich_post_process(
        self,
        perception_graph_after_matching: PerceptionGraph,
        immutable_new_nodes: AbstractSet[SemanticNode],
    ) -> Tuple[PerceptionGraph, AbstractSet[SemanticNode]]:
        object_root_nodes = immutableset(  # pylint:disable=protected-access
            node
            for node in perception_graph_after_matching._graph.nodes  # pylint:disable=protected-access
            if isinstance(node, ObjectPerception)
        )
        new_nodes = []
        perception_graph_after_processing = perception_graph_after_matching
        for object_root_node in object_root_nodes:
            fake_subgraph = subgraph(  # pylint:disable=protected-access
                perception_graph_after_matching._graph,  # pylint:disable=protected-access
                [object_root_node],
            )
            fake_perception_graph = PerceptionGraph(
                graph=fake_subgraph, dynamic=perception_graph_after_matching.dynamic
            )
            fake_pattern_graph = PerceptionGraphPattern.from_graph(fake_perception_graph)
            fake_object_semantic_node = ObjectSemanticNode(
                concept=FunctionalObjectConcept("unknown_object"), confidence=1.0
            )
            # perception_graph_after_processing = replace_match_root_with_object_semantic_node(
            #     object_semantic_node=fake_object_semantic_node,
            perception_graph_after_processing = replace_match_with_object_graph_node(
                matched_object_node=fake_object_semantic_node,
                current_perception=perception_graph_after_processing,
                pattern_match=PerceptionGraphPatternMatch(
                    matched_pattern=fake_pattern_graph.perception_graph_pattern,
                    graph_matched_against=perception_graph_after_matching,
                    matched_sub_graph=fake_perception_graph,
                    pattern_node_to_matched_graph_node=fake_pattern_graph.perception_graph_node_to_pattern_node,
                ),
            ).perception_graph_after_replacement
            new_nodes.append(fake_object_semantic_node)

        return (
            perception_graph_after_processing,
            immutableset(chain(immutable_new_nodes, new_nodes)),
        )


@attrs(slots=True)
class SubsetObjectLearner(AbstractObjectTemplateLearner, AbstractTemplateSubsetLearner):
    """
    An implementation of `TopLevelLanguageLearner` for subset learning based approach for single object detection.
    """

    def _new_concept(self, debug_string: str) -> ObjectConcept:
        return ObjectConcept(debug_string)

    def _hypotheses_from_perception(
        self,
        learning_state: LanguagePerceptionSemanticAlignment,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,
    ) -> AbstractSet[PerceptionGraphTemplate]:
        if bound_surface_template.slot_to_semantic_node:
            raise RuntimeError(
                "Object learner should not have slot to semantic node alignments!"
            )

        return immutableset(
            PerceptionGraphTemplate(
                graph_pattern=PerceptionGraphPattern.from_graph(
                    candidate_object
                ).perception_graph_pattern,
                template_variable_to_pattern_node=immutabledict(),
            )
            for candidate_object in extract_candidate_objects(
                learning_state.perception_semantic_alignment.perception_graph,
                sort_by_increasing_size=False,
            )
        )

    # I can't spot the difference in arguments pylint claims?
    def _keep_hypothesis(  # pylint: disable=arguments-differ
        self,
        hypothesis: PerceptionGraphTemplate,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,  # pylint:disable=unused-argument
    ) -> bool:
        if len(hypothesis.graph_pattern) < 3:
            # A two node graph is to small to meaningfully describe an object
            return False
        if all(isinstance(node, ObjectPerception) for node in hypothesis.graph_pattern):
            # A hypothesis which consists of just sub-object structure
            # with no other content is insufficiently distinctive.
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
            match_mode=MatchMode.OBJECT,
            allowed_matches=immutablesetmultidict(
                [
                    (node2, node1)
                    for previous_slot, node1 in previous_pattern_hypothesis.template_variable_to_pattern_node.items()
                    for new_slot, node2 in current_pattern_hypothesis.template_variable_to_pattern_node.items()
                    if previous_slot == new_slot
                ]
            ),
        )

    def _match_template(
        self,
        *,
        concept: Concept,
        pattern: PerceptionGraphTemplate,
        perception_graph: PerceptionGraph,
        confidence: float,
    ) -> Iterable[Tuple[PerceptionGraphPatternMatch, SemanticNode]]:
        # In the case of the object learner,
        # A template only has to match once; we don't care about finding additional matches.
        match = first(
            super()._match_template(
                concept=concept,
                pattern=pattern,
                perception_graph=perception_graph,
                confidence=confidence,
            ),
            None,
        )
        if match is not None:
            yield match


@attrs(slots=True)
class ProposeButVerifyObjectLearner(
    AbstractObjectTemplateLearner, AbstractProposeButVerifyLearner
):
    """
    An implementation of `TopLevelLanguageLearner` for Propose but Verify learning based approach for single object detection.
    """

    @staticmethod
    def from_params(
        params: Parameters,
        *,
        ontology: Optional[Ontology] = None,
        graph_logger: Optional[GraphLogger] = None,
    ) -> "ProposeButVerifyObjectLearner":
        rng = RandomChooser.for_seed(params.optional_integer("random_seed", default=0))

        return ProposeButVerifyObjectLearner(
            graph_match_confirmation_threshold=params.floating_point(
                "graph_match_confirmation_threshold", default=0.8
            ),
            graph_logger=graph_logger,
            rng=rng,
            ontology=ontology if ontology else GAILA_PHASE_1_ONTOLOGY,
            language_mode=params.enum(
                "language_mode", LanguageMode, default=LanguageMode.ENGLISH
            ),
        )

    def log_hypotheses(self, log_output_path: Path) -> None:
        logging.info(
            "Logging %s hypotheses to %s",
            len(self._concept_to_hypotheses),
            log_output_path,
        )
        for (concept, hypotheses) in self._concept_to_hypotheses.items():
            for (i, hypothesis) in enumerate(hypotheses):
                hypothesis.render_to_file(
                    concept.debug_string, log_output_path / f"{concept.debug_string}.{i}"
                )

    def _hypotheses_from_perception(
        self,
        learning_state: LanguagePerceptionSemanticAlignment,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,
    ) -> AbstractSet[PerceptionGraphTemplate]:
        potential_meanings = candidate_object_hypotheses(learning_state)

        # Pick a random meaning for our hypothesis (Independent of any prior data)
        return immutableset([self._rng.choice(potential_meanings)])

    def _new_concept(self, debug_string: str) -> Concept:
        return ObjectConcept(debug_string)

    def _match_template(
        self,
        *,
        concept: Concept,
        pattern: PerceptionGraphTemplate,
        perception_graph: PerceptionGraph,
        confidence: float = 1.0,
    ) -> Iterable[Tuple[PerceptionGraphPatternMatch, SemanticNode]]:
        # In the case of the object learner,
        # A template only has to match once; we don't care about finding additional matches.
        match = first(
            super()._match_template(
                concept=concept,
                pattern=pattern,
                perception_graph=perception_graph,
                confidence=confidence,
            ),
            None,
        )
        if match is not None:
            yield match


@attrs(slots=True)
class CrossSituationalObjectLearner(
    AbstractCrossSituationalLearner, AbstractObjectTemplateLearner
):
    """
    An implementation of `TopLevelLanguageLearner` for Cross Situational learning based approach for single object detection.
    """

    @staticmethod
    def from_params(
        params: Parameters,
        *,
        ontology: Optional[Ontology] = None,
        graph_logger: Optional[GraphLogger] = None,
    ) -> "CrossSituationalObjectLearner":
        return CrossSituationalObjectLearner(
            graph_match_confirmation_threshold=params.floating_point(
                "graph_match_confirmation_threshold"
            ),
            lexicon_entry_threshold=params.floating_point("lexicon_entry_threshold"),
            smoothing_parameter=params.floating_point("smoothing_parameter"),
            expected_number_of_meanings=params.floating_point(
                "expected_number_of_meanings"
            ),
            graph_logger=graph_logger,
            ontology=ontology if ontology else GAILA_PHASE_1_ONTOLOGY,
            language_mode=params.enum(
                "language_mode", LanguageMode, default=LanguageMode.ENGLISH
            ),
        )

    def _new_concept(self, debug_string: str) -> Concept:
        return ObjectConcept(debug_string)

    def _match_template(
        self,
        *,
        concept: Concept,
        pattern: PerceptionGraphTemplate,
        perception_graph: PerceptionGraph,
        confidence: float,
    ) -> Iterable[Tuple[PerceptionGraphPatternMatch, SemanticNode]]:
        # In the case of the object learner,
        # A template only has to match once; we don't care about finding additional matches.
        match_with_semantic_node = first(
            super()._match_template(
                concept=concept,
                pattern=pattern,
                perception_graph=perception_graph,
                confidence=confidence,
            ),
            None,
        )
        if match_with_semantic_node is not None:
            yield match_with_semantic_node

    def _hypotheses_from_perception(
        self,
        learning_state: LanguagePerceptionSemanticAlignment,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,
    ) -> Iterable[PerceptionGraph]:
        return get_objects_from_perception(
            learning_state.perception_semantic_alignment.perception_graph
        )

    def log_hypotheses(self, log_output_path: Path) -> None:
        logging.info(
            "Logging %s hypotheses to %s",
            len(self._concept_to_hypotheses),
            log_output_path,
        )
        for (concept, hypotheses) in self._concept_to_hypotheses.items():
            for (i, hypothesis) in enumerate(hypotheses):
                hypothesis.pattern_template.render_to_file(
                    concept.debug_string, log_output_path / f"{concept.debug_string}.{i}"
                )


@attrs(frozen=True, kw_only=True)
class ObjectRecognizerAsTemplateLearner(TemplateLearner):
    _object_recognizer: ObjectRecognizer = attrib(validator=instance_of(ObjectRecognizer))
    _language_mode: LanguageMode = attrib(
        validator=instance_of(LanguageMode), kw_only=True
    )
    _concepts_to_templates: ImmutableSetMultiDict[Concept, SurfaceTemplate] = attrib(
        init=False
    )

    def learn_from(
        self,
        language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment,
        offset: int = 0,
    ) -> None:
        # The object recognizer doesn't learn anything.
        # It just recognizes predefined object patterns.
        pass

    @staticmethod
    def _enrich_post_process(
        perception_graph_after_matching: PerceptionGraph,
        immutable_new_nodes: AbstractSet[SemanticNode],
    ) -> Tuple[PerceptionGraph, AbstractSet[SemanticNode]]:
        new_nodes = []
        perception_graph_after_processing = perception_graph_after_matching
        for candiate_object_graph in extract_candidate_objects(
            perception_graph_after_matching, sort_by_increasing_size=False
        ):
            fake_pattern_graph = PerceptionGraphPattern.from_graph(candiate_object_graph)
            fake_object_semantic_node = ObjectSemanticNode(
                concept=FunctionalObjectConcept("unknown_object"), confidence=1.0
            )
            perception_graph_after_processing = replace_match_with_object_graph_node(
                matched_object_node=fake_object_semantic_node,
                current_perception=perception_graph_after_processing,
                pattern_match=PerceptionGraphPatternMatch(
                    matched_pattern=fake_pattern_graph.perception_graph_pattern,
                    graph_matched_against=perception_graph_after_processing,
                    matched_sub_graph=candiate_object_graph,
                    pattern_node_to_matched_graph_node=fake_pattern_graph.perception_graph_node_to_pattern_node,
                ),
            ).perception_graph_after_replacement
            new_nodes.append(fake_object_semantic_node)

        return (
            perception_graph_after_processing,
            immutableset(chain(immutable_new_nodes, new_nodes)),
        )

    def enrich_during_learning(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> LanguagePerceptionSemanticAlignment:
        return self._object_recognizer.match_objects_with_language(
            language_perception_semantic_alignment, post_process=self._enrich_post_process
        )

    def enrich_during_description(
        self, perception_semantic_alignment: PerceptionSemanticAlignment
    ) -> PerceptionSemanticAlignment:
        (new_perception_semantic_alignment, _) = self._object_recognizer.match_objects(
            perception_semantic_alignment, post_process=self._enrich_post_process
        )
        return new_perception_semantic_alignment

    def templates_for_concept(self, concept: Concept) -> ImmutableSet[SurfaceTemplate]:
        if self._language_mode == LanguageMode.ENGLISH:
            return self._concepts_to_templates[concept]
        elif self._language_mode == LanguageMode.CHINESE:
            if concept.debug_string == "you":
                return immutableset(
                    [
                        SurfaceTemplate.for_object_name(
                            "ni3", language_mode=self._language_mode
                        )
                    ]
                )
            if concept.debug_string == "me":
                return immutableset(
                    [
                        SurfaceTemplate.for_object_name(
                            "wo3", language_mode=self._language_mode
                        )
                    ]
                )
            mappings = (
                GAILA_PHASE_1_CHINESE_LEXICON._ontology_node_to_word  # pylint:disable=protected-access
            )
            for k, v in mappings.items():
                if k.handle == concept.debug_string:
                    return immutableset(
                        [
                            SurfaceTemplate.for_object_name(
                                v.base_form, language_mode=self._language_mode
                            )
                        ]
                    )
        # FunctionalObjectConcepts mean we have recognized an object but don't have
        # Knowledge of what the lexicalization is. So we just return an empty set
        if isinstance(concept, FunctionalObjectConcept):
            return immutableset()
        raise RuntimeError(f"Invalid concept {concept}")

    def log_hypotheses(self, log_output_path: Path) -> None:
        for concept, hypothesis in self.concepts_to_patterns().items():
            hypothesis.render_to_file(
                graph_name="perception",
                output_file=Path(
                    log_output_path / f"{str(type(self))}-{concept.debug_string}"
                ),
            )

    def concepts_to_patterns(self) -> Dict[Concept, PerceptionGraphPattern]:
        return {
            k: v
            for k, v in self._object_recognizer._concepts_to_static_patterns.items()  # pylint:disable=protected-access
        }

    @_concepts_to_templates.default
    def _init_concepts_to_templates(
        self,
    ) -> ImmutableSetMultiDict[Concept, SurfaceTemplate]:
        # Ground is added explicitly to this list because the code
        # Which matches the ground matches by recognition and not shape
        # See: `ObjectRecognizer.match_objects`
        return immutablesetmultidict(
            (
                concept,
                SurfaceTemplate.for_object_name(name, language_mode=self._language_mode),
            )
            for (concept, name) in (
                list(
                    self._object_recognizer._concepts_to_names.items()  # pylint:disable=protected-access
                )
                + [(GROUND_OBJECT_CONCEPT, "ground")]
            )
        )


@attrs
class PursuitObjectLearner(AbstractPursuitLearner, AbstractObjectTemplateLearner):
    """
    An implementation of pursuit learner for object recognition
    """

    def _new_concept(self, debug_string: str) -> ObjectConcept:
        return ObjectConcept(debug_string)

    def _hypotheses_from_perception(
        self,
        learning_state: LanguagePerceptionSemanticAlignment,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,
    ) -> AbstractSet[PerceptionGraphTemplate]:
        if bound_surface_template.slot_to_semantic_node:
            raise RuntimeError(
                "Object learner should not have slot to semantic node alignments!"
            )

        return immutableset(
            PerceptionGraphTemplate(
                graph_pattern=PerceptionGraphPattern.from_graph(
                    candidate_object
                ).perception_graph_pattern,
                template_variable_to_pattern_node=immutabledict(),
            )
            for candidate_object in extract_candidate_objects(
                learning_state.perception_semantic_alignment.perception_graph,
                sort_by_increasing_size=False,
            )
        )

    # I can't spot the difference in arguments pylint claims?
    def _keep_hypothesis(  # pylint: disable=arguments-differ
        self,
        hypothesis: PerceptionGraphTemplate,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,  # pylint:disable=unused-argument
    ) -> bool:
        if len(hypothesis.graph_pattern) < 3:
            # A two node graph is to small to meaningfully describe an object
            return False
        if all(isinstance(node, ObjectPerception) for node in hypothesis.graph_pattern):
            # A hypothesis which consists of just sub-object structure
            # with no other content is insufficiently distinctive.
            return False
        return True

    @attrs(frozen=True)
    class ObjectHypothesisPartialMatch(AbstractPursuitLearner.PartialMatch):
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
        required_alignments: Mapping[
            SyntaxSemanticsVariable, ObjectSemanticNode
        ],  # pylint:disable=unused-argument
    ) -> "PursuitObjectLearner.ObjectHypothesisPartialMatch":
        pattern = hypothesis.graph_pattern
        hypothesis_pattern_common_subgraph = get_largest_matching_pattern(
            pattern,
            graph,
            debug_callback=self._debug_callback,
            graph_logger=self._hypothesis_logger,
            ontology=self._ontology,
            match_mode=MatchMode.OBJECT,
        )
        self.debug_counter += 1

        leading_hypothesis_num_nodes = len(pattern)
        num_nodes_matched = (
            len(hypothesis_pattern_common_subgraph.copy_as_digraph().nodes)
            if hypothesis_pattern_common_subgraph
            else 0
        )

        return PursuitObjectLearner.ObjectHypothesisPartialMatch(
            PerceptionGraphTemplate(graph_pattern=hypothesis_pattern_common_subgraph)
            if hypothesis_pattern_common_subgraph
            else None,
            num_nodes_matched=num_nodes_matched,
            num_nodes_in_pattern=leading_hypothesis_num_nodes,
        )

    def _find_identical_hypothesis(
        self,
        new_hypothesis: PerceptionGraphTemplate,
        candidates: Iterable[PerceptionGraphTemplate],
    ) -> Optional[PerceptionGraphTemplate]:
        for candidate in candidates:
            if new_hypothesis.graph_pattern.check_isomorphism(candidate.graph_pattern):
                return candidate
        return None

    # pylint:disable=abstract-method
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
            for (i, hypothesis) in enumerate(hypotheses_to_scores.keys()):
                hypothesis.render_to_file(
                    concept.debug_string, log_output_path / f"{concept.debug_string}.{i}"
                )

    def _match_template(
        self,
        *,
        concept: Concept,
        pattern: PerceptionGraphTemplate,
        perception_graph: PerceptionGraph,
        confidence: float,
    ) -> Iterable[Tuple[PerceptionGraphPatternMatch, SemanticNode]]:
        # In the case of the object learner,
        # A template only has to match once; we don't care about finding additional matches.
        match = first(
            super()._match_template(
                concept=concept,
                pattern=pattern,
                perception_graph=perception_graph,
                confidence=confidence,
            ),
            None,
        )
        if match is not None:
            yield match
