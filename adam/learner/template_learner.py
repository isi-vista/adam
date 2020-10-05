import logging

from attr.validators import instance_of
from abc import ABC, abstractmethod

from typing import (
    AbstractSet,
    Iterable,
    List,
    Mapping,
    Sequence,
    Tuple,
    Union,
    cast,
    Set,
    Optional,
)

from more_itertools import one, first
from networkx import connected_components
from vistautils.iter_utils import only

from adam.language import LinguisticDescription, TokenSequenceLinguisticDescription
from adam.learner import ComposableLearner, LearningExample, TopLevelLanguageLearner
from adam.learner.alignments import (
    LanguagePerceptionSemanticAlignment,
    PerceptionSemanticAlignment,
)
from adam.learner.language_mode import LanguageMode
from adam.learner.learner_utils import (
    pattern_match_to_description,
    pattern_match_to_semantic_node,
)
from adam.learner.object_recognizer import (
    PerceptionGraphFromObjectRecognizer,
    _get_root_object_perception,
    replace_object_match,
)
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.surface_templates import (
    SurfaceTemplate,
    SurfaceTemplateBoundToSemanticNodes,
)
from adam.learner.fallback_learner import ActionFallbackLearnerProtocol
from adam.ontology import IN_REGION
from adam.ontology.phase1_ontology import PART_OF
from adam.perception import PerceptualRepresentation, ObjectPerception, MatchMode
from adam.perception.deprecated import LanguageAlignedPerception
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception.perception_graph import (
    PerceptionGraph,
    PerceptionGraphPatternMatch,
    HoldsAtTemporalScopePredicate,
    RelationTypeIsPredicate,
    PatternMatching,
    AnyObjectPerception,
    ObjectSemanticNodePerceptionPredicate,
    PerceptionGraphPattern,
    IsOntologyNodePredicate,
    RegionPredicate,
    REFERENCE_OBJECT_LABEL,
)
from adam.semantics import (
    Concept,
    ActionConcept,
    ObjectConcept,
    ObjectSemanticNode,
    SemanticNode,
    FunctionalObjectConcept,
    SyntaxSemanticsVariable,
    ActionSemanticNode,
)
from attr import attrib, attrs, evolve
from immutablecollections import immutabledict, immutableset
from vistautils.preconditions import check_state

from adam.utils.networkx_utils import subgraph


@attrs
class AbstractTemplateLearner(
    TopLevelLanguageLearner[
        DevelopmentalPrimitivePerceptionFrame, TokenSequenceLinguisticDescription
    ],
    ABC,
):
    _observation_num: int = attrib(init=False, default=0)
    _language_mode: LanguageMode = attrib(
        validator=instance_of(LanguageMode), kw_only=True
    )

    def observe(
        self,
        learning_example: LearningExample[
            DevelopmentalPrimitivePerceptionFrame, LinguisticDescription
        ],
        observation_num: int = -1,
    ) -> None:
        if observation_num >= 0:
            logging.info(
                "Observation %s: %s",
                observation_num,
                learning_example.linguistic_description.as_token_string(),
            )
        else:
            logging.info(
                "Observation %s: %s",
                self._observation_num,
                learning_example.linguistic_description.as_token_string(),
            )
        self._observation_num += 1

        self._assert_valid_input(learning_example)

        # Pre-processing steps will be different depending on
        # what sort of structures we are running.
        preprocessed_input = self._preprocess_scene_for_learning(
            LanguageAlignedPerception(
                language=learning_example.linguistic_description,
                perception_graph=self._extract_perception_graph(
                    learning_example.perception
                ),
                node_to_language_span=immutabledict(),
            )
        )

        logging.info(f"Learner observing {preprocessed_input}")

        surface_template = self._extract_surface_template(
            preprocessed_input, self._language_mode
        )
        self._learning_step(preprocessed_input, surface_template)

    def describe(
        self, perception: PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame]
    ) -> Mapping[LinguisticDescription, float]:
        self._assert_valid_input(perception)

        original_perception_graph = self._extract_perception_graph(perception)
        preprocessing_result = self._preprocess_scene_for_description(
            original_perception_graph
        )
        preprocessed_perception_graph = preprocessing_result.perception_graph
        matched_objects_to_names = (
            preprocessing_result.description_to_matched_object_node.inverse()
        )
        # This accumulates our output.
        match_to_score: List[
            Tuple[TokenSequenceLinguisticDescription, PerceptionGraphTemplate, float]
        ] = []

        # We pull this out into a function because we do matching in two passes:
        # first against templates whose meanings we are sure of (=have lexicalized)
        # and then, if no match has been found, against those we are still learning.
        def match_template(
            *,
            description_template: SurfaceTemplate,
            pattern: PerceptionGraphTemplate,
            score: float,
        ) -> None:
            # try to see if (our model of) its semantics is present in the situation.
            matcher = pattern.graph_pattern.matcher(
                preprocessed_perception_graph,
                match_mode=MatchMode.NON_OBJECT,
                # debug_callback=self._debug_callback,
            )
            for match in matcher.matches(use_lookahead_pruning=True):
                # if there is a match, use that match to describe the situation.
                match_to_score.append(
                    (
                        pattern_match_to_description(
                            surface_template=description_template,
                            pattern=pattern,
                            match=match,
                            matched_objects_to_names=matched_objects_to_names,
                        ),
                        pattern,
                        score,
                    )
                )
                # A template only has to match once; we don't care about finding additional matches.
                return

        # For each template whose semantics we are certain of (=have been added to the lexicon)
        for (surface_template, graph_pattern, score) in self._primary_templates():
            match_template(
                description_template=surface_template, pattern=graph_pattern, score=score
            )

        if not match_to_score:
            # Try to match against patterns being learned
            # only if no lexicalized pattern was matched.
            for (surface_template, graph_pattern, score) in self._fallback_templates():
                match_template(
                    description_template=surface_template,
                    pattern=graph_pattern,
                    score=score,
                )
        return immutabledict(self._post_process_descriptions(match_to_score))

    @abstractmethod
    def _assert_valid_input(
        self,
        to_check: Union[
            LearningExample[DevelopmentalPrimitivePerceptionFrame, LinguisticDescription],
            PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame],
        ],
    ) -> None:
        """
        Check that the learner is capable of handling this sort of learning example
        (at training time) or perception (at description time).
        """

    @abstractmethod
    def _extract_perception_graph(
        self, perception: PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame]
    ) -> PerceptionGraph:
        """
        Transforms the observed *perception* into a `PerceptionGraph`.

        This should just do the basic transformation.
        Leave further processing on the graph for `_preprocess_scene_for_learning`
        and `preprocess_scene_for_description`.
        """

    @abstractmethod
    def _preprocess_scene_for_learning(
        self, language_concept_alignment: LanguageAlignedPerception
    ) -> LanguageAlignedPerception:
        """
        Does any preprocessing necessary before the learning process begins.

        This will typically share some common code with `_preprocess_scene_for_description`.
        """

    @abstractmethod
    def _preprocess_scene_for_description(
        self, perception_graph: PerceptionGraph
    ) -> PerceptionGraphFromObjectRecognizer:
        """
        Does any preprocessing necessary before attempting to describe a scene.

        This will typically share some common code with `_preprocess_scene_for_learning`.
        """

    @abstractmethod
    def _extract_surface_template(
        self,
        language_concept_alignment: LanguageAlignedPerception,
        language_mode: LanguageMode = LanguageMode.ENGLISH,
    ) -> SurfaceTemplate:
        r"""
        We treat learning as acquiring an association between "templates"
        over the token sequence and `PerceptionGraphTemplate`\ s.

        This method determines the surface template we are trying to learn semantics for
        for this particular training example.
        """

    def _learning_step(
        self,
        preprocessed_input: LanguageAlignedPerception,
        surface_template: SurfaceTemplate,
    ) -> None:
        pass

    def _primary_templates(
        self
    ) -> Iterable[Tuple[SurfaceTemplate, PerceptionGraphTemplate, float]]:
        pass

    def _fallback_templates(
        self
    ) -> Iterable[Tuple[SurfaceTemplate, PerceptionGraphTemplate, float]]:
        pass

    def _post_process_descriptions(
        self,
        match_results: Sequence[
            Tuple[TokenSequenceLinguisticDescription, PerceptionGraphTemplate, float]
        ],
    ) -> Mapping[TokenSequenceLinguisticDescription, float]:
        return immutabledict(
            (description, score) for (description, _, score) in match_results
        )


class TemplateLearner(ComposableLearner, ABC):
    _language_mode: LanguageMode = attrib(validator=instance_of(LanguageMode))

    @abstractmethod
    def templates_for_concept(self, concept: Concept) -> AbstractSet[SurfaceTemplate]:
        pass


@attrs
class AbstractTemplateLearnerNew(TemplateLearner, ABC):
    """
    Super-class for learners using template-based syntax-semantics mappings.
    """

    _observation_num: int = attrib(init=False, default=0)
    _language_mode: LanguageMode = attrib(validator=instance_of(LanguageMode))
    _action_fallback_learners: Sequence[ActionFallbackLearnerProtocol] = attrib(
        kw_only=True, converter=tuple, default=tuple()
    )

    def learn_from(
        self,
        language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment,
        observation_num: int = -1,
    ) -> None:
        if observation_num >= 0:
            logging.info(
                "Observation %s: %s",
                observation_num,
                language_perception_semantic_alignment.language_concept_alignment.language.as_token_string(),
            )
        else:
            logging.info(
                "Observation %s: %s",
                self._observation_num,
                language_perception_semantic_alignment.language_concept_alignment.language.as_token_string(),
            )

        self._observation_num += 1

        if not self._can_learn_from(language_perception_semantic_alignment):
            return

        # Pre-processing steps will be different depending on
        # what sort of structures we are running.
        preprocessed_input = evolve(
            language_perception_semantic_alignment,
            perception_semantic_alignment=self._preprocess_scene(
                language_perception_semantic_alignment.perception_semantic_alignment
            ),
        )

        for thing_whose_meaning_to_learn in self._candidate_templates(
            language_perception_semantic_alignment
        ):
            self._learning_step(preprocessed_input, thing_whose_meaning_to_learn)

    def enrich_during_learning(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> LanguagePerceptionSemanticAlignment:
        (
            perception_post_enrichment,
            newly_recognized_semantic_nodes,
        ) = self._enrich_common(
            language_perception_semantic_alignment.perception_semantic_alignment
        )
        return LanguagePerceptionSemanticAlignment(
            # We need to link the things we found to the language
            # so later learning stages can (a) know they are already covered
            # and (b) use this information in forming the surface templates.
            language_concept_alignment=language_perception_semantic_alignment.language_concept_alignment.copy_with_new_nodes(
                immutabledict(
                    # TODO: we currently handle only one template per concept
                    (
                        semantic_node,
                        one(self.templates_for_concept(semantic_node.concept)),
                    )
                    for semantic_node in newly_recognized_semantic_nodes
                    # We make an exception for a specific type of ObjectConcept which
                    # Indicates that we know this root is an object but we don't know
                    # how to refer to it linguisticly
                    if not isinstance(semantic_node.concept, FunctionalObjectConcept)
                ),
                filter_out_duplicate_alignments=True,
                # It's okay if we recognize objects we know how to describe,
                # but they just happen not to be mentioned in the linguistic description.
                fail_if_surface_templates_do_not_match_language=False,
            ),
            perception_semantic_alignment=perception_post_enrichment,
        )

    def enrich_during_description(
        self, perception_semantic_alignment: PerceptionSemanticAlignment
    ) -> PerceptionSemanticAlignment:
        # The other information returned by _enrich_common is only needed by
        # enrich_during_learning.
        return self._enrich_common(perception_semantic_alignment)[0]

    @abstractmethod
    def _enrich_post_process(
        self,
        perception_graph_after_matching: PerceptionGraph,
        immutable_new_nodes: AbstractSet[SemanticNode],
    ) -> Tuple[PerceptionGraph, AbstractSet[SemanticNode]]:
        """
        Allows a learner to do specific enrichment post-processing if needed
        """

    def _enrich_common(
        self, perception_semantic_alignment: PerceptionSemanticAlignment
    ) -> Tuple[PerceptionSemanticAlignment, AbstractSet[SemanticNode]]:
        """
        Shared code between `enrich_during_learning` and `enrich_during_description`.
        """
        preprocessing_result = self._preprocess_scene(perception_semantic_alignment)

        preprocessed_perception_graph = preprocessing_result.perception_graph

        # This accumulates our output.
        match_to_score: List[Tuple[SemanticNode, float]] = []

        # In the case of objects only, we alter the perception graph once they
        # are recognized by replacing the matched portion of the graph with the
        # ObjectSemanticNodes.  We gather them as we match and do the replacement below.
        matched_objects: List[Tuple[SemanticNode, PerceptionGraphPatternMatch]] = []

        # We pull this out into a function because we do matching in two passes:
        # first against templates whose meanings we are sure of (=have lexicalized)
        # and then, if no match has been found, against those we are still learning.
        def match_template(
            *, concept: Concept, pattern: PerceptionGraphTemplate, score: float
        ) -> bool:
            # try to see if (our model of) its semantics is present in the situation.
            matcher = pattern.graph_pattern.matcher(
                preprocessed_perception_graph,
                match_mode=MatchMode.NON_OBJECT,
                # debug_callback=self._debug_callback,
            )
            for match in matcher.matches(use_lookahead_pruning=True):
                # if there is a match, use that match to describe the situation.
                semantic_node_for_match = pattern_match_to_semantic_node(
                    concept=concept, pattern=pattern, match=match
                )
                match_to_score.append((semantic_node_for_match, score))
                # We want to replace object matches with their semantic nodes,
                # but we don't want to alter the graph while matching it,
                # so we accumulate these to replace later.
                if isinstance(concept, ObjectConcept):
                    matched_objects.append((semantic_node_for_match, match))
                # A template only has to match once; we don't care about finding additional matches.
                return True

            # If we reach this point, matching has failed. Handle this appropriately.
            # If we're in an action learner...
            if self._can_handle_failure(concept=concept, pattern=pattern):
                # Figure out where we failed.
                #
                # We have to do a cast because the type system doesn't know that we're guaranteed to
                # fail at this point (having already tried to do the match and having failed).
                failure = cast(
                    PatternMatching.MatchFailure, matcher.first_match_or_failure_info()
                )
                return self._on_match_failure(
                    failure=failure,
                    concept=concept,
                    pattern=pattern,
                    score=score,
                    match_template=match_template,
                )

        # For each template whose semantics we are certain of (=have been added to the lexicon)
        for (concept, graph_pattern, score) in self._primary_templates():
            check_state(isinstance(graph_pattern, PerceptionGraphTemplate))
            if (
                preprocessed_perception_graph.dynamic
                == graph_pattern.graph_pattern.dynamic
            ):
                match_template(concept=concept, pattern=graph_pattern, score=score)
            else:
                logging.debug(
                    f"Unable to try and match {concept} to {preprocessed_perception_graph} "
                    f"because both patterns must be static or dynamic"
                )
        if not match_to_score:
            # Try to match against patterns being learned
            # only if no lexicalized pattern was matched.
            for (concept, graph_pattern, score) in self._fallback_templates():
                # we may have multiple pattern hypotheses for a single concept, in which case we only want to identify the concept once
                if not any(m[0].concept == concept for m in match_to_score):
                    match_template(concept=concept, pattern=graph_pattern, score=score)

        perception_graph_after_matching = perception_semantic_alignment.perception_graph

        # Replace any objects found
        def by_pattern_complexity(pair):
            _, pattern_match = pair
            return len(pattern_match.matched_pattern)

        matched_objects.sort(key=by_pattern_complexity, reverse=True)
        already_replaced: Set[ObjectPerception] = set()
        new_nodes: List[SemanticNode] = []
        for (matched_object_node, pattern_match) in matched_objects:
            root: ObjectPerception = _get_root_object_perception(
                pattern_match.matched_sub_graph._graph,  # pylint:disable=protected-access
                immutableset(
                    pattern_match.matched_sub_graph._graph.nodes,  # pylint:disable=protected-access
                    disable_order_check=True,
                ),
            )
            if root not in already_replaced:
                perception_graph_after_matching = replace_object_match(
                    replacement_object_node=cast(ObjectSemanticNode, matched_object_node),
                    current_perception=perception_graph_after_matching,
                    pattern_match=pattern_match,
                    remove_internal_structure=False,
                )
                already_replaced.add(root)
                new_nodes.append(matched_object_node)
            else:
                logging.info(
                    f"Matched pattern for {matched_object_node} "
                    f"but root object {root} already replaced."
                )
        if matched_objects:
            immutable_new_nodes = immutableset(new_nodes)
        else:
            immutable_new_nodes = immutableset(node for (node, _) in match_to_score)

        (
            perception_graph_after_post_processing,
            nodes_after_post_processing,
        ) = self._enrich_post_process(
            perception_graph_after_matching, immutable_new_nodes
        )

        return (
            perception_semantic_alignment.copy_with_updated_graph_and_added_nodes(
                new_graph=perception_graph_after_post_processing,
                new_nodes=nodes_after_post_processing,
            ),
            nodes_after_post_processing,
        )

    def _can_handle_failure(
        self, *, concept: Concept, _pattern: PerceptionGraphTemplate, _score: float
    ) -> bool:
        return isinstance(concept, ActionConcept)

    def _on_match_failure(
        self,
        *,
        failure: PatternMatching.MatchFailure,
        concept: Concept,
        pattern: PerceptionGraphTemplate,
        score: float,
        match_template,
    ) -> bool:
        if isinstance(concept, ActionConcept):
            # Handle the case where we failed on the internal structure of a slot.
            slot_pattern_nodes = immutableset(
                pattern.template_variable_to_pattern_node.values()
            )
            unmatched_pattern_nodes = immutableset(
                [
                    pattern_node
                    for pattern_node in pattern.graph_pattern._graph.nodes  # pylint:disable=protected-access
                    if pattern_node
                    not in failure.largest_match_pattern_subgraph._graph.nodes  # pylint:disable=protected-access
                ]
            )

            # Here, we simplify the logic slightly
            # by assuming the edge will be a HoldsAtTemporalScopePredicate;
            # it should be, since the integrated learner
            # should never run the action learner on a static graph.
            def is_part_of_predicate(predicate: HoldsAtTemporalScopePredicate):
                unwrapped_predicate = predicate.wrapped_edge_predicate
                return (
                    isinstance(unwrapped_predicate, RelationTypeIsPredicate)
                    and unwrapped_predicate.relation_type == PART_OF
                )

            # If the slot pattern nodes all matched,
            # AND we *failed* to match a pattern node that's `partOf` one of the slots...
            if not slot_pattern_nodes.intersection(unmatched_pattern_nodes):
                # First, construct an action semantic node corresponding to the potential match.
                semantics = ActionSemanticNode(
                    concept=concept,
                    slot_fillings=immutabledict(
                        [
                            (
                                slot,
                                failure.pattern_node_to_graph_node_for_largest_match[
                                    pattern_node
                                ],
                            )
                            for slot, pattern_node in pattern.template_variable_to_pattern_node.items()
                        ]
                    ),
                )
                for (
                    slot,
                    slot_pattern_node,
                ) in pattern.template_variable_to_pattern_node.items():
                    # If any slot has an unmatched *direct* subobject...
                    #
                    # HACK. Probably this should check for indirect subobjects of the slot, too.
                    # -JAC
                    if any(
                        subobject in unmatched_pattern_nodes
                        and is_part_of_predicate(predicate)
                        for subobject, _, predicate in pattern.graph_pattern._graph.in_edges(
                            slot_pattern_node, data="predicate"
                        )
                    # *and* if any fallback learner says we can ignore this failure...
                    ) and any(
                        fallback_learner.ignore_slot_internal_structure_failure(
                            semantics, slot
                        )
                        for fallback_learner in self._action_fallback_learners
                    ):
                        logging.debug(
                            "Fallback learner says that we can ignore internal structure failure "
                            "for %s (failed slot was %s)",
                            semantics,
                            slot,
                        )
                        # Excise the internal structure of the failed slot part of the pattern
                        fixed_pattern = _delete_subobjects_of_object_in_pattern(
                            pattern.graph_pattern, slot_pattern_node
                        )
                        # Only proceed if all of the slots are in the pattern.
                        #
                        # This should always be true happen, because for it to be false, one
                        # of the slots would have to be part of another.
                        if all(
                            slot_pattern_node
                            in fixed_pattern._graph  # pylint:disable=protected-access
                            for slot_pattern_node in slot_pattern_nodes
                        ):
                            # Make a new PerceptionGraphTemplate, excising the failed part
                            updated_template = PerceptionGraphTemplate(
                                graph_pattern=fixed_pattern,
                                template_variable_to_pattern_node=pattern.template_variable_to_pattern_node,
                            )
                            # We use an if so that we will fall through if this fails.
                            if match_template(
                                concept=concept, pattern=updated_template, score=score
                            ):
                                return True

            # Otherwise, maybe a root-level matched object matched but one of its subobjects
            # failed to match.
            #
            # We can assume it will be an ObjectSemanticNode since (if we are running in the
            # integrated learner) an object learner should already have done the replacing for
            # us.
            matched_root_node_with_unmatched_subobject: Optional[
                Union[AnyObjectPerception, ObjectSemanticNodePerceptionPredicate]
            ] = first(
                (
                    object_node
                    for object_node in failure.largest_match_pattern_subgraph
                    if object_node not in pattern.pattern_node_to_template_variable
                    and not any(
                        is_part_of_predicate(predicate)
                        for _, _, predicate in pattern.graph_pattern._graph.out_edges(  # pylint:disable=protected-access
                            object_node, data="predicate"
                        )
                    )
                    and any(
                        is_part_of_predicate(predicate)
                        for subobject, _, predicate in pattern.graph_pattern._graph.in_edges(  # pylint:disable=protected-access
                            object_node, data="predicate"
                        )
                        if subobject not in failure.largest_match_pattern_subgraph
                    )
                ),
                None,
            )
            if matched_root_node_with_unmatched_subobject:
                # Excise the internal structure of the failed slot part of the pattern
                fixed_pattern = _delete_subobjects_of_object_in_pattern(
                    pattern.graph_pattern, matched_root_node_with_unmatched_subobject
                )
                # All of the slot pattern nodes must still be around for this to work.
                #
                # This should never happen, because it would require one of the slots to
                # be part of the other.
                if all(
                    slot_pattern_node in fixed_pattern._graph
                    for slot_pattern_node in slot_pattern_nodes
                ):
                    # Make a new PerceptionGraphTemplate, excising the failed part
                    updated_template = PerceptionGraphTemplate(
                        graph_pattern=fixed_pattern,
                        template_variable_to_pattern_node=pattern.template_variable_to_pattern_node,
                    )
                    if match_template(
                        concept=concept, pattern=updated_template, score=score
                    ):
                        return True
        return False

    @abstractmethod
    def _learning_step(
        self,
        language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,
    ) -> None:
        """
        Perform the actual learning logic.
        """

    @abstractmethod
    def _can_learn_from(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> bool:
        """
        Check that the learner is capable of learning from this sort of learning example
        (at training time).
        """

    @abstractmethod
    def _preprocess_scene(
        self, perception_semantic_alignment: PerceptionSemanticAlignment
    ) -> PerceptionSemanticAlignment:
        """
        Does any preprocessing necessary before attempting to process a scene.
        """

    @abstractmethod
    def _candidate_templates(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> AbstractSet[SurfaceTemplateBoundToSemanticNodes]:
        r"""
        We treat learning as acquiring an association between "templates"
        over the token sequence and `PerceptionGraphTemplate`\ s.

        This method determines the surface template we are trying to learn semantics for
        for this particular training example.
        """

    @abstractmethod
    def _primary_templates(
        self
    ) -> Iterable[Tuple[Concept, PerceptionGraphTemplate, float]]:
        """
        Our high-confidence (e.g. lexicalized) templates to match when describing a scene.
        """

    @abstractmethod
    def _fallback_templates(
        self
    ) -> Iterable[Tuple[Concept, PerceptionGraphTemplate, float]]:
        """
        Get the secondary templates to try during description if none of the primary templates
        matched.
        """


def _delete_subobjects_of_object_in_pattern(
    pattern: PerceptionGraphPattern,
    object_: Union[ObjectSemanticNodePerceptionPredicate, AnyObjectPerception],
) -> PerceptionGraphPattern:
    """
    Given a perception graph from after matching,
    return a new perception graph
    where we have removed every subobject of the given object node.

    Note that this does not clean up hanging nodes.
    """
    digraph = pattern._graph  # pylint:disable=protected-access

    def unwrap_predicate_if_wrapped(predicate) -> bool:
        return (
            predicate.wrapped_edge_predicate
            if isinstance(predicate, HoldsAtTemporalScopePredicate)
            else predicate
        )

    def is_relation_type_predicate(predicate, relation_type) -> bool:
        unwrapped_predicate = unwrap_predicate_if_wrapped(predicate)
        return (
            isinstance(unwrapped_predicate, RelationTypeIsPredicate)
            and unwrapped_predicate.relation_type == relation_type
        )

    def is_reference_object_predicate(predicate) -> bool:
        return is_relation_type_predicate(predicate, REFERENCE_OBJECT_LABEL)

    def is_in_region_predicate(predicate) -> bool:
        return is_relation_type_predicate(predicate, IN_REGION)

    def is_only_object_in_region(subobject, region) -> bool:
        return any(
            thing_in_region != subobject
            for thing_in_region, _, predicate in digraph.in_edges(
                region, data="predicate"
            )
            if is_in_region_predicate(predicate)
        )

    def is_part_of_predicate(predicate) -> bool:
        unwrapped_predicate = unwrap_predicate_if_wrapped(predicate)
        return (
            isinstance(unwrapped_predicate, RelationTypeIsPredicate)
            and unwrapped_predicate.relation_type == PART_OF
        )

    # This function could be more efficient. It seems efficient enough for now.
    def get_things_node_is_a_part_of(node):
        part_of = set()
        visited = set()
        to_visit = {node}
        while to_visit:
            current_node = to_visit.pop()
            visited.add(current_node)
            for _, successor, predicate in digraph.out_edges(
                current_node, data="predicate"
            ):
                if is_part_of_predicate(predicate):
                    part_of.add(successor)
                    to_visit.add(successor)

        return part_of

    subobjects = immutableset(
        [node for node in digraph.nodes if object_ in get_things_node_is_a_part_of(node)]
    )
    subobject_properties = immutableset(
        [
            node
            for subobject in subobjects
            for node in digraph.successors(subobject)
            if isinstance(node, IsOntologyNodePredicate)
        ]
    )
    subobject_regions = immutableset(
        [
            node
            for subobject in subobjects
            for node, _, predicate in digraph.in_edges(subobject, data="predicate")
            if isinstance(node, RegionPredicate)
            and is_reference_object_predicate(predicate)
        ]
        + [
            node
            for subobject in subobjects
            for _, node, predicate in digraph.out_edges(subobject, data="predicate")
            if isinstance(node, RegionPredicate)
            and is_only_object_in_region(subobject, node)
        ]
    )
    prune = subobjects | subobject_properties | subobject_regions

    pattern_digraph_without_subobjects = subgraph(
        digraph, immutableset([node for node in digraph.nodes if node not in prune])
    )

    fixed_pattern_as_undirected_graph = pattern_digraph_without_subobjects.to_undirected(
        as_view=True
    )
    try:
        # There should be exactly one connected component in the resulting graph.
        only(connected_components(fixed_pattern_as_undirected_graph))
    except ValueError:
        raise RuntimeError(
            "Removing subobjects of object results in a disconnected or empty pattern graph."
        )

    return PerceptionGraphPattern(
        pattern_digraph_without_subobjects, dynamic=pattern.dynamic
    )
