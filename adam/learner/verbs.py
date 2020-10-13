import itertools
import logging
from abc import ABC

from typing import AbstractSet, Mapping, Union, Iterable, Optional, Tuple, cast, Any

from more_itertools import only
from networkx import connected_components

from adam.learner.fallback_learner import ActionFallbackLearnerProtocol
from adam.learner.language_mode import LanguageMode
from adam.language import LinguisticDescription
from adam.learner import LearningExample
from adam.learner.alignments import (
    LanguagePerceptionSemanticAlignment,
    PerceptionSemanticAlignment,
)
from adam.learner.object_recognizer import (
    ObjectRecognizer,
    PerceptionGraphFromObjectRecognizer,
)
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.subset import (
    AbstractTemplateSubsetLearner,
    AbstractTemplateSubsetLearnerNew,
)
from adam.learner.surface_templates import (
    STANDARD_SLOT_VARIABLES,
    SurfaceTemplate,
    SurfaceTemplateBoundToSemanticNodes,
)
from adam.learner.template_learner import (
    AbstractTemplateLearner,
    AbstractTemplateLearnerNew,
)
from adam.ontology import IN_REGION
from adam.ontology.phase1_ontology import PART_OF
from adam.perception import PerceptualRepresentation, MatchMode
from adam.perception.deprecated import LanguageAlignedPerception
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception.perception_graph import (
    PerceptionGraph,
    PatternMatching,
    HoldsAtTemporalScopePredicate,
    RelationTypeIsPredicate,
    REFERENCE_OBJECT_LABEL,
    PerceptionGraphPattern,
    ObjectSemanticNodePerceptionPredicate,
    AnyObjectPerception,
    IsOntologyNodePredicate,
    RegionPredicate,
    PerceptionGraphPatternMatch,
)
from adam.semantics import (
    ActionConcept,
    ObjectSemanticNode,
    SyntaxSemanticsVariable,
    SemanticNode,
    Concept,
    ActionSemanticNode,
)
from attr import attrib, attrs
from immutablecollections import (
    immutabledict,
    immutableset,
    immutablesetmultidict,
    ImmutableSet,
)
from attr.validators import instance_of, deep_iterable
from adam.learner.learner_utils import (
    candidate_templates,
    AlignmentSlots,
    pattern_remove_incomplete_region_or_spatial_path,
)

# This is the maximum number of tokens we will hypothesize
# as the non-argument-slots portion of a surface template for an action.
from adam.utils.networkx_utils import subgraph

_MAXIMUM_ACTION_TEMPLATE_TOKEN_LENGTH = 3


# Workaround for https://github.com/python/mypy/issues/8389
def _action_fallback_learner_tuple(
    fallback_learners: Iterable[ActionFallbackLearnerProtocol]
) -> Tuple[ActionFallbackLearnerProtocol, ...]:
    return tuple(fallback_learners)


@attrs
class AbstractVerbTemplateLearnerNew(AbstractTemplateLearnerNew, ABC):
    _action_fallback_learners: Tuple[ActionFallbackLearnerProtocol, ...]

    # pylint:disable=abstract-method
    def _candidate_templates(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> AbstractSet[SurfaceTemplateBoundToSemanticNodes]:
        def candidate_verb_templates() -> Iterable[Tuple[AlignmentSlots, ...]]:
            # This function returns templates for the candidate verb templates
            # Terminology:
            # (A)rgument - Noun
            # (F)ixedString - A collection of str tokens which can be the verb or a modifier

            # First let's handle only one argument - Intransitive Verbs
            # This generates templates for examples like "Mom falls"
            for output in immutableset(
                itertools.permutations(
                    [AlignmentSlots.Argument, AlignmentSlots.FixedString], 2
                )
            ):
                yield output
            # Now we want to handle two arguments - transitive Verbs
            # We want to handle following verb syntaxes:
            # SOV, SVO, VSO, VOS, OVS, OSV
            # However, currently our templates don't distinguish subject and object
            # So we only need to handle AAF, AFA, FAA
            # Example: "Mom throws a ball"
            # We include an extra FixedString to account for adverbial modifiers such as in the example
            # "Mom throws a ball up"
            for output in immutableset(
                itertools.permutations(
                    [
                        AlignmentSlots.Argument,
                        AlignmentSlots.Argument,
                        AlignmentSlots.FixedString,
                        AlignmentSlots.FixedString,
                    ],
                    4,
                )
            ):
                yield output
            # Now we want to handle three arguments , which can either have one or two fixed strings
            # This is either ditransitive "Mom throws me a ball"
            # or includes a locational preposition phrase "Mom falls on the ground"
            for output in immutableset(
                itertools.permutations(
                    [
                        AlignmentSlots.Argument,
                        AlignmentSlots.Argument,
                        AlignmentSlots.Argument,
                        AlignmentSlots.FixedString,
                        AlignmentSlots.FixedString,
                        AlignmentSlots.FixedString,
                    ],
                    6,
                )
            ):
                yield output

        # Generate all the possible verb template alignments
        return candidate_templates(
            language_perception_semantic_alignment,
            _MAXIMUM_ACTION_TEMPLATE_TOKEN_LENGTH,
            self._language_mode,
            candidate_verb_templates,
        )

    def _enrich_post_process(
        self,
        perception_graph_after_matching: PerceptionGraph,
        immutable_new_nodes: AbstractSet[SemanticNode],
    ) -> Tuple[PerceptionGraph, AbstractSet[SemanticNode]]:
        return perception_graph_after_matching, immutable_new_nodes

    def _handle_match_failure(
        self,
        *,
        failure: PatternMatching.MatchFailure,
        concept: Concept,
        pattern: PerceptionGraphTemplate,
        perception_graph: PerceptionGraph,
    ) -> Optional[Tuple[PerceptionGraphPatternMatch, SemanticNode]]:
        """
        Handle a match failure and return whether we were able to match after executing all
        appropriate "fallback" logic.

        This should be called by _match_template.
        """
        if not isinstance(concept, ActionConcept):
            raise RuntimeError(
                f"Verb learners should only learn action concepts, but asked to handle failure for "
                f"concept {concept} of non-action type {type(concept)}."
            )
        concept = cast(ActionConcept, concept)
        graph_pattern_digraph = pattern.graph_pattern.copy_as_digraph()

        # Handle the case where we failed on the internal structure of a slot.
        slot_pattern_nodes = immutableset(
            pattern.template_variable_to_pattern_node.values()
        )
        unmatched_pattern_nodes = immutableset(
            [
                pattern_node
                for pattern_node in graph_pattern_digraph.nodes
                if pattern_node
                not in failure.largest_match_pattern_subgraph._graph.nodes  # pylint:disable=protected-access
            ]
        )  # pylint:disable=protected-access

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
                    and _is_part_of_predicate(predicate)
                    for subobject, _, predicate in graph_pattern_digraph.in_edges(
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
                        match_attempt = self._match_template(
                            concept=concept,
                            pattern=updated_template,
                            perception_graph=perception_graph,
                        )
                        if isinstance(match_attempt, tuple):
                            return match_attempt

        # Handle the case where a root-level non-slot matched object matched but one of its
        # subobjects failed to match.
        for object_node in failure.largest_match_pattern_subgraph:
            if (
                # non-slot
                object_node not in pattern.pattern_node_to_template_variable
                # root-level
                and not any(
                    _is_part_of_predicate(predicate)
                    for _, _, predicate in graph_pattern_digraph.out_edges(
                        object_node, data="predicate"
                    )
                )
                # one of its subobjects failed to match
                and any(
                    _is_part_of_predicate(predicate)
                    for subobject, _, predicate in pattern.graph_pattern._graph.in_edges(  # pylint:disable=protected-access
                        object_node, data="predicate"
                    )
                    if subobject not in failure.largest_match_pattern_subgraph
                )
            ):
                # Excise the internal structure of the failed slot part of the pattern
                fixed_pattern = _delete_subobjects_of_object_in_pattern(
                    pattern.graph_pattern, object_node
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
                    match_attempt = self._match_template(
                        concept=concept,
                        pattern=updated_template,
                        perception_graph=perception_graph,
                    )
                    if isinstance(match_attempt, tuple):
                        return match_attempt
                break
        return None


@attrs
class AbstractVerbTemplateLearner(AbstractTemplateLearner, ABC):
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
        if isinstance(to_check, LearningExample):
            perception = to_check.perception
        else:
            perception = to_check
        if len(perception.frames) != 2:
            raise RuntimeError(
                "Expected exactly two frames in a perception for verb learning"
            )

    def _extract_perception_graph(
        self, perception: PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame]
    ) -> PerceptionGraph:
        return PerceptionGraph.from_dynamic_perceptual_representation(perception)

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
        if len(language_concept_alignment.aligned_nodes) > len(STANDARD_SLOT_VARIABLES):
            raise RuntimeError("Input has too many aligned nodes for us to handle.")

        object_node_to_template_variable: Mapping[
            ObjectSemanticNode, SyntaxSemanticsVariable
        ] = immutabledict(
            zip(language_concept_alignment.aligned_nodes, STANDARD_SLOT_VARIABLES)
        )
        return language_concept_alignment.to_surface_template(
            object_node_to_template_variable=object_node_to_template_variable,
            determiner_prefix_slots=object_node_to_template_variable.values(),
            language_mode=language_mode,
        )


@attrs
class SubsetVerbLearner(AbstractTemplateSubsetLearner, AbstractVerbTemplateLearner):
    def _hypothesis_from_perception(
        self, preprocessed_input: LanguageAlignedPerception
    ) -> PerceptionGraphTemplate:
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
        return current_pattern_hypothesis.intersection(
            previous_pattern_hypothesis,
            ontology=self._ontology,
            match_mode=MatchMode.NON_OBJECT,
            match_restrictions=immutablesetmultidict(
                [
                    (node1, node2)
                    for previous_slot, node1 in previous_pattern_hypothesis.template_variable_to_pattern_node.items()
                    for new_slot, node2 in current_pattern_hypothesis.template_variable_to_pattern_node.items()
                    if previous_slot == new_slot
                ]
            ),
        )


@attrs
class SubsetVerbLearnerNew(
    AbstractTemplateSubsetLearnerNew, AbstractVerbTemplateLearnerNew
):
    _action_fallback_learners: Tuple[ActionFallbackLearnerProtocol, ...] = attrib(
        kw_only=True,
        default=tuple(),
        validator=deep_iterable(
            member_validator=instance_of(ActionFallbackLearnerProtocol),
            iterable_validator=instance_of(tuple),
        ),
        converter=_action_fallback_learner_tuple,
    )

    def _can_learn_from(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> bool:
        return (
            len(
                language_perception_semantic_alignment.perception_semantic_alignment.semantic_nodes
            )
            > 1
        )

    def _new_concept(self, debug_string: str) -> ActionConcept:
        return ActionConcept(debug_string)

    def _keep_hypothesis(
        self,
        *,
        hypothesis: PerceptionGraphTemplate,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,
    ) -> bool:
        num_template_arguments = len(bound_surface_template.slot_to_semantic_node)
        return len(hypothesis.graph_pattern) >= 2 * num_template_arguments

    def _hypotheses_from_perception(
        self,
        learning_state: LanguagePerceptionSemanticAlignment,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,
    ) -> AbstractSet[PerceptionGraphTemplate]:
        # For the subset learner, our hypothesis is the entire graph.
        return immutableset(
            [
                PerceptionGraphTemplate.from_graph(
                    learning_state.perception_semantic_alignment.perception_graph,
                    template_variable_to_matched_object_node=bound_surface_template.slot_to_semantic_node,
                )
            ]
        )

    def _preprocess_scene(
        self, perception_semantic_alignment: PerceptionSemanticAlignment
    ) -> PerceptionSemanticAlignment:
        return perception_semantic_alignment

    def _update_hypothesis(
        self,
        previous_pattern_hypothesis: PerceptionGraphTemplate,
        current_pattern_hypothesis: PerceptionGraphTemplate,
    ) -> Optional[PerceptionGraphTemplate]:
        return current_pattern_hypothesis.intersection(
            previous_pattern_hypothesis,
            ontology=self._ontology,
            match_mode=MatchMode.NON_OBJECT,
            match_restrictions=immutablesetmultidict(
                [
                    (node1, node2)
                    for previous_slot, node1 in previous_pattern_hypothesis.template_variable_to_pattern_node.items()
                    for new_slot, node2 in current_pattern_hypothesis.template_variable_to_pattern_node.items()
                    if previous_slot == new_slot
                ]
            ),
            trim_after_match=pattern_remove_incomplete_region_or_spatial_path,
            debug_callback=self._debug_callback,
        )

    def _should_return_match_failure(
        self, concept: Concept, pattern: PerceptionGraphTemplate
    ) -> bool:
        """
        Return whether we should return a match failure when matching fails or just return None.
        """
        return True

    def _match_template(
        self,
        *,
        concept: Concept,
        pattern: PerceptionGraphTemplate,
        perception_graph: PerceptionGraph,
    ) -> Optional[Tuple[PerceptionGraphPatternMatch, SemanticNode]]:
        rtrn = super()._match_template(
            concept=concept, pattern=pattern, perception_graph=perception_graph
        )
        if isinstance(rtrn, PatternMatching.MatchFailure):
            failure = rtrn
            rtrn = self._handle_match_failure(
                failure=failure,
                concept=concept,
                pattern=pattern,
                perception_graph=perception_graph,
            )

        return rtrn


def _unwrap_predicate_if_wrapped(predicate) -> bool:
    return (
        predicate.wrapped_edge_predicate
        if isinstance(predicate, HoldsAtTemporalScopePredicate)
        else predicate
    )


def _is_relation_type_predicate(predicate, relation_type) -> bool:
    unwrapped_predicate = _unwrap_predicate_if_wrapped(predicate)
    return (
        isinstance(unwrapped_predicate, RelationTypeIsPredicate)
        and unwrapped_predicate.relation_type == relation_type
    )


def _is_part_of_predicate(predicate) -> bool:
    return _is_relation_type_predicate(predicate, PART_OF)


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

    def is_reference_object_predicate(predicate) -> bool:
        return _is_relation_type_predicate(predicate, REFERENCE_OBJECT_LABEL)

    def is_in_region_predicate(predicate) -> bool:
        return _is_relation_type_predicate(predicate, IN_REGION)

    def is_only_object_in_region(subobject, region) -> bool:
        return any(
            thing_in_region != subobject
            for thing_in_region, _, predicate in digraph.in_edges(
                region, data="predicate"
            )
            if is_in_region_predicate(predicate)
        )

    def all_subobjects_of(root_object_node):
        part_of = []
        visited = set()
        to_visit = {root_object_node}
        while to_visit:
            current_node = to_visit.pop()
            visited.add(current_node)
            for predecessor, _, predicate in digraph.in_edges(
                current_node, data="predicate"
            ):
                if _is_part_of_predicate(predicate) and predecessor not in visited:
                    part_of.append(predecessor)
                    to_visit.add(predecessor)

        return part_of

    subobjects: ImmutableSet[Any] = immutableset(all_subobjects_of(object_))
    subobject_properties: ImmutableSet[Any] = immutableset(
        [
            node
            for subobject in subobjects
            for node in digraph.successors(subobject)
            if isinstance(node, IsOntologyNodePredicate)
        ]
    )
    subobject_regions: ImmutableSet[Any] = immutableset(
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
