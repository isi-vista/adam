import itertools
from abc import ABC

from typing import AbstractSet, Iterable, Optional, Tuple
from adam.learner.alignments import (
    LanguagePerceptionSemanticAlignment,
    PerceptionSemanticAlignment,
)
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.subset import AbstractTemplateSubsetLearnerNew
from adam.learner.surface_templates import SurfaceTemplateBoundToSemanticNodes
from adam.learner.template_learner import AbstractTemplateLearnerNew
from adam.perception import MatchMode
from adam.perception.perception_graph import PerceptionGraph
from adam.semantics import ActionConcept, SemanticNode
from attr import attrs
from immutablecollections import immutableset, immutablesetmultidict
from adam.learner.learner_utils import candidate_templates, AlignmentSlots

# This is the maximum number of tokens we will hypothesize
# as the non-argument-slots portion of a surface template for an action.
_MAXIMUM_ACTION_TEMPLATE_TOKEN_LENGTH = 3


@attrs
class AbstractVerbTemplateLearnerNew(AbstractTemplateLearnerNew, ABC):
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


@attrs
class SubsetVerbLearnerNew(
    AbstractTemplateSubsetLearnerNew, AbstractVerbTemplateLearnerNew
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

    def _new_concept(self, debug_string: str) -> ActionConcept:
        return ActionConcept(debug_string)

    def _keep_hypothesis(
        self,
        *,
        hypothesis: PerceptionGraphTemplate,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes
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
