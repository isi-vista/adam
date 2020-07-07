import itertools
from abc import ABC
from enum import Enum, auto
from typing import AbstractSet, Mapping, Union, List, Iterable, Optional, Dict, Tuple
from adam.learner.language_mode import LanguageMode
from adam.language import LinguisticDescription
from adam.learner import LearningExample
from adam.learner.alignments import (
    LanguagePerceptionSemanticAlignment,
    PerceptionSemanticAlignment,
)
from adam.learner.learner_utils import covers_entire_utterance
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
from adam.perception import PerceptualRepresentation, MatchMode
from adam.perception.deprecated import LanguageAlignedPerception
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception.perception_graph import PerceptionGraph
from adam.semantics import (
    ActionConcept,
    ObjectSemanticNode,
    SyntaxSemanticsVariable,
    SemanticNode,
)
from attr import attrib, attrs
from immutablecollections import (
    immutabledict,
    immutableset,
    ImmutableSet,
    immutablesetmultidict,
)
from attr.validators import instance_of
from vistautils.span import Span
from adam.learner import SemanticNodeWithSpan, aligned_object_nodes

# This is the maximum number of tokens we will hypothesize
# as the non-argument-slots portion of a surface template for an action.
_MAXIMUM_ACTION_TEMPLATE_TOKEN_LENGTH = 3


class VerbAlignmentSlots(Enum):
    Argument = auto()
    FixedString = auto()


@attrs
class AbstractVerbTemplateLearnerNew(AbstractTemplateLearnerNew, ABC):
    # pylint:disable=abstract-method
    def _candidate_templates(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> AbstractSet[SurfaceTemplateBoundToSemanticNodes]:
        ret = []
        language_concept_alignment = (
            language_perception_semantic_alignment.language_concept_alignment
        )
        # We make an assumption that the order of nouns in our sentence will be
        # in the same order as they appear in the sentence, left to right,
        # To avoid calculating this condition multiple times we do so once
        # For the number of arguments (nouns) our candidate template
        # desires and store the resulting possible aligments in this dict
        # for easy look up.
        num_arguments_to_alignments_sets: Dict[
            int, ImmutableSet[Tuple[SemanticNodeWithSpan, ...]]
        ] = {}
        sentence_tokens = language_concept_alignment.language.as_token_sequence()

        # Any recognized object is a potential verb argument.
        # This method does not properly handle arguments which themselves have complex structure.
        # See https://github.com/isi-vista/adam/issues/785

        # We currently do not handle verb arguments
        # which are dropped and/or expressed only via morphology:
        # https://github.com/isi-vista/adam/issues/786

        def is_legal_template_span(candidate_verb_token_span: Span) -> bool:
            # A template token span can't exceed the bounds of the utterance
            if candidate_verb_token_span.start < 0:
                return False
            if candidate_verb_token_span.end > len(sentence_tokens):
                return False
            # or be bigger than our maximum template size...
            if len(candidate_verb_token_span) > _MAXIMUM_ACTION_TEMPLATE_TOKEN_LENGTH:
                return False

            # or we have already aligned any of the tokens in between the objects
            # to some other meaning.
            for token_index in range(
                candidate_verb_token_span.start, candidate_verb_token_span.end
            ):
                if language_concept_alignment.token_index_is_aligned(token_index):
                    return False
            return True

        def candidate_verb_templates() -> Iterable[Tuple[VerbAlignmentSlots, ...]]:
            # This function returns templates for the candidate verb templates
            # Terminology:
            # (A)rgument - Noun
            # (F)ixedString - A collection of str tokens which can be the verb or a modifier

            # First let's handle only one argument - Intransitive Verbs
            # This generates templates for examples like "Mom falls"
            for output in immutableset(
                itertools.permutations(
                    [VerbAlignmentSlots.Argument, VerbAlignmentSlots.FixedString], 2
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
                        VerbAlignmentSlots.Argument,
                        VerbAlignmentSlots.Argument,
                        VerbAlignmentSlots.FixedString,
                        VerbAlignmentSlots.FixedString,
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
                        VerbAlignmentSlots.Argument,
                        VerbAlignmentSlots.Argument,
                        VerbAlignmentSlots.Argument,
                        VerbAlignmentSlots.FixedString,
                        VerbAlignmentSlots.FixedString,
                        VerbAlignmentSlots.FixedString,
                    ],
                    6,
                )
            ):
                yield output

        def process_aligned_objects_with_template(
            verb_template: Tuple[VerbAlignmentSlots, ...],
            aligned_nodes: Tuple[SemanticNodeWithSpan, ...],
        ) -> Iterable[Optional[SurfaceTemplateBoundToSemanticNodes]]:

            aligned_node_index = 0
            template_elements: List[Union[str, SyntaxSemanticsVariable]] = []
            slot_to_semantic_node: List[Tuple[SyntaxSemanticsVariable, SemanticNode]] = []

            # We need to handle fixed strings that are pre or post fix to the rest of the
            # Sentence differently as they don't have a fixed length so we could generate
            # multiple options.
            prefix_string_end = None
            postfix_string_start = None
            # In the event we generate a candidate template like:
            # A, F, F, A then we want to compute this like A, F, A
            # So we keep track if the previous token was a FixedString indicator
            previous_node_was_string = False

            for token in verb_template:
                # If the token in our template is an argument we need to assign it a
                # unique SyntaxSemanticsVariable, and map it to the SemanticNode
                if token == VerbAlignmentSlots.Argument:
                    slot_semantic_variable = STANDARD_SLOT_VARIABLES[aligned_node_index]
                    template_elements.append(slot_semantic_variable)
                    slot_to_semantic_node.append(
                        (slot_semantic_variable, aligned_nodes[aligned_node_index].node)
                    )
                    aligned_node_index += 1
                    previous_node_was_string = False
                else:
                    # We ignore this case to process A, F, F, A like A, F, A
                    if previous_node_was_string:
                        continue
                    # We make a note of where the end of our prefix string can be
                    # Then continue as we'll handle this case afterwards
                    elif aligned_node_index == 0:
                        prefix_string_end = aligned_nodes[aligned_node_index].span.start
                    # Similiarly to above, we instead mark the start of the postfix string
                    elif aligned_node_index == len(aligned_nodes):
                        postfix_string_start = aligned_nodes[
                            aligned_node_index - 1
                        ].span.end
                    else:
                        # If our FixedString is flanked by two Arguments we just want to acquire all the tokens
                        # between them
                        if (
                            aligned_nodes[aligned_node_index - 1].span.end
                            != aligned_nodes[aligned_node_index].span.start
                        ):
                            candidate_verb_token_span = Span(
                                aligned_nodes[aligned_node_index - 1].span.end,
                                aligned_nodes[aligned_node_index].span.start,
                            )
                            if not is_legal_template_span(candidate_verb_token_span):
                                yield None
                            template_elements.extend(
                                sentence_tokens[
                                    candidate_verb_token_span.start : candidate_verb_token_span.end
                                ]
                            )
                        previous_node_was_string = True
            # We need to handle searching before or after the aligned token
            # And we could generate multiple options of different lengths
            # between 1 and _MAXIMUM_ACTION_TEMPLATE_TOKEN_LENGTH
            if prefix_string_end and postfix_string_start:
                for max_token_length_for_template_prefix in range(
                    1, _MAXIMUM_ACTION_TEMPLATE_TOKEN_LENGTH + 1
                ):
                    prefix_candidate_verb_token_span = Span(
                        prefix_string_end - max_token_length_for_template_prefix,
                        prefix_string_end,
                    )
                    if is_legal_template_span(prefix_candidate_verb_token_span):
                        for max_token_length_for_template_postfix in range(
                            1, _MAXIMUM_ACTION_TEMPLATE_TOKEN_LENGTH + 1
                        ):
                            postfix_candidate_verb_token_span = Span(
                                postfix_string_start,
                                postfix_string_start
                                + max_token_length_for_template_postfix,
                            )
                            if is_legal_template_span(postfix_candidate_verb_token_span):
                                final_template_elements: List[
                                    Union[str, SyntaxSemanticsVariable]
                                ] = list(
                                    sentence_tokens[
                                        prefix_candidate_verb_token_span.start : prefix_candidate_verb_token_span.end
                                    ]
                                )
                                final_template_elements.extend(template_elements)
                                final_template_elements.extend(
                                    sentence_tokens[
                                        postfix_candidate_verb_token_span.start : postfix_candidate_verb_token_span.end
                                    ]
                                )
                                yield SurfaceTemplateBoundToSemanticNodes(
                                    surface_template=SurfaceTemplate(
                                        elements=final_template_elements,
                                        determiner_prefix_slots=[
                                            SLOT for (SLOT, _) in slot_to_semantic_node
                                        ],
                                        language_mode=self._language_mode,
                                    ),
                                    slot_to_semantic_node=slot_to_semantic_node,
                                )
            elif prefix_string_end:
                for max_token_length_for_template_prefix in range(
                    1, _MAXIMUM_ACTION_TEMPLATE_TOKEN_LENGTH + 1
                ):
                    prefix_candidate_verb_token_span = Span(
                        prefix_string_end - max_token_length_for_template_prefix,
                        prefix_string_end,
                    )
                    if is_legal_template_span(prefix_candidate_verb_token_span):
                        final_template_elements = list(
                            sentence_tokens[
                                prefix_candidate_verb_token_span.start : prefix_candidate_verb_token_span.end
                            ]
                        )
                        final_template_elements.extend(template_elements)
                        yield SurfaceTemplateBoundToSemanticNodes(
                            surface_template=SurfaceTemplate(
                                elements=final_template_elements,
                                determiner_prefix_slots=[
                                    SLOT for (SLOT, _) in slot_to_semantic_node
                                ],
                                language_mode=self._language_mode,
                            ),
                            slot_to_semantic_node=slot_to_semantic_node,
                        )
            elif postfix_string_start:
                for max_token_length_for_template_postfix in range(
                    1, _MAXIMUM_ACTION_TEMPLATE_TOKEN_LENGTH + 1
                ):
                    postfix_candidate_verb_token_span = Span(
                        postfix_string_start,
                        postfix_string_start + max_token_length_for_template_postfix,
                    )
                    if is_legal_template_span(postfix_candidate_verb_token_span):
                        final_template_elements = list(template_elements)
                        final_template_elements.extend(
                            sentence_tokens[
                                postfix_candidate_verb_token_span.start : postfix_candidate_verb_token_span.end
                            ]
                        )
                        yield SurfaceTemplateBoundToSemanticNodes(
                            surface_template=SurfaceTemplate(
                                elements=final_template_elements,
                                determiner_prefix_slots=[
                                    SLOT for (SLOT, _) in slot_to_semantic_node
                                ],
                                language_mode=self._language_mode,
                            ),
                            slot_to_semantic_node=slot_to_semantic_node,
                        )
            else:
                yield SurfaceTemplateBoundToSemanticNodes(
                    surface_template=SurfaceTemplate(
                        elements=template_elements,
                        determiner_prefix_slots=[
                            SLOT for (SLOT, _) in slot_to_semantic_node
                        ],
                        language_mode=self._language_mode,
                    ),
                    slot_to_semantic_node=slot_to_semantic_node,
                )

        # Generate all the possible verb template alignments
        for verb_template in candidate_verb_templates():
            for aligned_nodes in aligned_object_nodes(
                sum(1 for token in verb_template if token == VerbAlignmentSlots.Argument),
                num_arguments_to_alignments_sets,
                language_concept_alignment,
            ):
                # aligned_object_nodes is guaranteed to only give us alignments
                # Which the spans go from left most to right most
                for (
                    surface_template_bound_to_semantic_nodes
                ) in process_aligned_objects_with_template(verb_template, aligned_nodes):
                    if surface_template_bound_to_semantic_nodes:
                        ret.append(surface_template_bound_to_semantic_nodes)

        return immutableset(
            bound_surface_template
            for bound_surface_template in ret
            # For now, we require templates to account for the entire utterance.
            # See https://github.com/isi-vista/adam/issues/789
            if covers_entire_utterance(bound_surface_template, language_concept_alignment)
        )


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
