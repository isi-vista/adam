import itertools
from abc import ABC
from enum import Enum
from typing import AbstractSet, Mapping, Union, List, Iterable, Optional, Dict, Tuple
from pathlib import Path

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
    SLOT1,
    SLOT2,
    SLOT3,
    STANDARD_SLOT_VARIABLES,
    SurfaceTemplate,
    SurfaceTemplateBoundToSemanticNodes,
)
from adam.learner.template_learner import (
    AbstractTemplateLearner,
    AbstractTemplateLearnerNew,
)
from adam.perception import PerceptualRepresentation
from adam.perception.deprecated import LanguageAlignedPerception
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception.perception_graph import PerceptionGraph
from adam.semantics import ActionConcept, ObjectSemanticNode, SyntaxSemanticsVariable
from attr import attrib, attrs
from immutablecollections import immutabledict, immutableset, ImmutableSet
from attr.validators import instance_of
from vistautils.span import Span

# This is the maximum number of tokens we will hypothesize
# as the non-argument-slots portion of a surface template for an action.
_MAXIMUM_ACTION_TEMPLATE_TOKEN_LENGTH = 3
_LEFT = -1
_RIGHT = 1


class VerbAlignmentSlots(Enum):
    Argument = "Argument"
    FixedString = "FixedString"


@attrs(frozen=True)
class SemanticNodeWithSpan:
    node: SemanticNode = attrib(validator=instance_of(SemanticNode))
    span: Span = attrib(validator=instance_of(Span))


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
        num_arguments_to_alignments_sets: Dict[
            int, ImmutableSet[List[SemanticNodeWithSpan]]
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

        def candidate_verb_templates() -> Iterable[List[VerbAlignmentSlots]]:
            # First let's handle only one argument - Intransitive Verbs
            yield itertools.combinations(
                [VerbAlignmentSlots.Argument, VerbAlignmentSlots.FixedString], 2
            )
            # Now we want to handle two arguments - Ditransitive Verbs
            yield itertools.combinations(
                [
                    VerbAlignmentSlots.Argument,
                    VerbAlignmentSlots.Argument,
                    VerbAlignmentSlots.FixedString,
                ],
                3,
            )
            # Now we want to handle three arguments, which can either have one or two fixed strings
            yield itertools.combinations(
                [
                    VerbAlignmentSlots.Argument,
                    VerbAlignmentSlots.Argument,
                    VerbAlignmentSlots.Argument,
                    VerbAlignmentSlots.FixedString,
                    VerbAlignmentSlots.FixedString,
                ],
                5,
            )

        def is_valid_aligned_object_nodes(
            semantic_nodes: Tuple[SemanticNodeWithSpan]
        ) -> bool:
            previous_node = semantic_nodes[0]
            for i in range(1, len(semantic_nodes)):
                if not previous_node.span.precedes(semantic_nodes[i].span):
                    return False
                previous_node = semantic_nodes[i]
            return True

        def aligned_object_nodes(
            num_arguments: int
        ) -> Iterable[Tuple[SemanticNodeWithSpan]]:
            # We guarantee the return order is in order of sentence appearance from left to right
            if num_arguments not in num_arguments_to_alignments_sets.keys():
                semantic_nodes_with_spans = immutableset(
                    SemanticNodeWithSpan(node=node, span=span)
                    for (
                        node,
                        span,
                    ) in language_concept_alignment.node_to_language_span.items()
                )

                num_arguments_to_alignments_sets[num_arguments] = immutableset(
                    ordered_semantic_nodes
                    for ordered_semantic_nodes in itertools.product(
                        semantic_nodes_with_spans, repeat=num_arguments
                    )
                    if is_valid_aligned_object_nodes(ordered_semantic_nodes)
                )

            return num_arguments_to_alignments_sets[num_arguments]

        def process_aligned_objects_with_template(
            verb_template: List[VerbAlignmentSlots],
            aligned_nodes: Tuple[SemanticNodeWithSpan],
        ) -> Iterable[Optional[SurfaceTemplateBoundToSemanticNodes]]:
            aligned_node_index = 0
            template_elements = []
            slot_to_semantic_node = []

            for token in verb_template:
                if token == VerbAlignmentSlots.Argument:
                    slot_semantic_variable = STANDARD_SLOT_VARIABLES[aligned_node_index]
                    template_elements.append(slot_semantic_variable)
                    slot_to_semantic_node.append(
                        (slot_semantic_variable, aligned_nodes[aligned_node_index].node)
                    )
                    aligned_node_index += 1
                else:
                    # We need to handle searching before or after the aligned token
                    # And we could generate multiple options of different lengths
                    # between 1 and _MAXIMUM_ACTION_TEMPLATE_TOKEN_LENGTH
                    if aligned_node_index == 0:
                        raise NotImplementedError()
                    elif aligned_node_index == len(aligned_nodes):
                        raise NotImplementedError()
                    else:
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

            yield SurfaceTemplateBoundToSemanticNodes(
                surface_template=SurfaceTemplate(
                    elements=template_elements,
                    determiner_prefix_slots=[SLOT for (SLOT, _) in slot_to_semantic_node],
                ),
                slot_to_semantic_node=slot_to_semantic_node,
            )

        # Generate all the possible verb template alignments
        for verb_template in candidate_verb_templates():
            for aligned_nodes in aligned_object_nodes(
                sum(1 for token in verb_template if token == VerbAlignmentSlots.Argument)
            ):
                # aligned_object_nodes is guaranteed to only give us alignments
                # Which the spans go from left most to right most
                for (
                    surface_template_bound_to_semantic_nodes
                ) in process_aligned_objects_with_template(verb_template, aligned_nodes):
                    if surface_template_bound_to_semantic_nodes:
                        ret.append(surface_template_bound_to_semantic_nodes)

        # First, we handle intransitive verbs.
        # Let's take the example "Bob falls over".
        # For these, we need one recognized object, e.g. "Bob"
        for (
            object_node,
            span_for_object,
        ) in language_concept_alignment.node_to_language_span.items():
            # Two possible hypotheses: the subject follows the verb
            # or the verb follows the subject.
            # In our example, only the "right" direction ends up being interesting.
            for direction_to_look_for_verb in (_LEFT, _RIGHT):
                # The template for the verb itself could be multiple tokens
                # (e.g. "falls over").
                # We are going to run through the loop below twice, once for a verb template
                # size of 1 ("falls"),
                # and once for a verb template size of 2 ("falls over").
                for max_token_length_for_template in range(
                    1, _MAXIMUM_ACTION_TEMPLATE_TOKEN_LENGTH + 1
                ):
                    # First, determine the tokens in the candidate template.
                    if direction_to_look_for_verb == _LEFT:
                        candidate_verb_token_span = Span(
                            span_for_object.start - max_token_length_for_template,
                            span_for_object.start,
                        )
                    else:
                        candidate_verb_token_span = Span(
                            span_for_object.end,
                            span_for_object.end + max_token_length_for_template,
                        )

                    if is_legal_template_span(candidate_verb_token_span):
                        template_elements: List[
                            Union[str, SyntaxSemanticsVariable]
                        ] = list(
                            sentence_tokens[
                                candidate_verb_token_span.start : candidate_verb_token_span.end
                            ]
                        )
                        if direction_to_look_for_verb == _LEFT:
                            # Subject is on the right
                            template_elements.append(SLOT1)
                        else:
                            # Subject is on the left
                            template_elements.insert(0, SLOT1)

                        ret.append(
                            SurfaceTemplateBoundToSemanticNodes(
                                surface_template=SurfaceTemplate(
                                    elements=template_elements,
                                    determiner_prefix_slots=[SLOT1],
                                ),
                                slot_to_semantic_node=[(SLOT1, object_node)],
                            )
                        )

        # Handle transitive verbs.
        # Our example here will be "Mom eats the cookie."
        # For a transitive verb, we need at least *two* recognized objects.
        for (
            left_object_node,
            span_for_left_object,
        ) in language_concept_alignment.node_to_language_span.items():
            for (
                right_object_node,
                span_for_right_object,
            ) in language_concept_alignment.node_to_language_span.items():
                # Our code will be simpler if we can assume an ordering of the object aligned
                # tokens.
                if not span_for_left_object.precedes(span_for_right_object):
                    continue

                # We want to handle following verb syntaxes:
                # SOV, SVO, VSO, VOS, OVS, OSV
                # However, currently our templates don't distinguish subject and object,
                # just "leftmost slot" (=SLOT1) and "rightmost slot" (=SLOT2),
                # so we really just need to handle (using "A" for argument):
                # AAV, AVA, VAA

                # Let's handle AVA first.
                # We can only do this if the two candidate argument spans are not adjacent
                # - otherwise there is nothing in-between to use as a verb template!
                if span_for_left_object.end != span_for_right_object.start:
                    candidate_verb_token_span = Span(
                        span_for_left_object.end, span_for_right_object.start
                    )

                    if is_legal_template_span(candidate_verb_token_span):
                        template_elements = [SLOT1]
                        template_elements.extend(
                            sentence_tokens[
                                candidate_verb_token_span.start : candidate_verb_token_span.end
                            ]
                        )
                        template_elements.append(SLOT2)
                        ret.append(
                            SurfaceTemplateBoundToSemanticNodes(
                                surface_template=SurfaceTemplate(
                                    elements=template_elements,
                                    determiner_prefix_slots=[SLOT1, SLOT2],
                                ),
                                slot_to_semantic_node=[
                                    (SLOT1, left_object_node),
                                    (SLOT2, right_object_node),
                                ],
                            )
                        )

                # Handle AAV and VAA cases.
                # e.g. "Mom cookie eats" or "Eats Mom cookie".
                for verb_direction in (_LEFT, _RIGHT):
                    # We can only handle these cases if the two argument slot alignments
                    # are adjacent
                    for token_length_for_template in range(
                        1, _MAXIMUM_ACTION_TEMPLATE_TOKEN_LENGTH
                    ):
                        if verb_direction == _LEFT:
                            candidate_verb_token_span = Span(
                                span_for_left_object.start - token_length_for_template,
                                span_for_left_object.start,
                            )
                        else:
                            candidate_verb_token_span = Span(
                                span_for_right_object.end + 1,
                                span_for_right_object.end + token_length_for_template + 1,
                            )

                        if is_legal_template_span(candidate_verb_token_span):
                            template_elements = []
                            if verb_direction == _RIGHT:
                                template_elements.extend([SLOT1, SLOT2])
                            template_elements.extend(
                                sentence_tokens[
                                    candidate_verb_token_span.start : candidate_verb_token_span.end
                                ]
                            )
                            if verb_direction == _LEFT:
                                template_elements.extend([SLOT1, SLOT2])
                            ret.append(
                                SurfaceTemplateBoundToSemanticNodes(
                                    surface_template=SurfaceTemplate(
                                        elements=template_elements,
                                        determiner_prefix_slots=[SLOT1, SLOT2],
                                    ),
                                    slot_to_semantic_node=[
                                        (SLOT1, left_object_node),
                                        (SLOT2, right_object_node),
                                    ],
                                )
                            )

        # Handle transitive verbs with an expressed preposition
        # We don't currently learn the relations separate from the verbs
        # and appropriately combine the syntax later so this helps
        # Us learn for example "Mom throws me a ball"
        # This currently can't handle prepositional tokens
        # in the language (e.g. "Mom throws a ball to me"
        # TODO: https://github.com/isi-vista/adam/issues/787
        for (
            left_object_node,
            span_for_left_object,
        ) in language_concept_alignment.node_to_language_span.items():
            for (
                middle_object_node,
                span_for_middle_object,
            ) in language_concept_alignment.node_to_language_span.items():
                # Our code will be simpler if we can assume an ordering of the object aligned
                # tokens.
                if span_for_left_object.precedes(span_for_middle_object):
                    for (
                        right_object_node,
                        span_for_right_object,
                    ) in language_concept_alignment.node_to_language_span.items():
                        # Our code will be simpler if we can assume an ordering
                        # of the object aligned tokens.
                        if not span_for_middle_object.precedes(span_for_right_object):
                            continue

                        # Our templates don't distinguish between the roles of the nouns
                        # just "leftmost slot" (=SLOT1), "middle slot" (=SLOT2), "rightmost slot" (=SLOT3),
                        # so we really just need to handle (using "A" for argument)
                        # AAAV, AAVA, AVAA, VAAA
                        # This currently doesn't account for learning prepositional information on a verb

                        # Let's handle AVAA and AAVA first
                        # We can only do this if the three candidate argument spans are not adjacent
                        if (
                            span_for_left_object.end != span_for_middle_object.start
                            and span_for_middle_object.end != span_for_right_object.start
                        ):
                            # Now we need to determine between which arguments the verb template is.
                            if span_for_left_object.end == span_for_middle_object.start:
                                candidate_verb_token_span = Span(
                                    span_for_middle_object.end,
                                    span_for_right_object.start,
                                )
                                middle_prior = True
                            else:
                                candidate_verb_token_span = Span(
                                    span_for_left_object.end, span_for_middle_object.start
                                )
                                middle_prior = False
                            if is_legal_template_span(candidate_verb_token_span):
                                template_elements = [SLOT1]
                                if middle_prior:
                                    template_elements.append(SLOT2)
                                template_elements.extend(
                                    sentence_tokens[
                                        candidate_verb_token_span.start : candidate_verb_token_span.end
                                    ]
                                )
                                if not middle_prior:
                                    template_elements.append(SLOT2)
                                template_elements.append(SLOT3)

                                ret.append(
                                    SurfaceTemplateBoundToSemanticNodes(
                                        surface_template=SurfaceTemplate(
                                            elements=template_elements,
                                            determiner_prefix_slots=[SLOT1, SLOT2, SLOT3],
                                        ),
                                        slot_to_semantic_node=[
                                            (SLOT1, left_object_node),
                                            (SLOT2, middle_object_node),
                                            (SLOT3, right_object_node),
                                        ],
                                    )
                                )

                        # Now AAAV and VAAA
                        for verb_direction in (_LEFT, _RIGHT):
                            # We can handle these cases if the three argument slot alignments
                            # are adjacent
                            for token_length_for_template in range(
                                1, _MAXIMUM_ACTION_TEMPLATE_TOKEN_LENGTH
                            ):
                                if verb_direction == _LEFT:
                                    candidate_verb_token_span = Span(
                                        span_for_left_object.start
                                        - token_length_for_template,
                                        span_for_left_object.start,
                                    )
                                else:
                                    candidate_verb_token_span = Span(
                                        span_for_right_object.end + 1,
                                        span_for_right_object.end
                                        + token_length_for_template
                                        + 1,
                                    )

                                if is_legal_template_span(candidate_verb_token_span):
                                    template_elements = []
                                    if verb_direction == _RIGHT:
                                        template_elements.extend([SLOT1, SLOT2, SLOT3])
                                    template_elements.extend(
                                        sentence_tokens[
                                            candidate_verb_token_span.start : candidate_verb_token_span.end
                                        ]
                                    )
                                    if verb_direction == _LEFT:
                                        template_elements.extend([SLOT1, SLOT2, SLOT3])
                                    ret.append(
                                        SurfaceTemplateBoundToSemanticNodes(
                                            surface_template=SurfaceTemplate(
                                                elements=template_elements,
                                                determiner_prefix_slots=[
                                                    SLOT1,
                                                    SLOT2,
                                                    SLOT3,
                                                ],
                                            ),
                                            slot_to_semantic_node=[
                                                (SLOT1, left_object_node),
                                                (SLOT2, middle_object_node),
                                                (SLOT3, right_object_node),
                                            ],
                                        )
                                    )

        def covers_entire_utterance(
            bound_surface_template: SurfaceTemplateBoundToSemanticNodes
        ) -> bool:
            num_covered_tokens = 0
            for element in bound_surface_template.surface_template.elements:
                if isinstance(element, str):
                    num_covered_tokens += 1
                else:
                    num_covered_tokens += len(
                        language_concept_alignment.node_to_language_span[
                            bound_surface_template.slot_to_semantic_node[element]
                        ]
                    )
            # This assumes the slots and the non-slot elements are non-overlapping,
            # which is true for how we construct them.
            return num_covered_tokens == len(sentence_tokens)

        return immutableset(
            bound_surface_template
            for bound_surface_template in ret
            # For now, we require action templates to account for the entire
            # utterance.
            # See https://github.com/isi-vista/adam/issues/789
            if covers_entire_utterance(bound_surface_template)
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
        self, language_concept_alignment: LanguageAlignedPerception
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
