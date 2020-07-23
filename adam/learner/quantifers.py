from abc import ABC
from collections import defaultdict
from enum import Enum, auto
from math import log
from pathlib import Path
from typing import AbstractSet, Dict, Iterable, List

from more_itertools import only

from adam.learner import LanguageConceptAlignment, LanguageMode, SurfaceTemplate
from adam.learner.surface_templates import SLOT1
from adam.learner.template_learner import SemanticTemplateLearner
from adam.semantics import (
    Concept,
    GROUND_OBJECT_CONCEPT,
    LearnerSemantics,
    NumberConcept,
    ObjectConcept,
)
from attr import Factory, attrib, attrs
from attr.validators import instance_of
from immutablecollections import immutableset, immutablesetmultidict
from vistautils.span import Span

CHINESE_NO_QUANTIFIER_TEMPLATE = SurfaceTemplate(
    [SLOT1], language_mode=LanguageMode.CHINESE
)


class QuantifierTemplateLearner(
    SemanticTemplateLearner, ABC
):  # pylint:disable=abstract-method
    @staticmethod
    def pretrained_for_language_mode(
        language_mode: LanguageMode
    ) -> "QuantifierTemplateLearner":
        """
        Get a `QuantifierTemplateLearner` which already knows the correct number information
        for a language.

        This is useful when you are experimenting with other things and don't want to wait
        for numbers to be learned
        (or the training data for them is not included in your curriculum).
        """
        if language_mode == LanguageMode.ENGLISH:
            return PretrainedEnglishQuantifiers()
        elif language_mode == LanguageMode.CHINESE:
            return PretrainedChineseQuantifiers()
        else:
            raise RuntimeError(f"No pretrained quantifiers available for {language_mode}")


class LinguisticNumber(Enum):
    SINGULAR = auto()
    DUAL = auto()
    PLURAL = auto()


NO_OP_TEMPLATE_ENGLISH = SurfaceTemplate((SLOT1,), language_mode=LanguageMode.ENGLISH)
NO_OP_TEMPLATE_CHINESE = SurfaceTemplate((SLOT1,), language_mode=LanguageMode.CHINESE)


@attrs(frozen=True)
class ToleranceRuleQuantifierTemplateLearner(SemanticTemplateLearner):
    language_mode: LanguageMode = attrib(validator=instance_of(LanguageMode))
    min_types_to_lexicalize: int = attrib(default=5)

    _lexicalized_number_to_template: Dict[LinguisticNumber, SurfaceTemplate] = attrib(
        init=False, default=Factory(dict)
    )
    _number_to_concept_to_hypotheses: Dict[
        LinguisticNumber, Dict[ObjectConcept, List[SurfaceTemplate]]
    ] = attrib(
        init=False, default=Factory(lambda: defaultdict(lambda: defaultdict(list)))
    )
    _number_to_template_to_count: Dict[
        LinguisticNumber, Dict[SurfaceTemplate, int]
    ] = attrib(init=False, default=Factory(lambda: defaultdict(lambda: defaultdict(int))))

    _dont_know_template: SurfaceTemplate = attrib(init=False)
    """
    Template to return if we haven't yet learned how to express a given number.
    """

    @_dont_know_template.default
    def _default_dont_know_template(self) -> SurfaceTemplate:
        if self.language_mode == LanguageMode.ENGLISH:
            return NO_OP_TEMPLATE_ENGLISH
        elif self.language_mode == LanguageMode.CHINESE:
            return NO_OP_TEMPLATE_CHINESE
        else:
            raise RuntimeError(
                f"Don't know how to make a no-op template for {self.language_mode}"
            )

    def learn_from(
        self,
        language_concept_alignment: LanguageConceptAlignment,
        semantics: LearnerSemantics,
    ) -> None:
        # First, if there are multiple token spans aligned to objects of the same type
        # (e.g. the red boxes beside the pink boxes)
        # we bail out and don't try to learn anything at all,
        # because it would violate some assumptions we make below.
        object_types_to_language_span_of_aligned_objects = immutablesetmultidict(
            (object_node.concept, token_span)
            for (
                object_node,
                token_span,
            ) in language_concept_alignment.node_to_language_span.items()
        )

        if any(
            len(spans_aligned_to_objects_of_same_type) > 1
            for spans_aligned_to_objects_of_same_type in object_types_to_language_span_of_aligned_objects.value_groups()
        ):
            return

        # TODO: the below assumes the language marks number obligatorily if it marks it at all.
        # Do all do so?

        number_properties_modified_on_this_update = []

        # Otherwise, we learn for each object_type separately
        # (e.g. for "the red boxes and the green cups" we learn for boxes and cups separately)
        for object_type in object_types_to_language_span_of_aligned_objects:
            # # For English-special case determiner cases, bail out and don't try to learn.
            # if (
            #     self.language_mode == LanguageMode.ENGLISH
            #     and is_english_determiner_special_case(object_type)
            # ):
            #     continue

            # We count how many there are of the relevant object type
            # and bucket them as singular, dual, plural.
            num_objects_of_type = 0
            for object_node in semantics.objects:
                if object_node.concept == object_type:
                    num_objects_of_type += 1

            if num_objects_of_type == 0:
                raise RuntimeError("This should be impossible")
            elif num_objects_of_type == 1:
                number_property = LinguisticNumber.SINGULAR
            elif num_objects_of_type == 2:
                number_property = LinguisticNumber.DUAL
            else:
                number_property = LinguisticNumber.PLURAL

            if number_property in self._lexicalized_number_to_template:
                # We already "lexicalized" how to express this number,
                # so no need to keep trying to learn.
                continue

            object_span = only(
                object_types_to_language_span_of_aligned_objects[object_type]
            )

            # We then hypothesize ways that number is being expressed in this situation
            for surface_template_hypothesis in self._number_expression_hypotheses(
                object_span, language_concept_alignment
            ):
                self._number_to_template_to_count[number_property][
                    surface_template_hypothesis
                ] += 1
                if (
                    surface_template_hypothesis
                    not in self._number_to_concept_to_hypotheses[number_property][
                        object_type
                    ]
                ):
                    self._number_to_concept_to_hypotheses[number_property][
                        object_type
                    ].append(surface_template_hypothesis)
            number_properties_modified_on_this_update.append(number_property)

        # Now check if we can lexicalize any of our number properties.
        for number_property in immutableset(number_properties_modified_on_this_update):
            if number_property not in self._lexicalized_number_to_template:
                concept_to_hypothesis = self._number_to_concept_to_hypotheses[
                    number_property
                ]
                template_to_count = self._number_to_template_to_count[number_property]

                # Loop through templates in descending order of frequency.
                hypotheses_by_decreasing_frequency = [
                    hypothesis
                    for (hypothesis, count) in sorted(
                        template_to_count.items(), key=lambda x: x[1], reverse=True
                    )
                ]
                for hypothesis in hypotheses_by_decreasing_frequency:
                    votes_for = 0
                    votes_against = 0
                    for concept in concept_to_hypothesis:
                        # A concept is a vote "for" a template if it occurs with that template
                        # for the given number.
                        if hypothesis in concept_to_hypothesis[concept]:
                            votes_for += 1
                        # It is a vote "against" the template if it occurs with some *other*
                        # template
                        # for the given number and not this template.
                        elif concept_to_hypothesis[concept]:
                            votes_against += 1
                    # If the number of vote "for" exceeds the number of votes "against"
                    # by the TP threshold, we lexicalize.
                    num_types = votes_for + votes_against
                    if (
                        votes_for > self.min_types_to_lexicalize
                        and votes_against <= num_types / log(num_types)
                    ):
                        self._lexicalized_number_to_template[number_property] = hypothesis
                        break

    def templates_for_concept(self, concept: Concept) -> AbstractSet[SurfaceTemplate]:
        if isinstance(concept, NumberConcept):
            if concept.number == 1:
                linguistic_number = LinguisticNumber.SINGULAR
            elif concept.number == 2:
                linguistic_number = LinguisticNumber.DUAL
            else:
                linguistic_number = LinguisticNumber.PLURAL
        else:
            raise RuntimeError(
                f"Should only be called for number concepts, but got {concept}"
            )

        return immutableset(
            [
                self._lexicalized_number_to_template.get(
                    linguistic_number, self._dont_know_template
                )
            ]
        )

    def _number_expression_hypotheses(
        self, span_for_object: Span, language_concept_alignment: LanguageConceptAlignment
    ) -> Iterable[SurfaceTemplate]:
        ret = []

        object_at_left_edge = span_for_object.start == 0
        token_sequence = language_concept_alignment.language.as_token_sequence()
        object_at_right_edge = span_for_object.end == len(token_sequence)

        # We currently limit ourselves to cases
        # where the object + plural marker is the entire utterance.

        # Account for English determiner hack
        if token_sequence[span_for_object.start] in ("a", "the"):
            ret.append(
                SurfaceTemplate(
                    [token_sequence[span_for_object.start], SLOT1],
                    language_mode=self.language_mode,
                )
            )

        # Any tokens immediately before or after the expression of an object
        # are candidate expressions of pluralization.
        preceding_token_index = span_for_object.start - 1
        if (
            not object_at_left_edge
            and not language_concept_alignment.token_index_is_aligned(
                preceding_token_index
            )
            and object_at_right_edge
        ):
            ret.append(
                SurfaceTemplate(
                    [token_sequence[0], SLOT1], language_mode=self.language_mode
                )
            )
        following_token_index = span_for_object.end + 1
        if (
            not object_at_right_edge
            and not language_concept_alignment.token_index_is_aligned(
                following_token_index
            )
            and object_at_left_edge
        ):
            ret.append(
                SurfaceTemplate(
                    [SLOT1, token_sequence[-1]], language_mode=self.language_mode
                )
            )

        # We will consider a no-op hypothesis only if everything on both sides
        # is aready aligned
        if object_at_left_edge and object_at_right_edge:
            ret.append(SurfaceTemplate([SLOT1], language_mode=self.language_mode))

        return immutableset(ret)


class AbstractPretrainedQuantifiers(
    QuantifierTemplateLearner
):  # pylint:disable=abstract-method
    def learn_from(
        self,
        language_concept_alignment: LanguageConceptAlignment,
        semantics: LearnerSemantics,
    ) -> None:
        # Pretrained - nothing to learn.
        pass

    def log_hypotheses(self, log_output_path: Path) -> None:
        pass


THE_TEMPLATE = SurfaceTemplate(["the", SLOT1], language_mode=LanguageMode.ENGLISH)
A_TEMPLATE = SurfaceTemplate(["a", SLOT1], language_mode=LanguageMode.ENGLISH)
PLURAL_TEMPLATE = SurfaceTemplate([SLOT1, "s"], language_mode=LanguageMode.ENGLISH)
TWO_TEMPLATE = SurfaceTemplate(["two", SLOT1, "s"], language_mode=LanguageMode.ENGLISH)
MANY_TEMPLATE = SurfaceTemplate(["many", SLOT1, "s"], language_mode=LanguageMode.ENGLISH)
ENGLISH_NO_QUANTIFIER_TEMPLATE = SurfaceTemplate(
    [SLOT1], language_mode=LanguageMode.ENGLISH
)


class PretrainedEnglishQuantifiers(AbstractPretrainedQuantifiers):
    def templates_for_concept(self, concept: Concept) -> AbstractSet[SurfaceTemplate]:
        if not isinstance(concept, NumberConcept):
            raise RuntimeError(
                f"Quantification learner only understands NumberConcepts, but got "
                f"{concept}"
            )

        number = concept.number

        ret = []
        # Ground is a specific thing so we special case this to be assigned
        if concept == GROUND_OBJECT_CONCEPT:
            ret.append(THE_TEMPLATE)
        elif number == 1:
            ret.append(A_TEMPLATE)
        else:
            ret.append(PLURAL_TEMPLATE)
            if number == 2:
                ret.append(TWO_TEMPLATE)
            elif number > 2:
                ret.append(MANY_TEMPLATE)

        if not ret:
            ret.append(ENGLISH_NO_QUANTIFIER_TEMPLATE)

        return immutableset(ret)


class PretrainedChineseQuantifiers(AbstractPretrainedQuantifiers):
    def templates_for_concept(self, concept: Concept) -> AbstractSet[SurfaceTemplate]:
        # TODO: this is wrong, but classifiers make things complicated in Chinese
        return immutableset([CHINESE_NO_QUANTIFIER_TEMPLATE])
