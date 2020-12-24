"""
Representations of template-with-slots-like patterns over token strings.
"""
from typing import List, Mapping, Optional, Tuple, Union
from more_itertools import quantify

from adam.language import TokenSequenceLinguisticDescription
from adam.language_specific.english import ENGLISH_DETERMINERS
from adam.learner.language_mode import LanguageMode
from adam.semantics import ObjectSemanticNode, SyntaxSemanticsVariable
from attr import attrib, attrs
from attr.validators import deep_iterable, instance_of, deep_mapping
from immutablecollections import ImmutableDict, ImmutableSet, immutableset
from immutablecollections.converter_utils import (
    _to_immutabledict,
    _to_immutableset,
    _to_tuple,
)
from vistautils.span import Span


@attrs(frozen=True, slots=True)
class SurfaceTemplate:
    r"""
    A pattern over `TokenSequenceLinguisticDescription`\ s.

    Such a pattern consists of a sequence of token strings and `SyntaxSemanticsVariable`\ s.
    """
    elements: Tuple[Union[str, SyntaxSemanticsVariable], ...] = attrib(  # type: ignore
        converter=_to_tuple,
        validator=deep_iterable(instance_of((str, SyntaxSemanticsVariable))),
    )
    _language_mode: LanguageMode = attrib(validator=instance_of(LanguageMode))
    _determiner_prefix_slots: ImmutableSet[SyntaxSemanticsVariable] = attrib(
        converter=_to_immutableset,
        validator=deep_iterable(instance_of(SyntaxSemanticsVariable)),
        default=immutableset(),
    )
    num_slots: int = attrib(init=False)

    def instantiate(
        self,
        template_variable_to_filler: Mapping[SyntaxSemanticsVariable, Tuple[str, ...]],
        *,
        attribute_template: bool = False,
    ) -> TokenSequenceLinguisticDescription:

        """
        Turns a template into a `TokenSequenceLinguisticDescription` by filling in its variables.
        """

        output_tokens: List[Union[str, SyntaxSemanticsVariable]] = []
        for element in self.elements:
            if isinstance(element, SyntaxSemanticsVariable):
                filler_words = template_variable_to_filler[element]
                # Ground is a specific thing so we special case this to be assigned
                # However, we don't want to generate language like "the red the ground,"
                # so we don't use this special case for attribute templates.
                if filler_words[0] == "ground" and not attribute_template:
                    output_tokens.append("the")
                # English-specific hack to deal with us not understanding determiners:
                # https://github.com/isi-vista/adam/issues/498
                # The "is lower" check is a hack to block adding a determiner to proper names.
                elif (
                    self._language_mode == LanguageMode.ENGLISH
                    and element in self._determiner_prefix_slots
                    and len(filler_words) == 1
                    and filler_words[0][0].islower()
                    and filler_words[0] not in MASS_NOUNS
                ):
                    output_tokens.append("a")
                elif (
                    self._language_mode == LanguageMode.CHINESE
                    and len(filler_words) == 1
                    and element in self._determiner_prefix_slots
                ):
                    # casing on classifiers in Chinese -- this is a hack that's basically the same as the English one
                    if filler_words[0] in ["chwang2", "jr3", "jwo1 dz"]:
                        output_tokens.append("yi1_jang1")
                    elif filler_words[0] in ["shu1"]:
                        output_tokens.append("yi1_ben3")
                    elif filler_words[0] in ["wu1"]:
                        output_tokens.append("yi1_jyan1")
                    elif filler_words[0] in ["chi4 che1", "ka3 che1"]:
                        output_tokens.append("yi1_lyang4")
                    elif filler_words[0] in ["yi3 dz"]:
                        output_tokens.append("yi1_ba3")
                    elif filler_words[0] in ["shou3", "gou3", "mau1", "nyau3", "syung2"]:
                        output_tokens.append("yi1_jr1")
                    elif filler_words[0] in ["men2"]:
                        output_tokens.append("yi1_shan4")
                    elif filler_words[0] in ["mau4 dz"]:
                        output_tokens.append("yi1_ding3")
                    elif filler_words[0] in ["chyu1 chi2 bing3", "niu2 rou1"]:
                        output_tokens.append("yi1_kwai4")
                    elif filler_words[0] in ["niu2"]:
                        output_tokens.append("yi1_tiao2")
                    elif filler_words[0] in ["ji1"]:
                        output_tokens.append("yi1_zhi1")
                    # eliminate mass and proper nouns and use the default classifier if another one hasn't already been used
                    elif filler_words[0] not in [
                        "ba4 ba4",
                        "ma1 ma1",
                        "shwei3",
                        "gwo3 jr1",
                        "nyou2 nai3",
                        "di4 myan4",
                    ]:
                        output_tokens.append("yi1_ge4")
                output_tokens.extend(filler_words)
            else:
                # element must be a single token str due to object validity checks.
                output_tokens.append(element)
        return TokenSequenceLinguisticDescription(tuple(output_tokens))

    def to_short_string(self) -> str:
        return "_".join(
            element.name if isinstance(element, SyntaxSemanticsVariable) else element
            for element in self.elements
        )

    @num_slots.default
    def _init_num_slots(self) -> int:
        return quantify(
            element
            for element in self.elements
            if isinstance(element, SyntaxSemanticsVariable)
        )

    @staticmethod
    def for_object_name(
        object_name: str, *, language_mode: LanguageMode
    ) -> "SurfaceTemplate":
        return SurfaceTemplate(
            elements=(object_name,),
            determiner_prefix_slots=[],
            language_mode=language_mode,
        )

    def match_against_tokens(
        self,
        token_sequence_to_match_against: Tuple[str, ...],
        *,
        slots_to_filler_spans: Mapping[SyntaxSemanticsVariable, Span],
    ) -> Optional[Span]:
        """
        Gets the token indices, if any, for the first match of this template against
        *token_sequence_to_match_against*, assuming any slots are filled by the tokens given by
        *slots_to_fillers_spans*.
        """
        # First, we turn the template into a token sequence to search for
        # by filling in all the slots form the provided token span mapping.
        tokens_to_match = []
        for element in self.elements:
            if isinstance(element, str):
                # Hack to handle determiners.
                #
                # This may not handle Chinese properly; see
                # https://github.com/isi-vista/adam/issues/993
                try:
                    index = token_sequence_to_match_against.index(element)
                    if (
                        index - 1 >= 0
                        and token_sequence_to_match_against[index - 1]
                        in ENGLISH_DETERMINERS
                    ):
                        tokens_to_match.append(token_sequence_to_match_against[index - 1])
                except ValueError:
                    pass
                finally:
                    tokens_to_match.append(element)
            else:
                slot_filler_span = slots_to_filler_spans.get(element)
                if slot_filler_span:
                    # endpoints are exclusive
                    start = slot_filler_span.start
                    # Hack to handle determiners
                    #
                    # This may not handle Chinese properly; see
                    # https://github.com/isi-vista/adam/issues/993
                    if (
                        slot_filler_span.start - 1 >= 0
                        and token_sequence_to_match_against[slot_filler_span.start - 1]
                        in ENGLISH_DETERMINERS
                    ):
                        start -= 1
                    tokens_to_match.extend(
                        token_sequence_to_match_against[start : slot_filler_span.end]
                    )
                # If template contains an element not found in the mapping of slots to spans, we can return empty here.
                # We don't want to do this now because of generics.
                # else:
                #   return None

        # Now we need to check if the tokens to match occur in the given token sequence to
        # match against.  We don't expect these sequences to be long, so an inefficient solution
        # is okay.
        if not tokens_to_match:
            raise RuntimeError("Don't know how to match any empty token sequence")
        next_idx_to_search_from = 0
        while next_idx_to_search_from < len(token_sequence_to_match_against):
            try:
                index_of_first_token = token_sequence_to_match_against.index(
                    tokens_to_match[0], next_idx_to_search_from
                )
                candidate_match_exclusive_end = index_of_first_token + len(
                    tokens_to_match
                )
                if candidate_match_exclusive_end <= len(token_sequence_to_match_against):
                    if tokens_to_match == list(
                        token_sequence_to_match_against[
                            index_of_first_token:candidate_match_exclusive_end
                        ]
                    ):
                        # span endpoints are exclusive
                        return Span(index_of_first_token, candidate_match_exclusive_end)
                # False alarm - the first token matched, but not the whole sequence.
                next_idx_to_search_from = index_of_first_token + 1
            except ValueError:
                # If we can't even find the first token of what we are searching for,
                # we definitely have no match.
                return None
        # We got all the way to the end without finding a match
        return None


@attrs(frozen=True)
class SurfaceTemplateBoundToSemanticNodes:
    """
    A surface template together with a mapping from its slots to particular semantic roles.

    This is used to specify what the thing we are trying to learn the meaning of in
    a template learner is.  For example, "what does 'X eats Y' mean, given that
    we know X is this thing and Y is that other thing in this particular situation.
    """

    surface_template: SurfaceTemplate = attrib(validator=instance_of(SurfaceTemplate))
    slot_to_semantic_node: ImmutableDict[
        SyntaxSemanticsVariable, ObjectSemanticNode
    ] = attrib(
        converter=_to_immutabledict,
        validator=deep_mapping(
            instance_of(SyntaxSemanticsVariable), instance_of(ObjectSemanticNode)
        ),
    )


SLOT1 = SyntaxSemanticsVariable("slot1")
SLOT2 = SyntaxSemanticsVariable("slot2")
SLOT3 = SyntaxSemanticsVariable("slot3")
SLOT4 = SyntaxSemanticsVariable("slot4")
SLOT5 = SyntaxSemanticsVariable("slot5")
SLOT6 = SyntaxSemanticsVariable("slot6")

STANDARD_SLOT_VARIABLES = (SLOT1, SLOT2, SLOT3, SLOT4, SLOT5, SLOT6)

# These nouns are hard-coded not to receive determiners
# See https://github.com/isi-vista/adam/issues/498
MASS_NOUNS = ["juice", "water", "milk"]
