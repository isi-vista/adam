"""
Representations of template-with-slots-like patterns over token strings.
"""
from typing import List, Mapping, Optional, Tuple, Union
from adam.language_specific.english.english_language_generator import (
    GAILA_PHASE_1_LANGUAGE_GENERATOR,
)
from adam.language.language_generator import LanguageGenerator
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.language.dependency import LinearizedDependencyTree
from more_itertools import quantify

from adam.language import TokenSequenceLinguisticDescription
from adam.semantics import ObjectSemanticNode, SyntaxSemanticsVariable
from attr import attrib, attrs
from attr.validators import deep_iterable, instance_of
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

    Such a pattern consists of a sequence of token strings and `SurfaceTemplateVariable`\ s.
    """
    elements: Tuple[Union[str, SyntaxSemanticsVariable], ...] = attrib(  # type: ignore
        converter=_to_tuple,
        validator=deep_iterable(instance_of((str, SyntaxSemanticsVariable))),
    )
    _determiner_prefix_slots: ImmutableSet[SyntaxSemanticsVariable] = attrib(
        converter=_to_immutableset,
        validator=deep_iterable(instance_of(SyntaxSemanticsVariable)),
        default=immutableset(),
    )
    num_slots: int = attrib(init=False)

    def instantiate(
        self,
        template_variable_to_filler: Mapping[SyntaxSemanticsVariable, Tuple[str, ...]],
        language_generator: LanguageGenerator[
            HighLevelSemanticsSituation, LinearizedDependencyTree
        ] = GAILA_PHASE_1_LANGUAGE_GENERATOR,
    ) -> TokenSequenceLinguisticDescription:
        """
        Turns a template into a `TokenSequenceLinguisticDescription` by filling in its variables.
        """

        output_tokens: List[str] = []
        for element in self.elements:
            if isinstance(element, SyntaxSemanticsVariable):
                filler_words = template_variable_to_filler[element]
                # Ground is a specific thing so we special case this to be assigned
                if (
                    filler_words[0] == "ground"
                    and language_generator == GAILA_PHASE_1_LANGUAGE_GENERATOR
                ):
                    output_tokens.append("the")
                # English-specific hack to deal with us not understanding determiners:
                # https://github.com/isi-vista/adam/issues/498
                # The "is lower" check is a hack to block adding a determiner to proper names.
                elif (
                    language_generator == GAILA_PHASE_1_LANGUAGE_GENERATOR
                    and element in self._determiner_prefix_slots
                    and len(filler_words) == 1
                    and filler_words[0][0].islower()
                    and filler_words[0] not in MASS_NOUNS
                ):
                    output_tokens.append("a")
                output_tokens.extend(filler_words)
            else:
                # element must be a single token str due to object validity checks.
                output_tokens.append(element)
        return TokenSequenceLinguisticDescription(output_tokens)

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
    def for_object_name(object_name: str) -> "SurfaceTemplate":
        return SurfaceTemplate(elements=(object_name,), determiner_prefix_slots=[])

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
                tokens_to_match.append(element)
            else:
                slot_filler_span = slots_to_filler_spans.get(element)
                if slot_filler_span:
                    # endpoints are exclusive
                    tokens_to_match.extend(
                        token_sequence_to_match_against[
                            slot_filler_span.start : slot_filler_span.end
                        ]
                    )
                else:
                    raise RuntimeError(
                        f"Template contained variable {element}, "
                        f"but it was not found in the mapping of slots to spans: "
                        f"{slots_to_filler_spans}"
                    )

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
    ] = attrib(converter=_to_immutabledict)


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
