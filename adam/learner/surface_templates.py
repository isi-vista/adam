"""
Representations of template-with-slots-like patterns over token strings.
"""
from typing import Tuple, Union, Mapping, List, Iterable

from more_itertools import quantify

from adam.language import TokenSequenceLinguisticDescription
from adam.learner.alignments import LanguageConceptAlignment
from adam.semantics import ObjectSemanticNode, SyntaxSemanticsVariable
from attr import attrs, attrib
from attr.validators import instance_of, deep_iterable

from immutablecollections import ImmutableSet, immutableset
from immutablecollections.converter_utils import _to_tuple, _to_immutableset


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

    @staticmethod
    def from_language_aligned_perception(
        language_aligned_perception: LanguageConceptAlignment,
        object_node_to_template_variable: Mapping[
            ObjectSemanticNode, SyntaxSemanticsVariable
        ],
        *,
        determiner_prefix_slots: Iterable[SyntaxSemanticsVariable] = immutableset()
    ) -> "SurfaceTemplate":
        if len(object_node_to_template_variable) != len(
            language_aligned_perception.node_to_language_span
        ):
            raise RuntimeError(
                "We currently only allow for the situation "
                "where every matched object corresponds to a template node."
            )

        # This will be used to build the returned SurfaceTemplate.
        # We start from the full surface string...
        template_elements: List[Union[SyntaxSemanticsVariable, str]] = list(
            language_aligned_perception.language
        )

        # and the we will walk through it backwards
        # replacing any spans of text which are aligned to objects with variables.
        # We iterate backwards so we the indices don't get invalidated
        # as we replace parts of the token sequence with template variables.

        object_nodes_sorted_by_reversed_aligned_token_position = tuple(
            reversed(
                sorted(
                    language_aligned_perception.node_to_language_span.keys(),
                    key=lambda match_node: language_aligned_perception.node_to_language_span[
                        match_node
                    ],
                )
            )
        )

        for matched_object_node in object_nodes_sorted_by_reversed_aligned_token_position:
            aligned_token_span = language_aligned_perception.node_to_language_span[
                matched_object_node
            ]
            # If an object is aligned to multiple tokens,
            # we just delete all but the first.
            if len(aligned_token_span) > 1:
                for non_initial_aligned_index in range(
                    # -1 because the end index is exclusive
                    aligned_token_span.end - 1,
                    aligned_token_span.start,
                    -1,
                ):
                    del template_elements[non_initial_aligned_index]
            # Regardless, the first token is replaced by a variable.
            template_elements[
                aligned_token_span.start
            ] = object_node_to_template_variable[matched_object_node]

        return SurfaceTemplate(
            template_elements, determiner_prefix_slots=determiner_prefix_slots
        )

    def instantiate(
        self, template_variable_to_filler: Mapping[SyntaxSemanticsVariable, Tuple[str]]
    ) -> TokenSequenceLinguisticDescription:
        """
        Turns a template into a `TokenSequenceLinguisticDescription` by filling in its variables.
        """
        output_tokens: List[str] = []
        for element in self.elements:
            if isinstance(element, SyntaxSemanticsVariable):
                filler_words = template_variable_to_filler[element]
                # Ground is a specific thing so we special case this to be assigned
                if filler_words[0] == "ground":
                    output_tokens.append("the")
                # English-specific hack to deal with us not understanding determiners:
                # https://github.com/isi-vista/adam/issues/498
                # The "is lower" check is a hack to block adding a determiner to proper names.
                elif (
                    element in self._determiner_prefix_slots
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
