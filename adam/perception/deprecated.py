from typing import Iterable, List, Mapping, Optional, Union

from more_itertools import pairwise

from adam.language import LinguisticDescription
from adam.learner.language_mode import LanguageMode
from adam.learner.surface_templates import SurfaceTemplate
from adam.perception.perception_graph import PerceptionGraph, PerceptionGraphNode
from adam.semantics import ObjectSemanticNode, SyntaxSemanticsVariable
from attr import attrib, attrs
from attr.validators import instance_of
from immutablecollections import ImmutableDict, ImmutableSet, immutabledict, immutableset
from vistautils.span import Span


# Used by LanguageAlignedPerception below.
def _sort_mapping_by_token_spans(pairs) -> ImmutableDict[ObjectSemanticNode, Span]:
    # we type: ignore because the proper typing of pairs is huge and mypy is going to screw it up
    # anyway.
    unsorted = immutabledict(pairs)  # type: ignore
    return immutabledict(
        (matched_node, token_span)
        for (matched_node, token_span) in sorted(
            unsorted.items(),
            key=lambda item: Span.earliest_then_longest_first_key(item[1]),
        )
    )


@attrs(frozen=True)
class LanguageAlignedPerception:
    """
    Represents an alignment between a `PerceptionGraph` and a `TokensSequenceLinguisticDescription`.
    This can be generified in the future.
    *node_to_language_span* and *language_span_to_node* are both guaranteed to be sorted by
    the token spans.
    Aligned token spans may not overlap.
    """

    language: LinguisticDescription = attrib(validator=instance_of(LinguisticDescription))
    perception_graph: PerceptionGraph = attrib(validator=instance_of(PerceptionGraph))
    node_to_language_span: ImmutableDict[ObjectSemanticNode, Span] = attrib(
        converter=_sort_mapping_by_token_spans, default=immutabledict()
    )
    language_span_to_node: ImmutableDict[Span, PerceptionGraphNode] = attrib(init=False)
    aligned_nodes: ImmutableSet[ObjectSemanticNode] = attrib(init=False)

    @language_span_to_node.default
    def _init_language_span_to_node(self) -> ImmutableDict[PerceptionGraphNode, Span]:
        return immutabledict((v, k) for (k, v) in self.node_to_language_span.items())

    @aligned_nodes.default
    def _init_aligned_nodes(self) -> ImmutableSet[ObjectSemanticNode]:
        return immutableset(self.node_to_language_span.keys())

    def __attrs_post_init__(self) -> None:
        # In the converter, we guarantee that node_to_language_span is sorted by
        # token indices.
        for (span1, span2) in pairwise(self.node_to_language_span.values()):
            if not span1.precedes(span2):
                raise RuntimeError(
                    f"Aligned spans in a LanguageAlignedPerception must be "
                    f"disjoint but got {span1} and {span2}"
                )

    def to_surface_template(
        self,
        object_node_to_template_variable: Mapping[
            ObjectSemanticNode, SyntaxSemanticsVariable
        ],
        *,
        determiner_prefix_slots: Iterable[SyntaxSemanticsVariable] = immutableset(),
        restrict_to_span: Optional[Span] = None,
        language_mode: LanguageMode = LanguageMode.ENGLISH,
    ) -> SurfaceTemplate:
        """
        Creates a `SurfaceTemplate` corresponding to all or some portion of this alignment.

        The user specifies which semantic object nodes should have their aligned tokens
        replaced with wildcard slots in the template using *object_node_to_template_variable*.
        For example, if you have the language "Fred ate a sandwich"
        where "Fred" is aligned to object o_1 and "sandwich" is aligned to object o_2,
        then calling this method with { SLOT1: o_1, SLOT2: o_2} will produce the template
        "SLOT1 ate SLOT2".

        *determiner_prefix_slots* is passed along to the constructed `SurfaceTemplate`
        so it know what nodes should be prefixed with a determiner when the template
        is instantiated in the future.
        This is an English-specific hack tracked in https://github.com/isi-vista/adam/issues/498

        If *restrict_to_span* is specified, the template will be built only from tokens
        of `language` within that span.
        """

        # Restrict our attention to the portion of the tokens the user requested,
        # if they did so
        if restrict_to_span is not None:
            target_span = restrict_to_span
            # If we are restricting to a certain span, we shift all alignment spans to be
            # relative to the restricted span.
            node_to_language_span_restricted: Mapping[
                ObjectSemanticNode, Span
            ] = immutabledict(
                (node, aligned_span.shift(-target_span.start))
                for (node, aligned_span) in self.node_to_language_span.items()
                if aligned_span in target_span
            )
            tokens_in_target_span = tuple(self.language.as_token_sequence())[
                target_span.start : target_span.end
            ]
        else:
            node_to_language_span_restricted = self.node_to_language_span
            tokens_in_target_span = self.language.as_token_sequence()

        num_alignments_entirely_within_target_span = len(node_to_language_span_restricted)
        if (
            len(object_node_to_template_variable)
            != num_alignments_entirely_within_target_span
        ):
            raise RuntimeError(
                "We currently only allow for the situation "
                "where every matched object corresponds to a template node."
            )

        # This will be used to build the returned SurfaceTemplate.
        # We start from the full surface string.
        # Note we use list() to get a mutable copy
        template_elements: List[Union[SyntaxSemanticsVariable, str]] = list(
            tokens_in_target_span
        )

        # and the we will walk through it backwards
        # replacing any spans of text which are aligned to objects with variables.
        # We iterate backwards so we the indices don't get invalidated
        # as we replace parts of the token sequence with template variables.
        object_nodes_sorted_by_reversed_aligned_token_position = tuple(
            reversed(
                sorted(
                    node_to_language_span_restricted.keys(),
                    key=lambda match_node: node_to_language_span_restricted[match_node],
                )
            )
        )

        for matched_object_node in object_nodes_sorted_by_reversed_aligned_token_position:
            aligned_token_span = node_to_language_span_restricted[matched_object_node]
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
            template_elements,
            determiner_prefix_slots=determiner_prefix_slots,
            language_mode=language_mode,
        )
