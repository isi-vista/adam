import logging
from itertools import chain
from typing import Iterable, List, Mapping, Optional, Tuple, Union

from more_itertools import pairwise

from adam.language import LinguisticDescription
from adam.learner.surface_templates import SurfaceTemplate
from adam.perception.perception_graph import PerceptionGraph
from adam.semantics import ObjectSemanticNode, SemanticNode, SyntaxSemanticsVariable
from attr import attrib, attrs
from attr.validators import deep_iterable, instance_of
from immutablecollections import ImmutableDict, ImmutableSet, immutabledict, immutableset
from immutablecollections.converter_utils import _to_immutableset
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
class LanguageConceptAlignment:
    """
    Represents an alignment between a `LinguisticDescription` and an `ImmutableSet[SemanticNode]` where
    the nodes represent concepts.

    This can be generified in the future.

    *node_to_language_span* and *language_span_to_node* are both guaranteed to be sorted by
    the token spans.

    Aligned token spans may not overlap.
    """

    language: LinguisticDescription = attrib(validator=instance_of(LinguisticDescription))
    node_to_language_span: ImmutableDict[ObjectSemanticNode, Span] = attrib(
        converter=_sort_mapping_by_token_spans, default=immutabledict()
    )
    language_span_to_node: ImmutableDict[Span, SemanticNode] = attrib(init=False)
    aligned_nodes: ImmutableSet[SemanticNode] = attrib(init=False)
    aligned_token_indices: ImmutableSet[int] = attrib(init=False)
    is_entirely_aligned: bool = attrib(init=False)

    @staticmethod
    def create_unaligned(language: LinguisticDescription) -> "LanguageConceptAlignment":
        return LanguageConceptAlignment(language)

    def copy_with_added_token_alignments(
        self, new_token_alignments: Mapping[ObjectSemanticNode, Span]
    ) -> "LanguageConceptAlignment":
        return LanguageConceptAlignment(
            language=self.language,
            node_to_language_span=chain(
                self.node_to_language_span.items(), new_token_alignments.items()
            ),
        )

    def token_index_is_aligned(self, token_index: int) -> bool:
        return token_index in self.aligned_token_indices

    @language_span_to_node.default
    def _init_language_span_to_node(self) -> ImmutableDict[ObjectSemanticNode, Span]:
        return immutabledict((v, k) for (k, v) in self.node_to_language_span.items())

    @aligned_nodes.default
    def _init_aligned_nodes(self) -> ImmutableSet[ObjectSemanticNode]:
        return immutableset(self.node_to_language_span.keys())

    @aligned_token_indices.default
    def _init_aligned_token_indices(self) -> ImmutableSet[int]:
        token_indices_aligned = []
        for aligned_span in self.node_to_language_span.values():
            for aligned_token_index in range(aligned_span.start, aligned_span.end):
                token_indices_aligned.append(aligned_token_index)
        return immutableset(token_indices_aligned)

    @is_entirely_aligned.default
    def _init_is_entirely_aligned(self) -> bool:
        for token_index in range(len(self.language.as_token_sequence())):
            if token_index not in self.aligned_token_indices:
                return False
        return True

    def __attrs_post_init__(self) -> None:
        # In the converter, we guarantee that node_to_language_span is sorted by
        # token indices.
        for (span1, span2) in pairwise(self.node_to_language_span.values()):
            if not span1.precedes(span2):
                raise RuntimeError(
                    f"Aligned spans in a LanguageAlignedPerception must be "
                    f"disjoint but got {span1} and {span2}"
                )

    def copy_with_new_nodes(
        self,
        new_semantic_nodes_to_surface_templates: Mapping[
            ObjectSemanticNode, SurfaceTemplate
        ],
        *,
        filter_out_duplicate_alignments: bool,
    ) -> "LanguageConceptAlignment":
        """
        Get a new copy of this alignment,
        except with the given given semantic nodes aligned to the associated surface templates.
        """
        # This is what we will use to build the new alignment.
        new_node_to_language_span = list(self.node_to_language_span.items())
        for (
            new_semantic_node,
            surface_template,
        ) in new_semantic_nodes_to_surface_templates.items():
            slots_to_spans: List[Tuple[SyntaxSemanticsVariable, Span]] = []
            for (slot, filler) in new_semantic_node.slot_fillings.items():
                # We need to align the arguments of the new semantic node to tokens.
                filler_tokens = self.node_to_language_span.get(filler)
                if filler_tokens:
                    slots_to_spans.append((slot, filler_tokens))
                else:
                    raise RuntimeError(
                        f"For semantic node {new_semantic_node}, its slot {slot} is "
                        f"occupied by {filler}, but we don't know how to align "
                        f"that filler to tokens. Known alignments are: "
                        f"{self.node_to_language_span}"
                    )

            covered_token_span = surface_template.match_against_tokens(
                self.language.as_token_sequence(),
                slots_to_filler_spans=immutabledict(slots_to_spans),
            )

            if covered_token_span:
                intersecting_alignments = [
                    (span, node)
                    for (span, node) in self.language_span_to_node.items()
                    if covered_token_span.overlaps(span)
                ]
                if intersecting_alignments:
                    message = (
                        f"Ignoring attempt to align tokens "
                        f"{self.language.as_token_string(span=covered_token_span)} "
                        f"to {new_semantic_node} because the following alignments already exist:"
                        f" {intersecting_alignments}"
                    )
                    if filter_out_duplicate_alignments:
                        logging.info(message)
                    else:
                        raise RuntimeError(message)
                else:
                    new_node_to_language_span.append(
                        (new_semantic_node, covered_token_span)
                    )
            else:
                raise RuntimeError(
                    f"Could not match surface template {surface_template} "
                    f"with fillers {slots_to_spans} against "
                    f"{self.language.as_token_sequence()}"
                )

        return LanguageConceptAlignment(
            self.language, node_to_language_span=new_node_to_language_span
        )

    def to_surface_template(
        self,
        object_node_to_template_variable: Mapping[
            ObjectSemanticNode, SyntaxSemanticsVariable
        ],
        *,
        determiner_prefix_slots: Iterable[SyntaxSemanticsVariable] = immutableset(),
        restrict_to_span: Optional[Span] = None,
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
            template_elements, determiner_prefix_slots=determiner_prefix_slots
        )


@attrs(frozen=True)
class PerceptionSemanticAlignment:
    """
    Represents an alignment between a perception graph and a set of semantic nodes representing
    concepts.

    This is used to represent intermediate semantic data passed between new-style learners when
    describing a perception.
    """

    perception_graph: PerceptionGraph = attrib(validator=instance_of(PerceptionGraph))
    semantic_nodes: ImmutableSet[SemanticNode] = attrib(
        converter=_to_immutableset, validator=deep_iterable(instance_of(SemanticNode))
    )

    @staticmethod
    def create_unaligned(
        perception_graph: PerceptionGraph
    ) -> "PerceptionSemanticAlignment":
        return PerceptionSemanticAlignment(perception_graph, [])

    def copy_with_updated_graph_and_added_nodes(
        self, *, new_graph: PerceptionGraph, new_nodes: Iterable[SemanticNode]
    ) -> "PerceptionSemanticAlignment":
        if new_graph is self.perception_graph and not new_nodes:
            return self
        else:
            return PerceptionSemanticAlignment(
                perception_graph=new_graph,
                semantic_nodes=chain(self.semantic_nodes, new_nodes),
            )

    def __attrs_post_init__(self) -> None:
        for node in self.perception_graph._graph:
            if isinstance(node, SemanticNode):
                if not node in self.semantic_nodes:
                    raise RuntimeError(
                        "All semantic nodes appearing in the perception graph must "
                        "also be in semantic_nodes"
                    )


@attrs(frozen=True, kw_only=True)
class LanguagePerceptionSemanticAlignment:
    """
    Represents an alignment of both language and perception with some semantic nodes representing
    concepts.

    This is used to represent intermediate semantic data passed between new-style learners when
    learning from an example.
    """

    language_concept_alignment: LanguageConceptAlignment = attrib(
        validator=instance_of(LanguageConceptAlignment)
    )
    perception_semantic_alignment: PerceptionSemanticAlignment = attrib(
        validator=instance_of(PerceptionSemanticAlignment)
    )
