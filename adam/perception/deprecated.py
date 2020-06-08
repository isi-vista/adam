from adam.semantics import ObjectSemanticNode
from attr.validators import instance_of
from more_itertools import pairwise

from adam.language import LinguisticDescription
from adam.perception.perception_graph import PerceptionGraph, PerceptionGraphNode
from attr import attrib, attrs
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
