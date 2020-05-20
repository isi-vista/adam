from itertools import chain
from typing import Iterable, Mapping

from more_itertools import pairwise

from adam.language import LinguisticDescription
from adam.perception.perception_graph import PerceptionGraph
from adam.semantics import ObjectSemanticNode, SemanticNode
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
    Represents an alignment between a `PerceptionGraph` and a `TokensSequenceLinguisticDescription`.

    This can be generified in the future.

    *node_to_language_span* and *language_span_to_node* are both guaranteed to be sorted by
    the token spans.

    Aligned token spans may not overlap.
    """

    language: LinguisticDescription = attrib(validator=instance_of(LinguisticDescription))
    node_to_language_span: ImmutableDict[ObjectSemanticNode, Span] = attrib(
        converter=_sort_mapping_by_token_spans, default=immutabledict()
    )
    language_span_to_node: ImmutableDict[Span, ObjectSemanticNode] = attrib(init=False)
    aligned_nodes: ImmutableSet[ObjectSemanticNode] = attrib(init=False)

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

    @language_span_to_node.default
    def _init_language_span_to_node(self) -> ImmutableDict[ObjectSemanticNode, Span]:
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


@attrs(frozen=True)
class PerceptionSemanticAlignment:
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
    language_concept_alignment: LanguageConceptAlignment = attrib(
        validator=instance_of(LanguageConceptAlignment)
    )
    perception_semantic_alignment: PerceptionSemanticAlignment = attrib(
        validator=instance_of(PerceptionSemanticAlignment)
    )
