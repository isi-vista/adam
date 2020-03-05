from typing import Tuple, Union, Mapping, List

from adam.learner.surface_templates import SurfaceTemplateVariable
from adam.perception.perception_graph import LanguageAlignedPerception, MatchedObjectNode
from attr import attrs, attrib
from attr.validators import instance_of, deep_iterable

from immutablecollections.converter_utils import _to_tuple


@attrs(frozen=True, slots=True)
class SurfaceTemplate:
    r"""
    A pattern over `TokenSequenceLinguisticDescription`\ s.

    Such a pattern consists of a sequence of token strings and `SurfaceTemplateVariable`\ s.
    """
    elements: Tuple[Union[str, SurfaceTemplateVariable]] = attrib(  # type: ignore
        converter=_to_tuple,
        validator=deep_iterable(instance_of((str, SurfaceTemplateVariable))),
    )

    @staticmethod
    def from_language_aligned_perception(
        language_aligned_perception: LanguageAlignedPerception,
        object_node_to_template_variable: Mapping[
            MatchedObjectNode, SurfaceTemplateVariable
        ],
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
        template_elements: List[Union[SurfaceTemplateVariable, str]] = list(
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
                    stop=aligned_token_span.start,
                    step=-1,
                ):
                    del template_elements[non_initial_aligned_index]
            # Regardless, the first token is replaced by a variable.
            template_elements[
                aligned_token_span.start
            ] = object_node_to_template_variable[matched_object_node]

        return SurfaceTemplate(template_elements)
