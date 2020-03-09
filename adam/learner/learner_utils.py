from typing import Mapping, Tuple, cast

from adam.language import TokenSequenceLinguisticDescription
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.surface_templates import SurfaceTemplate
from adam.perception.perception_graph import (
    PatternMatching,
    MatchedObjectNode,
    MatchedObjectPerceptionPredicate,
    PerceptionGraphPatternMatch,
)
from immutablecollections import immutabledict


def pattern_match_to_description(
    *,
    surface_template: SurfaceTemplate,
    pattern: PerceptionGraphTemplate,
    match: PerceptionGraphPatternMatch,
    matched_objects_to_names: Mapping[MatchedObjectNode, Tuple[str, ...]]
) -> TokenSequenceLinguisticDescription:
    """
    Given a `SurfaceTemplate`, will fill it in using a *match* for a *pattern*.
    This requires a mapping from matched object nodes in the perception
    to the strings which should be used to name them.
    """
    return surface_template.instantiate(
        template_variable_to_filler=immutabledict(
            (
                pattern.pattern_node_to_template_variable[pattern_node],
                # Wrapped in a tuple because
                # fillers can in general be
                # multiple words.
                (
                    matched_objects_to_names[
                        # We know, but the type
                        # system does not,
                        # that if a
                        # MatchedObjectPerceptionPredicate
                        # matched,
                        # the graph node must be a
                        # MatchedObjectNode
                        cast(MatchedObjectNode, matched_graph_node)
                    ],
                ),
            )
            for (
                pattern_node,
                matched_graph_node,
            ) in match.pattern_node_to_matched_graph_node.items()
            if isinstance(pattern_node, MatchedObjectPerceptionPredicate)
        )
    )
