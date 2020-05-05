from typing import Mapping, Tuple, Union, cast

from immutablecollections import immutabledict, immutableset

from adam.language import LinguisticDescription, TokenSequenceLinguisticDescription
from adam.learner import LearningExample
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.surface_templates import SurfaceTemplate
from adam.perception import PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception.perception_graph import (
    MatchedObjectNode,
    MatchedObjectPerceptionPredicate,
    PerceptionGraphPatternMatch,
)


def pattern_match_to_description(
    *,
    surface_template: SurfaceTemplate,
    pattern: PerceptionGraphTemplate,
    match: PerceptionGraphPatternMatch,
    matched_objects_to_names: Mapping[MatchedObjectNode, Tuple[str, ...]],
) -> TokenSequenceLinguisticDescription:
    """
    Given a `SurfaceTemplate`, will fill it in using a *match* for a *pattern*.
    This requires a mapping from matched object nodes in the perception
    to the strings which should be used to name them.
    """

    matched_object_nodes = immutableset(
        perception_node
        for perception_node in match.pattern_node_to_matched_graph_node.values()
        if isinstance(perception_node, MatchedObjectNode)
    )
    uniques = {node.name: node for node in matched_object_nodes}
    matched_object_nodes_without_names = matched_object_nodes - immutableset(
        matched_objects_to_names.keys()
    )
    if matched_object_nodes_without_names:
        if all(
            node.name in uniques.keys() for node in matched_object_nodes_without_names
        ):
            # For plurals, there is an exception when the match.match_node is not in matched_objects_to_names
            # This happens because the matcher in parent function returns the first match, while there could
            # multiple objects of the same kind
            matched_objects_to_names = {v: k for k, v in uniques.items()}
        else:
            raise RuntimeError(
                f"The following matched object nodes lack descriptions: "
                f"{matched_object_nodes_without_names}"
            )

    try:
        return surface_template.instantiate(
            template_variable_to_filler=immutabledict(
                (
                    pattern.pattern_node_to_template_variable[pattern_node],
                    matched_objects_to_names[
                        # We know, but the type system does not,
                        # that if a MatchedObjectPerceptionPredicate matched,
                        # the graph node must be a MatchedObjectNode
                        cast(MatchedObjectNode, matched_graph_node)
                    ],
                )
                for (
                    pattern_node,
                    matched_graph_node,
                ) in match.pattern_node_to_matched_graph_node.items()
                if isinstance(pattern_node, MatchedObjectPerceptionPredicate)
                # There can sometimes be relevant matched object nodes which are not themselves
                # slots, like the addressed possessor for "your X".
                and pattern_node in pattern.pattern_node_to_template_variable
            )
        )
    except KeyError:
        print("foo")
        raise


def assert_static_situation(
    to_check: Union[
        LearningExample[DevelopmentalPrimitivePerceptionFrame, LinguisticDescription],
        PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame],
    ]
):
    if isinstance(to_check, LearningExample):
        perception = to_check.perception
    else:
        perception = to_check

    if len(perception.frames) != 1:
        raise RuntimeError("Pursuit learner can only handle single frames for now")
    if not isinstance(perception.frames[0], DevelopmentalPrimitivePerceptionFrame):
        raise RuntimeError(f"Cannot process frame type: {type(perception.frames[0])}")
