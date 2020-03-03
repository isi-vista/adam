from typing import Any, Iterable, List, Mapping, Tuple

from attr import attrib, attrs
from attr.validators import deep_mapping, instance_of
from immutablecollections import immutableset
from immutablecollections.converter_utils import _to_immutabledict
from networkx import DiGraph

from adam.perception.perception_graph import (
    MatchedObjectNode,
    MatchedObjectPerceptionPredicate,
    PerceptionGraphPattern,
)

VerbSurfaceTemplate = Tuple[str, ...]

# Constants used to map locations in a verb phrase for mapping
_AGENT = "AGENT"
_PATIENT = "PATIENT"
_GOAL = "GOAL"
_THEME = "THEME"
_INSTRUMENT = "INSTRUMENT"

_EXPECTED_OBJECT_VARIABLES = {_AGENT, _PATIENT, _GOAL, _THEME, _INSTRUMENT}


@attrs(frozen=True, slots=True, eq=False)
class VerbPattern:
    # This pattern hold temporal scope information in the edges
    graph_pattern: PerceptionGraphPattern = attrib(
        validator=instance_of(PerceptionGraphPattern), kw_only=True
    )
    # Similar to prepositions, we have a mapping object locations from to match nodes
    object_variable_name_to_pattern_node: Mapping[
        str, MatchedObjectPerceptionPredicate
    ] = attrib(
        converter=_to_immutabledict,
        kw_only=True,
        validator=deep_mapping(
            instance_of(str), instance_of(MatchedObjectPerceptionPredicate)
        ),
    )

    @staticmethod
    def from_graph(
        perception_graph: DiGraph,
        description_to_match_object_node: Iterable[Tuple[str, MatchedObjectNode]],
    ) -> "VerbPattern":
        """
        This function returns a `VerbPattern` from a dynamic *perception_graph* and a *description_to_match_object_node*
        """
        description_to_node_immutable = immutableset(description_to_match_object_node)
        descriptions, _ = zip(*description_to_match_object_node)
        if _AGENT not in descriptions:
            raise RuntimeError(
                f"Expected at least one subject in a preposition graph. Found "
                f"{description_to_node_immutable}"
            )

        # Create a dynamic pattern from the digraph
        pattern_from_graph = PerceptionGraphPattern.from_graph(perception_graph, dynamic=True)
        pattern_graph = pattern_from_graph.perception_graph_pattern

        matched_object_to_matched_predicate = (
            pattern_from_graph.perception_graph_node_to_pattern_node
        )

        object_variable_name_to_pattern_node: List[Any] = []
        for description, object_node in description_to_node_immutable:
            if object_node in matched_object_to_matched_predicate:
                object_variable_name_to_pattern_node.append(
                    (description, matched_object_to_matched_predicate[object_node])
                )

        return VerbPattern(
            graph_pattern=pattern_graph,
            object_variable_name_to_pattern_node=_to_immutabledict(
                object_variable_name_to_pattern_node
            ),
        )

    def __attrs_post_init__(self) -> None:
        actual_object_nodes = set(self.object_variable_name_to_pattern_node.values())

        for object_node in actual_object_nodes:
            if (
                object_node
                not in self.graph_pattern._graph.nodes  # pylint:disable=protected-access
            ):
                raise RuntimeError(
                    f"Expected mapping which contained graph nodes"
                    f" but got {object_node} with id {id(object_node)}"
                    f" which doesn't exist in {self.graph_pattern}"
                )

        actual_object_variable_names = set(
            self.object_variable_name_to_pattern_node.keys()
        )
        if any(
            name not in _EXPECTED_OBJECT_VARIABLES
            for name in actual_object_variable_names
        ):
            raise RuntimeError(
                f"Expected a verb pattern to have "
                f"the object variables {_EXPECTED_OBJECT_VARIABLES} "
                f"but got {actual_object_variable_names}"
            )
