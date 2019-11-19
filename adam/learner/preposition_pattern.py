from typing import Any, Iterable, List, Mapping, Tuple

from attr.validators import deep_mapping, instance_of
from immutablecollections import immutableset
from immutablecollections.converter_utils import _to_immutabledict
from networkx import DiGraph

from adam.perception.perception_graph import (
    MatchedObjectNode,
    MatchedObjectPerceptionPredicate,
    PerceptionGraphPattern,
)
from attr import attrib, attrs

# Constants used to map locations in a prepositional phrase for mapping
_MODIFIED = "MODIFIED"
_GROUND = "GROUND"

_EXPECTED_OBJECT_VARIABLES = {_MODIFIED, _GROUND}


@attrs(frozen=True, slots=True, eq=False)
class PrepositionPattern:
    graph_pattern: PerceptionGraphPattern = attrib(
        validator=instance_of(PerceptionGraphPattern), kw_only=True
    )
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
    ) -> "PrepositionPattern":
        """
        This function returns a `PrepositionPattern` from a *perception_graph* and a *description_to_match_object_node*
        mapping where GROUND is the first item in the list and MODIFIED is the second.
        """
        description_to_node_immutable = immutableset(description_to_match_object_node)
        if len(description_to_node_immutable) != 2:
            raise RuntimeError(
                f"Expected only two prepositions in a preposition graph. Found "
                f"{len(description_to_node_immutable)} in {description_to_node_immutable}"
            )
        pattern_from_graph = PerceptionGraphPattern.from_graph(perception_graph)
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

        return PrepositionPattern(
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
        if actual_object_variable_names != _EXPECTED_OBJECT_VARIABLES:
            raise RuntimeError(
                f"Expected a preposition pattern to have "
                f"the object variables {_EXPECTED_OBJECT_VARIABLES} "
                f"but got {actual_object_variable_names}"
            )

    def intersection(self, pattern: "PrepositionPattern") -> "PrepositionPattern":
        graph_pattern = self.graph_pattern.intersection(pattern.graph_pattern)
        mapping_builder = []
        items_to_iterate: List[Tuple[str, MatchedObjectPerceptionPredicate]] = []
        items_to_iterate.extend(self.object_variable_name_to_pattern_node.items())
        items_to_iterate.extend(pattern.object_variable_name_to_pattern_node.items())
        for name, pattern_node in immutableset(items_to_iterate):
            if (
                pattern_node
                in graph_pattern._graph.nodes  # pylint:disable=protected-access
            ):
                mapping_builder.append((name, pattern_node))

        return PrepositionPattern(
            graph_pattern=graph_pattern,
            object_variable_name_to_pattern_node=mapping_builder,
        )
