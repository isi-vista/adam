from logging import INFO
from typing import Any, List, Optional, Callable, Mapping

from attr import attrib, attrs
from attr.validators import deep_mapping, instance_of
from immutablecollections import ImmutableDict, immutabledict, immutableset
from immutablecollections.converter_utils import _to_immutabledict
from networkx import number_weakly_connected_components

from adam.learner.surface_templates import SurfaceTemplateVariable
from adam.ontology.ontology import Ontology
from adam.perception.perception_graph import (
    GraphLogger,
    MatchedObjectNode,
    MatchedObjectPerceptionPredicate,
    PerceptionGraph,
    PerceptionGraphPattern,
)


@attrs(frozen=True, slots=True, eq=False)
class PerceptionGraphTemplate:
    graph_pattern: PerceptionGraphPattern = attrib(
        validator=instance_of(PerceptionGraphPattern), kw_only=True
    )
    template_variable_to_pattern_node: ImmutableDict[
        SurfaceTemplateVariable, MatchedObjectPerceptionPredicate
    ] = attrib(
        converter=_to_immutabledict,
        kw_only=True,
        validator=deep_mapping(
            instance_of(SurfaceTemplateVariable),
            instance_of(MatchedObjectPerceptionPredicate),
        ),
        default=immutabledict(),
    )
    pattern_node_to_template_variable: ImmutableDict[
        MatchedObjectPerceptionPredicate, SurfaceTemplateVariable
    ] = attrib(init=False)

    @staticmethod
    def from_graph(
        perception_graph: PerceptionGraph,
        template_variable_to_matched_object_node: Mapping[
            SurfaceTemplateVariable, MatchedObjectNode
        ],
    ) -> "PerceptionGraphTemplate":
        # It is possible the perception graph has additional recognized objects
        # which are not aligned to surface template slots.
        # We assume these are not arguments of the verb and remove them from the perception
        # before creating a pattern.

        matched_object_nodes_aligned_to_template_slots = immutableset(
            template_variable_to_matched_object_node.values()
        )

        perception_digraph_without_irrelevant_objects = perception_graph.copy_as_digraph()
        nodes_to_remove = immutableset(
            node
            for node in perception_digraph_without_irrelevant_objects.nodes
            if isinstance(node, MatchedObjectNode)
            and node not in matched_object_nodes_aligned_to_template_slots
        )
        for node_to_remove in nodes_to_remove:
            perception_digraph_without_irrelevant_objects.remove_node(node_to_remove)
        perception_without_irrelevant_objects = PerceptionGraph(
            perception_digraph_without_irrelevant_objects,
            dynamic=perception_graph.dynamic,
        )

        pattern_from_graph = PerceptionGraphPattern.from_graph(
            perception_without_irrelevant_objects
        )
        pattern_graph = pattern_from_graph.perception_graph_pattern
        matched_object_to_matched_predicate = (
            pattern_from_graph.perception_graph_node_to_pattern_node
        )

        template_variable_to_pattern_node: List[Any] = []

        for (
            template_variable,
            object_node,
        ) in template_variable_to_matched_object_node.items():
            if object_node in matched_object_to_matched_predicate:
                template_variable_to_pattern_node.append(
                    (template_variable, matched_object_to_matched_predicate[object_node])
                )

        return PerceptionGraphTemplate(
            graph_pattern=pattern_graph,
            template_variable_to_pattern_node=template_variable_to_pattern_node,
        )

    def __attrs_post_init__(self) -> None:
        object_predicate_nodes = set(self.template_variable_to_pattern_node.values())

        for object_node in object_predicate_nodes:
            if (
                object_node
                not in self.graph_pattern._graph.nodes  # pylint:disable=protected-access
            ):
                raise RuntimeError(
                    f"Expected mapping which contained graph nodes"
                    f" but got {object_node} with id {id(object_node)}"
                    f" which doesn't exist in {self.graph_pattern}"
                )

    def intersection(
        self,
        pattern: "PerceptionGraphTemplate",
        *,
        graph_logger: Optional[GraphLogger] = None,
        ontology: Ontology,
        debug_callback: Optional[Callable[[Any, Any], None]] = None,
    ) -> Optional["PerceptionGraphTemplate"]:
        r"""
        Gets the `PerceptionGraphTemplate` which contains all aspects of a pattern
        which are both in this template and *other_template*.

        If this intersection is an empty graph or would not contain all `SurfaceTemplateVariables`\ s
        """
        if self.graph_pattern.dynamic != pattern.graph_pattern.dynamic:
            raise RuntimeError("Can only intersection patterns of the same dynamic-ness")

        num_self_weakly_connected = number_weakly_connected_components(
            self.graph_pattern._graph  # pylint:disable=protected-access
        )
        if num_self_weakly_connected > 1:
            raise RuntimeError(
                f"Graph pattern contains multiple ( {num_self_weakly_connected} ) "
                f"weakly connected components heading into intersection. "
                f"Violating pattern: {self}"
            )

        num_pattern_weakly_connected = number_weakly_connected_components(
            pattern.graph_pattern._graph  # pylint:disable=protected-access
        )
        if num_pattern_weakly_connected > 1:
            raise RuntimeError(
                f"Graph pattern contains multiple ( {num_pattern_weakly_connected} ) "
                f"weakly connected components heading into intersection. "
                f"Violating pattern: {pattern}"
            )

        # First we just intersect the pattern graph.
        intersected_pattern = self.graph_pattern.intersection(
            pattern.graph_pattern,
            graph_logger=graph_logger,
            ontology=ontology,
            debug_callback=debug_callback,
        )

        if intersected_pattern:
            if self.graph_pattern.dynamic != intersected_pattern.dynamic:
                raise RuntimeError(
                    "Something is wrong - pattern dynamic-ness should not change "
                    "after intersection"
                )

            # If we get a successful intersection,
            # we then need to make sure we have the correct SurfaceTemplateVariables.

            # It would be more intuitive to use self.template_variable_to_pattern_node,
            # but the pattern intersection code seems to prefer to return nodes
            # from the right-hand graph.
            template_variable_to_pattern_node = pattern.template_variable_to_pattern_node
            if graph_logger:
                graph_logger.log_graph(intersected_pattern, INFO, "Intersected pattern")
            for (
                surface_template_variable,
                object_wildcard,
            ) in template_variable_to_pattern_node.items():
                if object_wildcard not in intersected_pattern:
                    raise RuntimeError(
                        f"Result of intersection lacks a wildcard node "
                        f"for template variable {surface_template_variable}"
                    )

            return PerceptionGraphTemplate(
                graph_pattern=intersected_pattern,
                template_variable_to_pattern_node=template_variable_to_pattern_node,
            )
        else:
            return None

    @pattern_node_to_template_variable.default
    def _init_pattern_node_to_template_variable(
        self
    ) -> ImmutableDict[MatchedObjectPerceptionPredicate, SurfaceTemplateVariable]:
        return immutabledict(
            {v: k for k, v in self.template_variable_to_pattern_node.items()}
        )
