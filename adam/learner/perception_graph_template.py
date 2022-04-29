from logging import INFO
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Union, Iterable

from attr import attrib, attrs
from attr.validators import deep_mapping, instance_of
from immutablecollections import (
    ImmutableDict,
    immutabledict,
    immutablesetmultidict,
    ImmutableSetMultiDict,
)
from immutablecollections.converter_utils import _to_immutabledict
from networkx import number_weakly_connected_components

from adam.ontology.ontology import Ontology
from adam.perception import MatchMode
from adam.perception.perception_graph import (
    GraphLogger,
    ObjectSemanticNodePerceptionPredicate,
    PerceptionGraph,
    PerceptionGraphPattern,
    raise_graph_exception,
    NodePredicate,
    TemporalScope,
    PerceptionGraphPatternMatch,
)
from adam.semantics import ObjectSemanticNode, SyntaxSemanticsVariable


@attrs(frozen=True, slots=True, eq=False)
class PerceptionGraphTemplate:
    graph_pattern: PerceptionGraphPattern = attrib(
        validator=instance_of(PerceptionGraphPattern), kw_only=True
    )
    template_variable_to_pattern_node: ImmutableDict[
        SyntaxSemanticsVariable, ObjectSemanticNodePerceptionPredicate
    ] = attrib(
        converter=_to_immutabledict,
        kw_only=True,
        validator=deep_mapping(
            instance_of(SyntaxSemanticsVariable),
            instance_of(ObjectSemanticNodePerceptionPredicate),
        ),
        default=immutabledict(),
    )
    pattern_node_to_template_variable: ImmutableDict[
        ObjectSemanticNodePerceptionPredicate, SyntaxSemanticsVariable
    ] = attrib(init=False)

    @staticmethod
    def from_graph(
        perception_graph: PerceptionGraph,
        template_variable_to_matched_object_node: Mapping[
            SyntaxSemanticsVariable, ObjectSemanticNode
        ],
        *,
        min_continuous_feature_match_score: float,
    ) -> "PerceptionGraphTemplate":
        # It is possible the perception graph has additional recognized objects
        # which are not aligned to surface template slots.
        # We assume these are not arguments of the verb and remove them from the perception
        # before creating a pattern.
        pattern_from_graph = PerceptionGraphPattern.from_graph(
            perception_graph,
            min_continuous_feature_match_score=min_continuous_feature_match_score,
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
        allowed_matches: ImmutableSetMultiDict[
            NodePredicate, NodePredicate
        ] = immutablesetmultidict(),
        match_mode: MatchMode,
        trim_after_match: Optional[
            Callable[[PerceptionGraphPattern], PerceptionGraphPattern]
        ] = None,
    ) -> Optional["PerceptionGraphTemplate"]:
        r"""
        Gets the `PerceptionGraphTemplate` which contains all aspects of a pattern
        which are both in this template and *other_template*.

        If this intersection is an empty graph or would not contain all `SyntaxSemanticsVariable`\ s,
        this returns None.
        """
        result = self.intersection_getting_match(
            pattern,
            graph_logger=graph_logger,
            ontology=ontology,
            debug_callback=debug_callback,
            allowed_matches=allowed_matches,
            match_mode=match_mode,
            trim_after_match=trim_after_match,
        )

        return result.intersection if result else result

    def intersection_getting_match(
        self,
        pattern: "PerceptionGraphTemplate",
        *,
        graph_logger: Optional[GraphLogger] = None,
        ontology: Ontology,
        debug_callback: Optional[Callable[[Any, Any], None]] = None,
        allowed_matches: ImmutableSetMultiDict[
            NodePredicate, NodePredicate
        ] = immutablesetmultidict(),
        match_mode: MatchMode,
        trim_after_match: Optional[
            Callable[[PerceptionGraphPattern], PerceptionGraphPattern]
        ] = None,
    ):
        if self.graph_pattern.dynamic != pattern.graph_pattern.dynamic:
            raise RuntimeError("Can only intersection patterns of the same dynamic-ness")

        num_self_weakly_connected = number_weakly_connected_components(
            self.graph_pattern._graph  # pylint:disable=protected-access
        )
        if num_self_weakly_connected > 1:
            raise_graph_exception(
                f"Graph pattern contains multiple ( {num_self_weakly_connected} ) "
                f"weakly connected components heading into intersection. ",
                self.graph_pattern,
            )

        num_pattern_weakly_connected = number_weakly_connected_components(
            pattern.graph_pattern._graph  # pylint:disable=protected-access
        )
        if num_pattern_weakly_connected > 1:
            raise_graph_exception(
                f"Graph pattern contains multiple ( {num_pattern_weakly_connected} ) "
                f"weakly connected components heading into intersection. ",
                pattern.graph_pattern,
            )
        # First we just intersect the pattern graph.
        intersected_pattern_match = self.graph_pattern.intersection_getting_match(
            pattern.graph_pattern,
            graph_logger=graph_logger,
            ontology=ontology,
            debug_callback=debug_callback,
            allowed_matches=allowed_matches,
            match_mode=match_mode,
            trim_after_match=trim_after_match,
        )
        if intersected_pattern_match is not None:
            if (
                self.graph_pattern.dynamic
                != intersected_pattern_match.matched_pattern.dynamic
            ):
                raise RuntimeError(
                    "Something is wrong - pattern dynamic-ness should not change "
                    "after intersection"
                )

            # If we get a successful intersection,
            # we then need to make sure we have the correct SyntaxSemanticsVariables.

            # It would be more intuitive to use self.template_variable_to_pattern_node,
            # but the pattern intersection code seems to prefer to return nodes
            # from the right-hand graph.
            template_variable_to_pattern_node = pattern.template_variable_to_pattern_node
            if graph_logger:
                graph_logger.log_graph(
                    intersected_pattern_match.matched_pattern, INFO, "Intersected pattern"
                )
            slots_preserved = True
            for (_, object_wildcard) in template_variable_to_pattern_node.items():
                # we return none here since this means that the given template cannot be learned from since one of the slots has been pruned away
                if object_wildcard not in intersected_pattern_match.matched_pattern:
                    slots_preserved = False
                    break

            result = (
                PerceptionGraphTemplateIntersectionResult(
                    intersected_pattern_match,
                    PerceptionGraphTemplate(
                        graph_pattern=intersected_pattern_match.matched_pattern,
                        template_variable_to_pattern_node=template_variable_to_pattern_node,
                    ),
                )
                if slots_preserved
                else None
            )
        else:
            result = None
        return result

    @pattern_node_to_template_variable.default
    def _init_pattern_node_to_template_variable(
        self,
    ) -> ImmutableDict[ObjectSemanticNodePerceptionPredicate, SyntaxSemanticsVariable]:
        return immutabledict(
            {v: k for k, v in self.template_variable_to_pattern_node.items()}
        )

    def render_to_file(  # pragma: no cover
        self,
        graph_name: str,
        output_file: Path,
        *,
        match_correspondence_ids: Mapping[Any, str] = immutabledict(),
        robust=True,
    ):
        self.graph_pattern.render_to_file(
            graph_name,
            output_file,
            match_correspondence_ids=match_correspondence_ids,
            robust=robust,
            replace_node_labels=immutabledict(
                (pattern_node, template_variable.name)
                for (
                    pattern_node,
                    template_variable,
                ) in self.pattern_node_to_template_variable.items()
            ),
        )

    def copy_with_temporal_scopes(
        self, required_temporal_scopes: Union[TemporalScope, Iterable[TemporalScope]]
    ) -> "PerceptionGraphTemplate":
        r"""
        Produces a copy of this perception graph pattern
        where all edge predicates now require that the edge in the target graph being matched
        hold at all of the *required_temporal_scopes*.
        """
        return PerceptionGraphTemplate(
            graph_pattern=self.graph_pattern.copy_with_temporal_scopes(
                required_temporal_scopes
            ),
            template_variable_to_pattern_node=self.template_variable_to_pattern_node,
        )

    def copy_replacing_nodes(
        self, current_to_new_node: Mapping[NodePredicate, NodePredicate]
    ) -> "PerceptionGraphTemplate":
        """
        Creates a copy of the perception graph template with the pattern updated to align to the new
        nodes passed in via the mapping.
        """
        return PerceptionGraphTemplate(
            graph_pattern=self.graph_pattern.copy_replacing_nodes(current_to_new_node),
            template_variable_to_pattern_node=self.template_variable_to_pattern_node,
        )

    def copy_removing_temporal_scopes(self) -> "PerceptionGraphTemplate":
        if not self.graph_pattern.dynamic:
            return self

        return PerceptionGraphTemplate(
            graph_pattern=self.graph_pattern.copy_removing_temporal_scopes(),
            template_variable_to_pattern_node=self.template_variable_to_pattern_node,
        )


@attrs(frozen=True, slots=True)
class PerceptionGraphTemplateIntersectionResult:
    match: PerceptionGraphPatternMatch = attrib(
        validator=instance_of(PerceptionGraphPatternMatch)
    )
    intersection: PerceptionGraphTemplate = attrib(
        validator=instance_of(PerceptionGraphTemplate)
    )

    def confirm_match(self):
        # Assumes that when we do the intersection in the template learner, we call
        # the intersection_getting_match() method on self, not the argument.
        self.match.confirm_pattern_match()
