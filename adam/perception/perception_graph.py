r"""
Code for representing `DevelopmentalPrimitivePerceptionFrame`\ s as directed graphs
and for matching patterns against such graph.
Such patterns could be used to implement object recognition,
among other things.

This file first defines `PerceptionGraph`\ s,
then defines `PerceptionGraphPattern`\ s to match them.

Readers should start with `PerceptionGraphProtocol`, `PerceptionGraph`,
and `PerceptionGraphPattern` before reading other parts of this module.
"""
import logging
import pickle
from abc import ABC, abstractmethod
from enum import Enum
from itertools import chain, product, repeat
from pathlib import Path
from time import process_time
from typing import (
    AbstractSet,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    Sized,
    Tuple,
    TypeVar,
    Union,
    cast,
    Generic,
    Sequence,
)
from uuid import uuid4

import graphviz
from more_itertools import first, ilen
from networkx import DiGraph, connected_components, is_isomorphic, set_node_attributes
from typing_extensions import Protocol

from adam.axes import AxesInfo, HasAxes
from adam.axis import GeonAxis
from adam.geon import Geon, MaybeHasGeon, CrossSection
from adam.math_3d import Point, DepthPoint
from adam.ontology import OntologyNode
from adam.ontology.ontology import Ontology
from adam.ontology.phase1_ontology import (
    COLOR,
    GAILA_PHASE_1_ONTOLOGY,
    LIQUID,
    PART_OF,
    RECOGNIZED_PARTICULAR_PROPERTY,
    VOLITIONALLY_INVOLVED,
)
from adam.ontology import IS_SPEAKER, IS_ADDRESSEE
from adam.ontology.phase1_spatial_relations import (
    Direction,
    PathOperator,
    Region,
    SpatialPath,
)
from adam.ontology.structural_schema import ObjectStructuralSchema
from adam.perception import (
    ObjectPerception,
    PerceptualRepresentation,
    MatchMode,
    PerceptionT,
)
from adam.perception._matcher import GraphMatching
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
    HasBinaryProperty,
    HasColor,
    RgbColorPerception,
)
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator,
)
from adam.perception.perception_graph_nodes import (
    PerceptionGraphNode,
    UnwrappedPerceptionGraphNode,
    ObjectClusterNode,
    CategoricalNode,
    ContinuousNode,
    RgbColorNode,
    CielabColorNode,
    GraphNode,
    ObjectStroke,
    StrokeGNNRecognitionNode,
    JointPointNode,
    TrajectoryRecognitionNode,
)
from adam.perception.perception_graph_predicates import (
    NodePredicate,
    GeonPredicate,
    CrossSectionPredicate,
    RegionPredicate,
    AxisPredicate,
    AnyObjectPerception,
    IsOntologyNodePredicate,
    IsColorNodePredicate,
    ObjectSemanticNodePerceptionPredicate,
    IsPathPredicate,
    PathOperatorPredicate,
    AnyObjectPredicate,
    CategoricalPredicate,
    ContinuousPredicate,
    RgbColorPredicate,
    CielabColorPredicate,
    ObjectStrokePredicate,
    StrokeGNNRecognitionPredicate,
    DistributionalContinuousPredicate,
    JointPointPredicate,
    TrajectoryRecognitionPredicate,
)
from adam.perception.visual_perception import VisualPerceptionFrame
from adam.random_utils import RandomChooser
from adam.relation import Relation
from adam.semantics import ObjectSemanticNode, SemanticNode
from adam.situation import SituationObject
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.utilities import sign
from adam.utils.networkx_utils import copy_digraph, digraph_with_nodes_sorted_by
from attr import attrib, attrs
from attr.validators import deep_iterable, instance_of
from immutablecollections import (
    ImmutableDict,
    ImmutableSet,
    immutabledict,
    immutableset,
    immutablesetmultidict,
    ImmutableSetMultiDict,
)
from immutablecollections.converter_utils import (
    _to_immutabledict,
    _to_immutableset,
)
from vistautils.misc_utils import str_list_limited
from vistautils.preconditions import check_arg
from vistautils.range import Range

LABEL = "label"


class Incrementer:
    def __init__(self, initial_value=0) -> None:
        self._value = initial_value

    def value(self) -> int:
        return self._value

    def increment(self, amount=1) -> None:
        self._value += amount


DebugCallableType = Callable[[DiGraph, Dict[Any, Any]], None]

# If this is changed, assert_valid_edge_label below needs to be updated.
EdgeLabel = Union[OntologyNode, str, Direction[Any]]
"""
This is the core information stored on a perception graph edge.
This is wrapped in `TemporallyScopedEdgeLabel`
before actually being applied to a dynamic `DiGraph` edge.
"""


def assert_valid_edge_label(base_edge_label: Any) -> None:
    # This needs to be updated
    # if the EdgeLabel type alias is updated.
    # Sadly type aliases cannot be used with isinstance. :-(
    if not isinstance(base_edge_label, (OntologyNode, str, Direction)):
        raise RuntimeError(
            f"Edge labels must be either OntologyNode, str, "
            f"or Direction, but got {base_edge_label}"
        )


def valid_edge_label(
    inst: Any, attr: Any, value: Any  # pylint:disable=unused-argument
) -> None:
    """
    Wraps `assert_valid_edge_label` for use as an *attrs* validator.
    """
    assert_valid_edge_label(value)


class TemporalScope(Enum):
    """
    In a dynamic situation,
    specifies the relationship of perception graph edges to the perception frames.
    """

    BEFORE = "before"
    """
    Indicates a relationship holds in the first frame.
    """
    AFTER = "after"
    """
    Indicates a relationship holds in the second frame.
    """
    DURING = "during"
    """
    Indicates a relationship continuously holds in the interval between frames.
    """
    AT_SOME_POINT = "at-some-point"
    """
    Indicates a relationship holds at some point in the interval between frames.
    """


ENTIRE_SCENE = immutableset([TemporalScope.BEFORE, TemporalScope.AFTER])


@attrs(slots=True, frozen=True, repr=False)
class TemporallyScopedEdgeLabel:
    r"""
    An edge attribute in a `PerceptionGraph` which is annotated for what times it holds true.

    These should only be used in  `PerceptionGraph`\ s representing dynamic situations,
    in which every edge label should be wrapped with this class.
    """
    attribute: EdgeLabel = attrib(validator=valid_edge_label)
    temporal_specifiers: ImmutableSet[TemporalScope] = attrib(
        converter=_to_immutableset,
        default=immutableset(),
        validator=deep_iterable(instance_of(TemporalScope)),
    )

    def __attrs_post_init__(self) -> None:
        if not self.temporal_specifiers:
            raise RuntimeError(
                "Cannot have a TemporallyScopedPerceptionGraphEdgeAttribute "
                "without any temporal specifiers"
            )

    @staticmethod
    def for_dynamic_perception(
        attribute: EdgeLabel, when=Union[TemporalScope, Iterable[TemporalScope]]
    ) -> "TemporallyScopedEdgeLabel":
        if isinstance(when, TemporalScope):
            when = [when]
        return TemporallyScopedEdgeLabel(attribute, when)

    def __repr__(self) -> str:
        if self.temporal_specifiers == _BEFORE_AND_AFTER:
            # To reduce clutter, we omit temporal scopes for things
            # which hold both before and after an action
            return repr(self.attribute)
        else:
            temporal_scope_names = [
                temporal_specifier.name for temporal_specifier in self.temporal_specifiers
            ]
            return f"{self.attribute!r}@{temporal_scope_names}"


# certain constant edges used by PerceptionGraphs
REFERENCE_OBJECT_SOURCE_LABEL = OntologyNode("reference-object-source")
REFERENCE_OBJECT_DESTINATION_LABEL = OntologyNode("reference-object-destination")

REFERENCE_OBJECT_LABEL = OntologyNode("reference-object")
"""
Edge label in a `PerceptionGraph` linking a `Region` to its reference object.
"""
PRIMARY_AXIS_LABEL = OntologyNode("primary-axis")
"""
Edge label in a `PerceptionGraph` linking a `Geon` to its primary `GeonAxis`.
"""

HAS_AXIS_LABEL = OntologyNode("has-axis")
"""
Edge label in a `PerceptionGraph` linking any node of type `HasAxes` 
to its associated axes.
"""
GENERATING_AXIS_LABEL = OntologyNode("generating-axis")
"""
Edge label in a `PerceptionGraph` linking a `Geon` to its generating `GeonAxis`.
"""
HAS_GEON_LABEL = OntologyNode("geon")
"""
Edge label in a `PerceptionGraph` linking an `ObjectPerception` to its associated `Geon`.
"""
HAS_PROPERTY_LABEL = OntologyNode("has-property")
"""
Edge label in a `PerceptionGraph` linking an `ObjectPerception` to its associated `PropertyPerception`.
"""
FACING_OBJECT_LABEL = OntologyNode("facing-axis")
"""
Edge label in a `PerceptionGraph` linking a `GeonAxis` to an `ObjectPerception` it is facing
"""
REFERENCE_AXIS_LABEL = OntologyNode("reference-axis")
"""
Edge label in a `PerceptionGraph` linking a `SpatialPath` to its reference axis.
"""
HAS_PATH_LABEL = OntologyNode("has-path")
"""
Edge label in a `PerceptionGraph` linking an object to the `SpatialPath`
which it takes in a dynamic situation.
"""
HAS_PATH_OPERATOR = OntologyNode("has-path-operator")
"""
Edge label in a `PerceptionGraph` linking a `SpatialPath`
to its `PathOperator`
"""
HAS_STROKE_LABEL = OntologyNode("has-stroke-label")
"""
Edge label in a `PerceptionGraph` linking an object cluster to one
of the `ObjectStroke`s which make up the object.
"""
ADJACENT_STROKE_LABEL = OntologyNode("adjacent-stroke-label")
"""
Edge label in a `PerceptionGraph` linking an `ObjectStroke` to another
`ObjectStroke` to indicate the two are adjacent. The relationship is
symmetric even though the edge is not.
"""
ORIENTATION_CHANGED_PROPERTY = OntologyNode("orientation-changed")
"""
Property used in perception graphs to indicate an orientation change
while an object traverses a path.
"""
INVOLVED_IN_ACTION = OntologyNode("action-references-object")
"""
Edge label used in perception graphs to indicate an object cluster participates in a given action.
"""
JOINT_POINT_TRAVERSAL = OntologyNode("joint-point-traversal")
"""
Edge label used in perception graphs to link an action decode to its corresponding joint points. 

`JointPointNode`s are also referenced by their `ObjectClusterNode` with a `HAS_PROPERTY` label.
"""


class PerceptionGraphProtocol(Protocol):
    _graph: DiGraph
    dynamic: bool

    def copy_as_digraph(self) -> DiGraph:
        return self._graph.copy()

    def render_to_file(
        self,
        graph_name: str,
        output_file: Path,
        *,
        match_correspondence_ids: Mapping[Any, str] = immutabledict(),
        robust=True,
        replace_node_labels: Mapping[Any, str] = immutabledict(),
    ) -> None:
        """
        Debugging tool to render the graph to PDF using *dot*.
        """

    def __len__(self) -> int:
        """Length of the internal graph."""

    def __contains__(self, item) -> bool:
        """Check for node containment."""

    def __iter__(self):
        """Iterate over the graph nodes."""

    def text_dump(self) -> str:
        lines = []
        lines.append("Nodes:")
        lines.extend(f"\t{node}" for node in self._graph.nodes)
        lines.append("\nEdges:")
        lines.extend(f"\t{edge}" for edge in self._graph.edges(data=LABEL))
        return "\n".join(lines)


@attrs(frozen=True, repr=False)
class PerceptionGraph(PerceptionGraphProtocol, Sized, Iterable[PerceptionGraphNode]):
    r"""
    Represents a `DevelopmentalPrimitivePerceptionFrame` as a directed graph.

    Perception graphs may be static (representing a single snapshot of a situation)
    or dynamic, representing a changing situation.
    This is encoded by the *_dynamic* field.

    `ObjectPerception`\ s, properties, `Geon`\ s, `GeonAxis`\ s, and `Region`\ s are nodes.
    Edges should have the *label* attribute mapped to an `EdgeLabel`, if the graph is static,
    or to `TemporallyScopedEdgeLabel`, if dynamic.

    These can be matched against by `PerceptionGraphPattern`\ s.
    """
    _graph: DiGraph = attrib(validator=instance_of(DiGraph), converter=copy_digraph)
    dynamic: bool = attrib(validator=instance_of(bool), default=False)

    @staticmethod
    def from_frame(frame: DevelopmentalPrimitivePerceptionFrame) -> "PerceptionGraph":
        """
        Gets the `PerceptionGraph` corresponding to a `DevelopmentalPrimitivePerceptionFrame`.
        """
        return _DevelopmentalPrimitivePerceptionFrameTranslation().translate_frame(frame)

    @staticmethod
    def from_dynamic_perceptual_representation(
        perceptual_representation: PerceptualRepresentation[
            DevelopmentalPrimitivePerceptionFrame
        ],
    ) -> "PerceptionGraph":
        return _DevelopmentalPrimitivePerceptionFrameTranslation().translate_frames(
            perceptual_representation
        )

    @staticmethod
    def add_temporal_scopes_to_edges(
        digraph: DiGraph, temporal_scopes: Union[TemporalScope, Iterable[TemporalScope]]
    ) -> DiGraph:
        r"""
        Modifies the given digraph in place, applying the given `TemporalScope`\ s to all edges.
        This new graph will be dynamic.

        Note that this should only be applied to static perception digraphs.
        """
        # Assume the graph is dynamic if an arbitrary edge label is temporally scoped.
        _, _, a_label = first(digraph.edges(data=LABEL))
        if isinstance(a_label, TemporallyScopedEdgeLabel):
            raise RuntimeError(
                "Cannot use add_temporal_scopes_to_edges on a graph which is "
                "already dynamic"
            )

        for (source, target) in digraph.edges():
            unwrapped_label = digraph.edges[source, target][LABEL]
            temporally_scoped_label = TemporallyScopedEdgeLabel.for_dynamic_perception(
                unwrapped_label, when=temporal_scopes
            )
            digraph.edges[source, target][LABEL] = temporally_scoped_label

        return digraph

    @staticmethod
    def from_simulated_frame(frame: VisualPerceptionFrame) -> "PerceptionGraph":
        return _VisualPerceptionFrameTranslation().translate_frame(frame)

    @staticmethod
    def from_dynamic_simulated_perception_frame(
        perceptual_representation: PerceptualRepresentation[VisualPerceptionFrame],
    ) -> "PerceptionGraph":
        return _VisualPerceptionFrameTranslation().translate_frames(
            perceptual_representation
        )

    def copy_with_temporal_scopes(
        self, temporal_scopes: Union[TemporalScope, Iterable[TemporalScope]]
    ) -> "PerceptionGraph":
        r"""
        Produces a copy of this perception graph with the given `TemporalScope`\ s
        applied to all edges. This new graph will be dynamic.

        This graph must be a static graph or a `RuntimeError` will be raised.
        """
        if self.dynamic:
            raise RuntimeError(
                "Cannot use dynamic_copy_with_temporal_scopes on a graph which is "
                "already dynamic"
            )

        wrapped_graph = self.add_temporal_scopes_to_edges(
            self._graph.copy(), temporal_scopes
        )

        return PerceptionGraph(dynamic=True, graph=wrapped_graph)

    def subgraph_by_nodes(
        self, nodes_to_keep: AbstractSet[PerceptionGraphNode]
    ) -> "PerceptionGraph":
        return PerceptionGraph(self._graph.subgraph(nodes_to_keep), dynamic=self.dynamic)

    def count_nodes_matching(
        self, node_predicate: Callable[["PerceptionGraphNode"], bool]
    ):
        return ilen(filter(node_predicate, self._graph.nodes))  # type: ignore

    def render_to_file(  # pragma: no cover
        self,
        graph_name: str,
        output_file: Path,
        *,
        match_correspondence_ids: Mapping[Any, str] = immutabledict(),
        robust=True,
        replace_node_labels: Mapping[Any, str] = immutabledict(),
    ) -> None:
        """
        Debugging tool to render the graph to PDF using *dot*.

        If this graph has been matched against a pattern,
        the matched nodes can be highlighted and given labels which show
        what the correspond to in a pattern by supplying
        *match_correspondence_ids* which maps graph nodes to
        the desired correspondence labels.

        If *robust* is *True* (the default), then this will suppress crashes on failures.
        """
        dot_graph = graphviz.Digraph(graph_name)
        dot_graph.attr(rankdir="LR")
        # combine parallel edges to cut down on clutter
        dot_graph.attr(concentrate="true")

        next_node_id = Incrementer()

        # add all nodes to the graph
        perception_nodes_to_dot_node_ids = {
            perception_node: self._to_dot_node(
                dot_graph,
                perception_node,
                next_node_id,
                match_correspondence_ids,
                replace_node_labels=replace_node_labels,
            )
            for perception_node in self._graph.nodes
        }

        for (source_node, target_node, label) in self._graph.edges.data(LABEL):
            if isinstance(label, Direction):
                sign_string = "+" if label.positive else "-"
                edge_label = f"{sign_string}relative-to"
            else:
                edge_label = str(label)

            source_dot_node = perception_nodes_to_dot_node_ids[source_node]
            target_dot_node = perception_nodes_to_dot_node_ids[target_node]

            # we want bigger things on the left, smaller things on the right,
            # but graphviz wants to put edge sources on the left,
            # so we need to reverse these edges from GraphViz's point-of-view.
            reverse_rank_order = label == PART_OF or edge_label == "reference-object"

            # in-region relationships can run anywhich way
            # without respect to a good layout hierarchy
            constraint = edge_label != "in-region"
            constraint_string = "true" if constraint else "false"

            if reverse_rank_order:
                dot_graph.edge(
                    target_dot_node,
                    source_dot_node,
                    edge_label,
                    dir="back",
                    constraint=constraint_string,
                )
            else:
                dot_graph.edge(
                    source_dot_node,
                    target_dot_node,
                    edge_label,
                    constraint=constraint_string,
                )

        try:
            dot_graph.render(str(output_file))
        except Exception as e:  # pylint:disable=broad-except
            if robust:
                logging.warning("Error during dot rendering: %s", e)
            else:
                raise

    def _to_dot_node(
        self,
        dot_graph: graphviz.Digraph,
        perception_node: PerceptionGraphNode,
        next_node_id: Incrementer,
        match_correspondence_ids: Mapping[Any, str],
        *,
        replace_node_labels: Mapping[Any, str] = immutabledict(),
    ) -> str:
        unwrapped_perception_node: UnwrappedPerceptionGraphNode
        if isinstance(perception_node, tuple):
            unwrapped_perception_node = perception_node[0]
        else:
            unwrapped_perception_node = perception_node

        if perception_node in replace_node_labels:
            label = replace_node_labels[perception_node]
        elif isinstance(unwrapped_perception_node, ObjectPerception):
            # object perceptions have no content, so they are blank nodes
            label = unwrapped_perception_node.debug_handle
        elif isinstance(unwrapped_perception_node, Region):
            # regions do have content but we express those as edges to other nodes
            label = f"reg:{unwrapped_perception_node}"
        elif isinstance(unwrapped_perception_node, GeonAxis):
            label = f"axis:{unwrapped_perception_node.debug_name}"
        elif isinstance(unwrapped_perception_node, RgbColorPerception):
            label = unwrapped_perception_node.hex
        elif isinstance(unwrapped_perception_node, RgbColorPerception):
            label = unwrapped_perception_node.hex
        elif isinstance(unwrapped_perception_node, OntologyNode):
            label = unwrapped_perception_node.handle
        elif isinstance(unwrapped_perception_node, Geon):
            label = str(unwrapped_perception_node.cross_section) + str(
                unwrapped_perception_node.cross_section_size
            )
        elif isinstance(unwrapped_perception_node, CrossSection):
            label = str(unwrapped_perception_node)
        elif isinstance(unwrapped_perception_node, ObjectSemanticNode):
            label = " ".join(unwrapped_perception_node.concept.debug_string)
        elif isinstance(unwrapped_perception_node, SpatialPath):
            label = "path"
        elif isinstance(unwrapped_perception_node, PathOperator):
            label = unwrapped_perception_node.name
        elif isinstance(unwrapped_perception_node, ObjectClusterNode):
            label = f"Object Cluster {unwrapped_perception_node.cluster_id} | View: {unwrapped_perception_node.viewpoint_id}"
        elif isinstance(unwrapped_perception_node, ObjectStroke):
            label = f"Stroke: [{', '.join(str(point) for point in unwrapped_perception_node.normalized_coordinates)}]"
        elif isinstance(
            unwrapped_perception_node,
            (
                ContinuousNode,
                CategoricalNode,
                RgbColorNode,
                CielabColorNode,
                StrokeGNNRecognitionNode,
            ),
        ):
            label = str(unwrapped_perception_node)
        else:
            raise RuntimeError(
                f"Do not know how to perception node render node "
                f"{unwrapped_perception_node} with dot"
            )

        # if we are rendering a pattern which partially matched against a graph,
        # the user can supply IDs for those nodes which matched to show the correspondence.
        # We also make these nodes bold.
        mapping_id = match_correspondence_ids.get(perception_node)
        if mapping_id is not None:
            attributes = {
                LABEL: f"{label} [{mapping_id}]",
                "style": "filled",
                "fillcolor": "gray",
            }
        else:
            attributes = {LABEL: label, "style": "solid"}

        node_id = f"node-{next_node_id.value()}"
        next_node_id.increment()

        dot_graph.node(node_id, **attributes)

        return node_id

    def __repr__(self) -> str:
        return (
            f"PerceptionGraph(nodes={str_list_limited(self._graph.nodes, 10)}, edges="
            f"{str_list_limited(self._graph.edges(data=LABEL), 15)})"
        )

    def __len__(self) -> int:
        return len(self._graph)

    def __contains__(self, item) -> bool:
        return item in self._graph

    def __iter__(self):
        return iter(self._graph)

    def successors(self, graph_node: PerceptionGraphNode):
        return iter(self._graph.successors(graph_node))

    def __attrs_post_init__(self) -> None:
        # Every edge must have a label
        for (source, target, data_dict) in self._graph.edges(data=True):
            try:
                if LABEL in data_dict:
                    label_value = data_dict[LABEL]
                    if self.dynamic:
                        if isinstance(label_value, TemporallyScopedEdgeLabel):
                            assert_valid_edge_label(label_value.attribute)
                        else:
                            raise RuntimeError(
                                "In a dynamic graph, all edge labels must be "
                                "wrapped in TemporallyScopedEdgeLabel"
                            )
                    else:
                        if isinstance(label_value, TemporallyScopedEdgeLabel):
                            raise RuntimeError(
                                "TemporallyScopedEdgeLabels may not appear "
                                "in a static graph."
                            )
                        else:
                            assert_valid_edge_label(label_value)
                else:
                    raise RuntimeError(
                        "Every edge in a PerceptionGraph must have a 'label' attribute"
                    )
            except RuntimeError as e:
                raise RuntimeError(
                    f"Error validating PerceptionGraphEdge from {source} to {target} "
                    f"with attributes {data_dict}"
                ) from e


@attrs(frozen=True, slots=True, repr=False)
class PerceptionGraphPattern(PerceptionGraphProtocol, Sized, Iterable["NodePredicate"]):
    r"""
    A pattern which can match `PerceptionGraph`\ s.

    Such patterns could be used, for example, to represent a learner's
    knowledge of an object for object recognition.
    """

    _graph: DiGraph = attrib(validator=instance_of(DiGraph), converter=copy_digraph)
    dynamic: bool = attrib(validator=instance_of(bool), default=False)

    def _node_match(
        self, node1: Dict[str, "NodePredicate"], node2: Dict[str, "NodePredicate"]
    ) -> bool:
        return node1["node"].is_equivalent(node2["node"])
        # TODO: Compare node predicates with isEquivalent

    def _edge_match(
        self, edge1: Dict[str, "EdgePredicate"], edge2: Dict[str, "EdgePredicate"]
    ) -> bool:
        return edge1["predicate"].dot_label() == edge2["predicate"].dot_label()

    def check_isomorphism(self, other_graph: "PerceptionGraphPattern") -> bool:
        """
        Compares two pattern graphs and returns true if they are isomorphic, including edges and
        node attributes.
        """
        return is_isomorphic(
            self._graph,
            other_graph.copy_as_digraph(),
            node_match=self._node_match,
            edge_match=self._edge_match,
        )

    def matcher(
        self,
        graph_to_match_against: PerceptionGraphProtocol,
        *,
        debug_callback: Optional[DebugCallableType] = None,
        match_mode: MatchMode,
        allowed_matches: ImmutableSetMultiDict[Any, Any] = immutablesetmultidict(),
    ) -> "PatternMatching":
        """
        Creates an object representing an attempt to match this pattern
        against *graph_to_match_against*.
        """
        if graph_to_match_against.dynamic != self.dynamic:
            pattern_adjective = "dynamic" if self.dynamic else "static"
            graph_adjective = "dynamic" if graph_to_match_against.dynamic else "static"
            raise RuntimeError(
                f"Static patterns can only be applied to static graphs "
                f"and dynamic patterns to dynamic graphcs, "
                f"but tried to apply a {pattern_adjective} pattern to a "
                f"{graph_adjective} graph."
            )
        return PatternMatching(
            pattern=self,
            graph_to_match_against=graph_to_match_against,
            debug_callback=debug_callback,
            match_mode=match_mode,
            allowed_matches=allowed_matches,
        )

    @staticmethod
    def from_schema(
        object_schema: ObjectStructuralSchema,
        *,
        perception_generator: HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator,
        min_continuous_feature_match_score: float,
    ) -> "PerceptionGraphPattern":
        """
        Creates a pattern for recognizing an object based on its *object_schema*.
        """

        # First, we generate a PerceptionGraph corresponding to this schema
        # by instantiating a situation which contains a single object
        # of the desired type.
        schema_situation_object = SituationObject.instantiate_ontology_node(
            ontology_node=object_schema.ontology_node, ontology=GAILA_PHASE_1_ONTOLOGY
        )
        situation = HighLevelSemanticsSituation(
            ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[schema_situation_object]
        )
        # We explicitly exclude groundin perception generation, which were not
        # specified in the schema
        perception = perception_generator.generate_perception(
            situation, chooser=RandomChooser.for_seed(0), include_ground=False
        )
        perception_graph_as_digraph = PerceptionGraph.from_frame(
            first(perception.frames)
        ).copy_as_digraph()

        # We remove color and properties that are added by situation generation,
        # but are not in schemas.
        nodes_to_remove = [
            node
            for node in perception_graph_as_digraph
            # Perception generation wraps properties in tuples, so we need to unwrap them.
            if isinstance(node, tuple)
            and (
                isinstance(node[0], RgbColorPerception)
                or isinstance(node[0], OntologyNode)
            )
        ]
        perception_graph_as_digraph.remove_nodes_from(nodes_to_remove)

        # Finally, we convert the PerceptionGraph DiGraph representation to a PerceptionGraphPattern
        return PerceptionGraphPattern.from_graph(
            perception_graph=PerceptionGraph(perception_graph_as_digraph),
            min_continuous_feature_match_score=min_continuous_feature_match_score,
        ).perception_graph_pattern

    @staticmethod
    def from_graph(
        perception_graph: PerceptionGraph, *, min_continuous_feature_match_score: float
    ) -> "PerceptionGraphPatternFromGraph":
        """
        Creates a pattern for recognizing an object based on its *perception_graph*.
        """
        pattern_graph = DiGraph()
        perception_node_to_pattern_node: Dict[PerceptionGraphNode, NodePredicate] = {}
        PerceptionGraphPattern._translate_graph(
            perception_graph=perception_graph.copy_as_digraph(),
            pattern_graph=pattern_graph,
            perception_node_to_pattern_node=perception_node_to_pattern_node,
            min_continuous_feature_match_score=min_continuous_feature_match_score,
        )
        return PerceptionGraphPatternFromGraph(
            perception_graph_pattern=PerceptionGraphPattern(
                pattern_graph, dynamic=perception_graph.dynamic
            ),
            perception_graph_node_to_pattern_node=perception_node_to_pattern_node,
        )

    @staticmethod
    def from_ontology_node(
        node: OntologyNode,
        ontology: Ontology,
        *,
        perception_generator: HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator,
        min_continuous_feature_match_score: float,
    ) -> "PerceptionGraphPattern":
        """
        Creates a pattern for recognizing an obect based on its *ontology_node*
        """
        # First, we check to see if the ontology node has a corresponding
        # schema in the ontology. If so we just use `from_schema`
        if perception_generator.ontology != ontology:
            raise RuntimeError(
                "Ontology of perception generator does not match ontology of "
                "from_ontology_node"
            )

        if (
            node
            in ontology._structural_schemata.keys()  # pylint:disable=protected-access
        ):
            return PerceptionGraphPattern.from_schema(
                first(ontology.structural_schemata(node)),
                perception_generator=perception_generator,
                min_continuous_feature_match_score=min_continuous_feature_match_score,
            )
        # If the node doesn't have a corresponding structural schemata we see if it can be
        # created as a single object scene
        schema_situation_object = SituationObject.instantiate_ontology_node(
            ontology_node=node,
            ontology=ontology,
            properties=node.non_inheritable_properties,
        )
        situation = HighLevelSemanticsSituation(
            ontology=ontology, salient_objects=[schema_situation_object]
        )
        # We explicitly exclude ground in perception generation
        perception = perception_generator.generate_perception(
            situation, chooser=RandomChooser.for_seed(0), include_ground=False
        )
        perception_graph_as_digraph = PerceptionGraph.from_frame(
            first(perception.frames)
        ).copy_as_digraph()

        nodes_to_remove = [
            node
            for node in perception_graph_as_digraph
            # Perception generation wraps properties in tuples, so we need to unwrap them.
            if isinstance(node, tuple)
            and isinstance(node[0], OntologyNode)
            and not (
                ontology.is_subtype_of(node[0], RECOGNIZED_PARTICULAR_PROPERTY)
                or node[0] is LIQUID
                # hack for me/you
                or node[0] is IS_SPEAKER
                or node[0] is IS_ADDRESSEE
            )
        ]
        perception_graph_as_digraph.remove_nodes_from(nodes_to_remove)

        # We then turn this DiGraph representation into a PerceptionGraphPattern
        return PerceptionGraphPattern.from_graph(
            perception_graph=PerceptionGraph(perception_graph_as_digraph),
            min_continuous_feature_match_score=min_continuous_feature_match_score,
        ).perception_graph_pattern

    @staticmethod
    def phase3_pattern(
        node: OntologyNode, *, min_continuous_feature_match_score: float
    ) -> "PerceptionGraphPattern":
        digraph = DiGraph()

        fake_cluster = ObjectClusterNode(
            center_y=0.0,
            center_x=0.0,
            cluster_id="fake_cluster",
            std=0.0,
            viewpoint_id=0,
            weight=1.0,
        )
        fake_gnn_recognition = StrokeGNNRecognitionNode(
            object_recognized=node.handle, confidence=1.0, weight=1.0
        )
        digraph.add_edge(fake_cluster, fake_gnn_recognition, label=HAS_PROPERTY_LABEL)

        return PerceptionGraphPattern.from_graph(
            perception_graph=PerceptionGraph(digraph, dynamic=False),
            min_continuous_feature_match_score=min_continuous_feature_match_score,
        ).perception_graph_pattern

    def pattern_complexity(self) -> int:
        return len(self._graph) + len(self._graph.edges)

    def __len__(self) -> int:
        return len(self._graph)

    def __contains__(self, item) -> bool:
        return item in self._graph

    def __iter__(self):
        return iter(self._graph)

    def copy_with_temporal_scopes(
        self, required_temporal_scopes: Union[TemporalScope, Iterable[TemporalScope]]
    ) -> "PerceptionGraphPattern":
        r"""
        Produces a copy of this perception graph pattern
        where all edge predicates now require that the edge in the target graph being matched
        hold at all of the *required_temporal_scopes*.

        The new pattern will be dynamic.

        This pattern must be a static graph or a `RuntimeError` will be raised.
        """
        if self.dynamic:
            raise RuntimeError(
                "Cannot use copy_with_temporal_scopes on a pattern which is already dynamic"
            )

        wrapped_graph = self.copy_as_digraph()

        # For convenience we allow the user to specify only a single temporal scope
        # instead of a collection.
        if isinstance(required_temporal_scopes, TemporalScope):
            required_temporal_scopes = [required_temporal_scopes]

        for (source, target) in wrapped_graph.edges():
            unwrapped_predicate = wrapped_graph.edges[source, target]["predicate"]
            temporally_scoped_predicate = HoldsAtTemporalScopePredicate(
                unwrapped_predicate, required_temporal_scopes
            )
            wrapped_graph.edges[source, target]["predicate"] = temporally_scoped_predicate

        return PerceptionGraphPattern(dynamic=True, graph=wrapped_graph)

    def copy_removing_temporal_scopes(self) -> "PerceptionGraphPattern":
        """Produces a copy of this perception graph pattern
        where all edge predicates have their temporal scope removed.

        The new pattern will be static and MAY be nonsensical."""

        unwrapped_graph = self.copy_as_digraph()

        for (source, target) in unwrapped_graph.edges():
            wrapped_predicate = unwrapped_graph.edges[source, target]["predicate"]
            unwrapped_graph.edges[source, target][
                "predicate"
            ] = wrapped_predicate.wrapped_edge_predicate

        return PerceptionGraphPattern(dynamic=False, graph=unwrapped_graph)

    def count_nodes_matching(
        self, node_predicate: Callable[[NodePredicate], bool]
    ) -> int:
        return ilen(filter(node_predicate, self._graph.nodes))  # type: ignore

    def copy_replacing_nodes(
        self, current_to_new_node: Mapping[NodePredicate, NodePredicate]
    ) -> "PerceptionGraphPattern":
        digraph = DiGraph()
        digraph.add_nodes_from(
            current_to_new_node[node] if node in current_to_new_node else node
            for node in self._graph.nodes
        )
        for u, v, data in self._graph.edges.data():
            digraph.add_edge(
                current_to_new_node[u] if u in current_to_new_node else u,
                current_to_new_node[v] if v in current_to_new_node else v,
                **data,
            )

        return PerceptionGraphPattern(graph=digraph, dynamic=self.dynamic)

    @staticmethod
    def _translate_graph(
        perception_graph: DiGraph,
        pattern_graph: DiGraph,
        *,
        perception_node_to_pattern_node: Dict[Any, "NodePredicate"],
        min_continuous_feature_match_score: float,
    ) -> None:
        # Two mapping methods that map nodes and edges from the source PerceptionGraph onto the corresponding
        # node and edge representations on the PerceptionGraphPattern.
        def map_node(node: Any) -> "NodePredicate":

            key = node
            if key not in perception_node_to_pattern_node:
                # first unwrap any nodes which have been tuple-ized to force distinctness.
                if isinstance(node, tuple):
                    node = node[0]
                if isinstance(node, Geon):
                    perception_node_to_pattern_node[key] = GeonPredicate.exactly_matching(
                        node
                    )
                elif isinstance(node, CrossSection):
                    perception_node_to_pattern_node[
                        key
                    ] = CrossSectionPredicate.exactly_matching(node)
                elif isinstance(node, Region):
                    perception_node_to_pattern_node[
                        key
                    ] = RegionPredicate.matching_distance(node)
                elif isinstance(node, GeonAxis):
                    perception_node_to_pattern_node[key] = AxisPredicate.from_axis(node)
                elif isinstance(node, ObjectPerception):
                    perception_node_to_pattern_node[key] = AnyObjectPerception(
                        debug_handle=node.debug_handle
                    )
                elif isinstance(node, OntologyNode):
                    perception_node_to_pattern_node[key] = IsOntologyNodePredicate(node)
                elif isinstance(node, RgbColorPerception):
                    perception_node_to_pattern_node[key] = IsColorNodePredicate(node)
                elif isinstance(node, ObjectSemanticNode):
                    perception_node_to_pattern_node[
                        key
                    ] = ObjectSemanticNodePerceptionPredicate()
                elif isinstance(node, SpatialPath):
                    perception_node_to_pattern_node[key] = IsPathPredicate()
                elif isinstance(node, PathOperator):
                    perception_node_to_pattern_node[key] = PathOperatorPredicate(node)
                elif isinstance(node, ObjectClusterNode):
                    perception_node_to_pattern_node[key] = AnyObjectPredicate()
                elif isinstance(node, CategoricalNode):
                    perception_node_to_pattern_node[key] = CategoricalPredicate.from_node(
                        node
                    )
                elif isinstance(node, ContinuousNode):
                    perception_node_to_pattern_node[
                        key
                    ] = DistributionalContinuousPredicate.from_node(
                        node, min_match_score=min_continuous_feature_match_score
                    )
                elif isinstance(node, RgbColorNode):
                    perception_node_to_pattern_node[key] = RgbColorPredicate.from_node(
                        node
                    )
                elif isinstance(node, CielabColorNode):
                    perception_node_to_pattern_node[key] = CielabColorPredicate.from_node(
                        node, min_match_score=min_continuous_feature_match_score
                    )
                elif isinstance(node, ObjectStroke):
                    perception_node_to_pattern_node[
                        key
                    ] = ObjectStrokePredicate.from_node(node)
                elif isinstance(node, StrokeGNNRecognitionNode):
                    perception_node_to_pattern_node[key] = StrokeGNNRecognitionPredicate(
                        recognized_object=node.object_recognized
                    )
                elif isinstance(node, TrajectoryRecognitionNode):
                    perception_node_to_pattern_node[key] = TrajectoryRecognitionPredicate(
                        recognized_action=node.action_recognized
                    )
                elif isinstance(node, JointPointNode):
                    perception_node_to_pattern_node[key] = JointPointPredicate(
                        temporal_index=node.temporal_index, joint_index=node.joint_index
                    )
                else:
                    raise RuntimeError(f"Don't know how to map node {node}")
            return perception_node_to_pattern_node[key]

        def map_edge(label: Any) -> Mapping[str, "EdgePredicate"]:
            if isinstance(label, OntologyNode):
                return {"predicate": RelationTypeIsPredicate(label)}
            elif isinstance(label, Direction):
                return {"predicate": DirectionPredicate.exactly_matching(label)}
            elif isinstance(label, TemporallyScopedEdgeLabel):
                scopes = label.temporal_specifiers
                non_temporal_predicate = map_edge(label.attribute)["predicate"]
                return {
                    "predicate": HoldsAtTemporalScopePredicate(
                        wrapped_edge_predicate=non_temporal_predicate,
                        temporal_scopes=scopes,
                    )
                }
            else:
                raise RuntimeError(f"Cannot map edge {label}")

        # We add every node in the source graph, after translating their type with map_node .
        for original_node in perception_graph.nodes:
            # Add each node
            pattern_node = map_node(original_node)
            pattern_graph.add_node(pattern_node)
            set_node_attributes(pattern_graph, {pattern_node: {"node": pattern_node}})

        # Once all nodes are translated, we add all edges from the source graph by iterating over each node and
        # extracting its edges.
        for original_node in perception_graph.nodes:
            edges_from_node = perception_graph.edges(original_node, data=True)
            for (_, original_dest_node, original_edge_data) in edges_from_node:
                pattern_from_node = perception_node_to_pattern_node[original_node]
                pattern_to_node = perception_node_to_pattern_node[original_dest_node]
                pattern_graph.add_edge(pattern_from_node, pattern_to_node)
                mapped_edge = map_edge(original_edge_data[LABEL])
                pattern_graph.edges[pattern_from_node, pattern_to_node].update(
                    mapped_edge
                )

    def render_to_file(  # pragma: no cover
        self,
        graph_name: str,
        output_file: Path,
        *,
        match_correspondence_ids: Mapping[Any, str] = immutabledict(),
        robust=True,
        replace_node_labels: Mapping[Any, str] = immutabledict(),
    ) -> None:
        """
        Debugging tool to render the pattern to PDF using *dot*.

        If this pattern has been matched against a `PerceptionGraph`,
        the matched nodes can be highlighted and given labels which show
        what the correspond to in a pattern by supplying
        *match_correspondence_ids* which maps graph nodes to
        the desired correspondence labels.
        """

        dot_graph = graphviz.Digraph(graph_name)
        dot_graph.attr(rankdir="LR")

        next_node_id = Incrementer()

        def to_dot_node(pattern_node: "NodePredicate") -> str:
            node_id = f"node-{next_node_id.value()}"
            next_node_id.increment()
            if pattern_node in replace_node_labels:
                base_label = replace_node_labels[pattern_node]
            else:
                base_label = pattern_node.dot_label()

            # if we are rendering a match against another graph,
            # we show IDs that align the nodes between the graphs
            # and make the nodes which have matches bold.
            correspondence_id = match_correspondence_ids.get(pattern_node)
            if correspondence_id is not None:
                attributes = {
                    LABEL: f"{base_label} [{correspondence_id}]",
                    "style": "filled",
                    "fillcolor": "gray",
                }
            else:
                attributes = {LABEL: base_label, "style": "solid"}

            dot_graph.node(node_id, **attributes)
            return node_id

        pattern_nodes_to_dot_node_ids = {
            pattern_node: to_dot_node(pattern_node) for pattern_node in self._graph.nodes
        }

        for (source_node, target_node, predicate) in self._graph.edges.data("predicate"):
            source_dot_node = pattern_nodes_to_dot_node_ids[source_node]
            target_dot_node = pattern_nodes_to_dot_node_ids[target_node]

            if not predicate.reverse_in_dot_graph():
                dot_graph.edge(source_dot_node, target_dot_node, predicate.dot_label())
            else:
                dot_graph.edge(
                    target_dot_node, source_dot_node, predicate.dot_label(), back="true"
                )

        try:
            dot_graph.render(str(output_file))
        except Exception as e:  # pylint:disable=broad-except
            if robust:
                logging.warning("Error during dot rendering: %s", e)
            else:
                raise

    def intersection(
        self,
        graph_pattern: "PerceptionGraphPattern",
        *,
        debug_callback: Optional[DebugCallableType] = None,
        graph_logger: Optional["GraphLogger"] = None,
        ontology: Ontology,
        allowed_matches: ImmutableSetMultiDict[Any, Any] = immutablesetmultidict(),
        match_mode: MatchMode,
        trim_after_match: Optional[
            Callable[["PerceptionGraphPattern"], "PerceptionGraphPattern"]
        ] = None,
    ) -> Optional["PerceptionGraphPattern"]:
        """
        Determine the largest partial match between two `PerceptionGraphPattern`s

        The algorithm used is approximate and is not guaranteed to return the largest
        possible match.
        """
        attempted_match = self.intersection_getting_match(
            graph_pattern,
            debug_callback=debug_callback,
            graph_logger=graph_logger,
            ontology=ontology,
            allowed_matches=allowed_matches,
            match_mode=match_mode,
            trim_after_match=trim_after_match,
        )
        return attempted_match.matched_pattern if attempted_match else None

    def intersection_getting_match(
        self,
        graph_pattern,
        *,
        debug_callback: Optional[DebugCallableType] = None,
        graph_logger: Optional["GraphLogger"] = None,
        ontology: Ontology,
        allowed_matches: ImmutableSetMultiDict[Any, Any] = immutablesetmultidict(),
        match_mode: MatchMode,
        trim_after_match: Optional[
            Callable[["PerceptionGraphPattern"], "PerceptionGraphPattern"]
        ] = None,
    ) -> Optional["PerceptionGraphPatternMatch"]:
        matcher = PatternMatching(
            pattern=graph_pattern,
            graph_to_match_against=self,
            debug_callback=debug_callback,
            matching_pattern_against_pattern=True,
            match_mode=match_mode,
            allowed_matches=allowed_matches,
        )
        return matcher.relax_pattern_until_it_matches_getting_match(
            graph_logger=graph_logger,
            ontology=ontology,
            trim_after_match=trim_after_match,
        )

    def __repr__(self) -> str:
        return (
            f"PerceptionGraphPattern(nodes={str_list_limited(self._graph.nodes, 10)}, "
            f"edges="
            f"{str_list_limited(self._graph.edges(data='predicate'), 10)})"
        )


class DumpPartialMatchCallback:
    """
    Helper callable class for debugging purposes. An instance of this object can be provided as the `debug_callback` argument of `GraphMatching.match` to render
    the match search process at every 100 time steps. We start rendering after the first 60 seconds.
    """

    def __init__(
        self,
        render_path,
        seconds_to_wait_before_rendering: int = 0,
        dump_every_x_calls: int = 100,
    ) -> None:
        self.render_path = render_path
        self.calls_to_match_counter = 0
        self.start_time = process_time()
        self.seconds_to_wait_before_rendering = seconds_to_wait_before_rendering
        self.dump_every_x_calls = dump_every_x_calls

    def __call__(
        self, graph: DiGraph, graph_node_to_pattern_node: Dict[Any, Any]
    ) -> None:
        self.calls_to_match_counter += 1
        current_time = process_time()
        if (
            self.calls_to_match_counter % self.dump_every_x_calls == 0
            and (current_time - self.start_time) > self.seconds_to_wait_before_rendering
        ):
            perception_graph = PerceptionGraph(graph)
            title = (
                "id_"
                + str(id(self))
                + "_graph_"
                + str(id(graph))
                + "_call_"
                + str(self.calls_to_match_counter).zfill(4)
            )
            mapping = {k: "match" for k, v in graph_node_to_pattern_node.items()}
            perception_graph.render_to_file(
                graph_name=title,
                output_file=Path(self.render_path + title),
                match_correspondence_ids=mapping,
            )


@attrs(frozen=True, slots=True, auto_attribs=True)
class PerceptionGraphPatternFromGraph:
    """
    See `PerceptionGraphPattern.from_graph`
    """

    perception_graph_pattern: PerceptionGraphPattern
    perception_graph_node_to_pattern_node: ImmutableDict[
        PerceptionGraphNode, "NodePredicate"
    ] = attrib(converter=_to_immutabledict)


@attrs(slots=True, eq=False)
class PatternMatching:
    """
    An attempt to align a `PerceptionGraphPattern` to nodes in a `PerceptionGraph`.

    This is equivalent to finding a sub-graph of *graph_to_match*
    which is isomorphic to *pattern*.
    Currently we only handle node-induced sub-graph isomorphisms,
    but we might really want edge-induced: https://github.com/isi-vista/adam/issues/400
    """

    pattern: PerceptionGraphPattern = attrib(
        validator=instance_of(PerceptionGraphPattern)
    )
    graph_to_match_against: PerceptionGraphProtocol = attrib(
        validator=instance_of(PerceptionGraphProtocol)
    )
    # the attrs mypy plugin complains for the below
    # "Non-default attributes not allowed after default attributes."
    # But that doesn't seem to be our situation? And it works fine?
    matching_pattern_against_pattern: bool = attrib(validator=instance_of(bool))
    _match_mode: MatchMode = attrib(  # type: ignore
        validator=instance_of(MatchMode), kw_only=MatchMode.OBJECT
    )

    # Callable object for debugging purposes. We use this to track the number of calls to match and render the graphs.
    debug_callback: Optional[DebugCallableType] = attrib(default=None, kw_only=True)

    allowed_matches: ImmutableSetMultiDict[Any, Any] = attrib(
        validator=instance_of(ImmutableSetMultiDict),
        kw_only=True,
        default=immutablesetmultidict(),
    )

    @attrs(frozen=True, kw_only=True, auto_attribs=True)
    class MatchFailure:
        """
        Indicates a failed attempt at matching a `PerceptionGraphPattern`.

        *pattern_node_to_graph_node_for_largest_match* indicates the partial match found
        with the largest number of nodes.
        Note that this is not necessarily the largest possible partial match,
        just the largest one encountered during the search process
        before the algorithm decided that a full match was not possible.

        *last_failed_pattern_node* is that the last pattern node attempted to be matched
        at the point the algorithm decided a match was impossible.
        Note that this is not necessarily the node responsible for the match failure;
        it could fail due to the edge predicate on the connecting edge,
        or it could fail due to there being no proper match for a pattern node
        which had no good alignment earlier in the match process.
        """

        pattern: PerceptionGraphPattern
        graph: PerceptionGraphProtocol
        pattern_node_to_graph_node_for_largest_match: Mapping[Any, Any]
        last_failed_pattern_node: "NodePredicate"
        largest_match_pattern_subgraph: PerceptionGraphPattern = attrib()
        # TODO: the below is just a DiGraph because these is currently overloaded
        # to return match failures for both pattern-perception graph
        # and pattern-pattern matches.
        # It can be made a PerceptionGraph or PerceptionGraphPattern pending
        # https://github.com/isi-vista/adam/issues/489
        largest_match_graph_subgraph: DiGraph = attrib(validator=instance_of(DiGraph))

        def __attrs_post_init__(self) -> None:
            if (
                self.last_failed_pattern_node
                not in self.pattern._graph.nodes  # pylint:disable=protected-access
            ):

                raise RuntimeError(
                    f"Something has gone wrong: the pattern "
                    f"does not contain the failed node:"
                    f"{self.last_failed_pattern_node}"
                )

            if set(self.pattern_node_to_graph_node_for_largest_match.keys()) != set(
                self.largest_match_pattern_subgraph._graph.nodes  # pylint:disable=protected-access
            ):
                raise RuntimeError(
                    "Mismatch between node alignment and largest partial " "pattern match"
                )

            if set(self.pattern_node_to_graph_node_for_largest_match.values()) != set(
                self.largest_match_graph_subgraph.nodes
            ):
                raise RuntimeError(
                    "Mismatch between node alignment and largest partial graph " "match"
                )

        @largest_match_pattern_subgraph.default  # noqa: F821
        def _matched_pattern_subgraph_default(self) -> DiGraph:
            return PerceptionGraphPattern(
                self.pattern._graph.subgraph(  # pylint:disable=protected-access
                    self.pattern_node_to_graph_node_for_largest_match.keys()
                )
            )

        @largest_match_graph_subgraph.default  # noqa: F821
        def _matched_graph_subgraph_default(self) -> DiGraph:
            return self.graph._graph.subgraph(  # pylint:disable=protected-access
                immutableset(self.pattern_node_to_graph_node_for_largest_match.values())
            )

    def matches(
        self,
        *,
        use_lookahead_pruning: bool,
        suppress_multiple_alignments_to_same_nodes: bool = True,
        initial_partial_match: Mapping[Any, Any] = immutabledict(),
        graph_logger: Optional["GraphLogger"] = None,
    ) -> Iterable["PerceptionGraphPatternMatch"]:
        """
        Attempt the matching and returns a generator over the set of possible matches.

        If *suppress_multiple_alignments_to_same_nodes* is *True* (default *True*),
        then only the first alignment encountered for a given set of nodes will be returned.
        This prevents you from e.g. getting multiple matches for different ways
        of aligning axes for symmetric objects.
        The cost is that we need to keep around a memory of previous node matches.

        *matching_pattern_against_pattern* indicates
        you are matching one pattern against another,
        rather than against a perception graph.
        This should get split off into its own distinct method:
        https://github.com/isi-vista/adam/issues/487
        """
        for match in self._internal_matches(
            graph_to_match_against=self.graph_to_match_against,
            pattern=self.pattern,
            debug_callback=self.debug_callback,
            use_lookahead_pruning=use_lookahead_pruning,
            suppress_multiple_alignments_to_same_nodes=suppress_multiple_alignments_to_same_nodes,
            initial_partial_match=initial_partial_match,
            graph_logger=graph_logger,
        ):
            # we want to ignore the failure objects returned for the use of
            # first_match_or_failure_info
            if isinstance(match, PerceptionGraphPatternMatch):
                yield match

    def first_match_or_failure_info(
        self,
        *,
        initial_partial_match: Mapping[Any, Any] = immutabledict(),
        graph_logger: Optional["GraphLogger"] = None,
    ) -> Union["PerceptionGraphPatternMatch", "PatternMatching.MatchFailure"]:
        """
        Gets the first match encountered of the pattern against the graph
        (which one is first is deterministic but undefined)
        or a `PatternMatching.MatchFailure` giving debugging information
        about a failed match attempt.
        """
        return first(
            self._internal_matches(
                graph_to_match_against=self.graph_to_match_against,
                pattern=self.pattern,
                debug_callback=self.debug_callback,
                # lookahead pruning messes up the debug information by making the
                # cause of failures appear "earlier" in the pattern than they really are.
                use_lookahead_pruning=False,
                suppress_multiple_alignments_to_same_nodes=True,
                initial_partial_match=initial_partial_match,
                graph_logger=graph_logger,
            )
        )

    def relax_pattern_until_it_matches(
        self,
        *,
        graph_logger: Optional["GraphLogger"] = None,
        ontology: Ontology,
        min_ratio: Optional[float] = None,
        trim_after_match: Optional[
            Callable[[PerceptionGraphPattern], PerceptionGraphPattern]
        ],
    ) -> Optional[PerceptionGraphPattern]:
        """
        Prunes or relaxes the *pattern* for this matching until it successfully matches
        using heuristic rules.

        If a matching relaxed `PerceptionGraphPattern` can be found, it is returned.
        Otherwise, *None* is returned.
        """
        match = self.relax_pattern_until_it_matches_getting_match(
            graph_logger=graph_logger,
            ontology=ontology,
            min_ratio=min_ratio,
            trim_after_match=trim_after_match,
        )
        return (
            match.matched_pattern
            if isinstance(match, PerceptionGraphPatternMatch)
            else None
        )

    def relax_pattern_until_it_matches_getting_match(
        self,
        *,
        graph_logger: Optional["GraphLogger"] = None,
        ontology: Ontology,
        min_ratio: Optional[float] = None,
        trim_after_match: Optional[
            Callable[[PerceptionGraphPattern], PerceptionGraphPattern]
        ],
    ) -> Optional["PerceptionGraphPatternMatch"]:
        """
        Prunes or relaxes the *pattern* for this matching until it successfully matches
        using heuristic rules.

        If a matching relaxed `PerceptionGraphPattern` can be found, the corresponding
        `PerceptionGraphPatternMatch` is returned. Otherwise, *None* is returned.
        """

        min_num_nodes_to_continue = 1
        if min_ratio:
            check_arg(min_ratio in Range.open_closed(0.0, 1.0))
            min_num_nodes_to_continue = int(min_ratio * len(self.pattern))

        # We start with the original pattern and attempt to match progressively relaxed versions.
        cur_pattern: Optional[PerceptionGraphPattern] = self.pattern
        # When we try to match relaxed patterns, we will remember how much we could match
        # last time so we don't start from scratch.
        partial_match: Mapping[Any, Any] = {}

        relaxation_step = 0
        while (
            cur_pattern and len(cur_pattern) >= min_num_nodes_to_continue
        ):  # pylint:disable=len-as-condition
            match_attempt = first(
                self._internal_matches(
                    graph_to_match_against=self.graph_to_match_against,
                    pattern=cur_pattern,
                    debug_callback=self.debug_callback,
                    # Using lookahead pruning would make our guess at the "cause"
                    # of the match failure be too "early" in the pattern graph search.
                    use_lookahead_pruning=False,
                    suppress_multiple_alignments_to_same_nodes=True,
                    initial_partial_match=partial_match,
                )
            )
            if isinstance(match_attempt, PerceptionGraphPatternMatch):
                return match_attempt
            else:
                relaxation_step += 1
                # If we couldn't successfully match the current part of the pattern,
                # chop off the node which we think might have caused the match to fail.
                # Why is this cast necessary? mypy should be able to infer this...
                cur_pattern = self._relax_pattern(
                    match_attempt, graph_logger=graph_logger, ontology=ontology
                )
                if trim_after_match and cur_pattern:
                    cur_pattern = trim_after_match(cur_pattern)
                if graph_logger and cur_pattern:
                    graph_logger.log_graph(
                        cur_pattern,
                        logging.INFO,
                        "Relaxation step %s, cur pattern size %s",
                        relaxation_step,
                        len(cur_pattern),
                    )
        # no relaxation could successfully match
        return None

    def _internal_matches(
        self,
        *,
        graph_to_match_against: PerceptionGraphProtocol,
        pattern: PerceptionGraphPattern,
        debug_callback: Optional[Callable[[Any, Any], None]],
        use_lookahead_pruning: bool,
        suppress_multiple_alignments_to_same_nodes: bool = True,
        collect_debug_statistics: bool = False,
        initial_partial_match: Mapping[Any, Any] = immutabledict(),
        graph_logger: Optional["GraphLogger"] = None,
    ) -> Iterable[Union["PerceptionGraphPatternMatch", "PatternMatching.MatchFailure"]]:
        # Controlling the iteration order of the graphs
        # controls the order in which nodes are matched.
        # This has a significant, benchmark-confirmed impact on performance.
        sorted_graph_to_match_against = digraph_with_nodes_sorted_by(
            graph_to_match_against._graph,  # pylint: disable=W0212
            _graph_node_order
            if not self.matching_pattern_against_pattern
            else _pattern_matching_node_order,
        )
        sorted_pattern = digraph_with_nodes_sorted_by(
            pattern._graph, _pattern_matching_node_order  # pylint: disable=W0212
        )

        matching = GraphMatching(
            sorted_graph_to_match_against,
            sorted_pattern,
            use_lookahead_pruning=use_lookahead_pruning,
            matching_pattern_against_pattern=self.matching_pattern_against_pattern,
            match_mode=self._match_mode,
        )

        sets_of_nodes_matched: Set[ImmutableSet[PerceptionGraphNode]] = set()

        got_a_match = False
        if debug_callback:
            self.debug_callback = debug_callback

        allowed_matches = self.allowed_matches
        logging.debug("PatternMatcher code... Allowed matches were %s", allowed_matches)

        for node in initial_partial_match.keys():
            if (
                node in allowed_matches
                and initial_partial_match[node] not in allowed_matches[node]
            ):
                raise RuntimeError(
                    "Initial partial match is not compatible with set of allowed matches!"
                )

        # Our initial partial match may already match up some of our nodes that have matching
        # restrictions (as given by allowed_matches). In that case, we don't need to iterate over
        # the different allowed ways to match up those nodes, so we remove them from our collection
        # of allowed matches.
        allowed_matches = immutablesetmultidict(
            (node, allowed_match)
            for node, allowed_match in allowed_matches.items()
            if node not in initial_partial_match
        )

        # We want to iterate over all possible initial matchings, which means taking the product of
        # the sequences of allowed pairings for each node in allowed_matches.
        logging.debug(
            "PatternMatcher code... After filtering, allowed matches were %s",
            allowed_matches,
        )
        for allowed_matching in product(
            *[zip(repeat(node), allowed_matches[node]) for node in allowed_matches.keys()]
        ):
            logging.debug(
                "PatternMatcher code... Allowed matching was %s", allowed_matching
            )
            allowed_matching = immutabledict(allowed_matching)
            merged_initial_partial_match: Mapping[Any, Any] = immutabledict(
                chain(allowed_matching.items(), initial_partial_match.items())
            )

            for (
                graph_node_to_matching_pattern_node
            ) in matching.subgraph_monomorphisms_iter(
                collect_debug_statistics=collect_debug_statistics,
                debug_callback=debug_callback,
                initial_partial_match=merged_initial_partial_match,
            ):
                matched_graph_nodes: ImmutableSet[PerceptionGraphNode] = immutableset(
                    graph_node_to_matching_pattern_node
                )
                if (
                    matched_graph_nodes not in sets_of_nodes_matched
                    or not suppress_multiple_alignments_to_same_nodes
                ):
                    got_a_match = True

                    matched_subgraph_digraph = matching.graph.subgraph(
                        graph_node_to_matching_pattern_node.keys()
                    ).copy()
                    matched_subgraph_dynamic = graph_to_match_against.dynamic

                    matched_subgraph: PerceptionGraphProtocol
                    if isinstance(graph_to_match_against, PerceptionGraph):
                        matched_subgraph = PerceptionGraph(
                            graph=matched_subgraph_digraph,
                            dynamic=matched_subgraph_dynamic,
                        )
                    elif isinstance(graph_to_match_against, PerceptionGraphPattern):
                        matched_subgraph = PerceptionGraphPattern(
                            graph=matched_subgraph_digraph,
                            dynamic=matched_subgraph_dynamic,
                        )
                    else:
                        raise RuntimeError(
                            f"Can only match against PerceptionGraphs or "
                            f"PerceptionGraphPatterns but got a "
                            f"{type(graph_to_match_against)}"
                        )

                    yield PerceptionGraphPatternMatch(
                        graph_matched_against=graph_to_match_against,
                        matched_pattern=pattern,
                        # mypy doesn't like
                        matched_sub_graph=matched_subgraph,
                        pattern_node_to_matched_graph_node=_invert_to_immutabledict(
                            graph_node_to_matching_pattern_node
                        ),
                    )
                sets_of_nodes_matched.add(matched_graph_nodes)
        if not got_a_match:
            # mypy doesn't like the assignment to the pattern_node_to_graph_node_for_largest_match
            # argument for reasons which are unclear to me. It works fine, though.
            match_failure = PatternMatching.MatchFailure(  # type: ignore
                pattern=pattern,
                graph=graph_to_match_against,
                pattern_node_to_graph_node_for_largest_match=immutabledict(
                    matching.debug_largest_match
                ),
                last_failed_pattern_node=cast(
                    NodePredicate, matching.failing_pattern_node_for_deepest_match
                ),
            )
            if graph_logger:
                graph_logger.log_match_failure(
                    match_failure, logging.INFO, "Match failure"
                )
            yield match_failure

    def _relax_pattern(
        self,
        match_failure: "PatternMatching.MatchFailure",
        ontology: Ontology,
        *,
        graph_logger: Optional["GraphLogger"] = None,
    ) -> Optional[PerceptionGraphPattern]:
        """
        Attempts to produce a "relaxed" version of pattern
        which has a better chance of matching,
        given the clue that *last_failed_pattern_node* was the last pattern node
        which could not be aligned in the previous pattern alignment attempt.

        This is to support `relax_pattern_until_it_matches`.
        """
        if len(match_failure.pattern) == 1:
            # It's the end of the line: we've pruned out all the pattern nodes.
            return None

        pattern_as_digraph = match_failure.pattern.copy_as_digraph()

        # first, we delete the last_failed_pattern_node.
        # If that node was an object, we also recursively delete its sub-objects
        # and any Regions it or they belonged to.
        nodes_to_delete_directly: List[NodePredicate] = []

        RELATION_EDGES_TO_FOLLOW_WHEN_DELETING = {  # pylint:disable=invalid-name
            PART_OF,
            # IN_REGION,
        }

        def gather_nodes_to_excise(focus_node: NodePredicate) -> None:
            if focus_node in match_failure.pattern_node_to_graph_node_for_largest_match:
                # don't delete or continue deleting through a node which successfully matched
                return
            nodes_to_delete_directly.append(focus_node)

            # If this is an object, also excise any sub-objects.
            for predecessor in pattern_as_digraph.pred[focus_node]:
                edge_label = pattern_as_digraph.edges[predecessor, focus_node][
                    "predicate"
                ]
                if (
                    isinstance(edge_label, RelationTypeIsPredicate)
                    and edge_label.relation_type in RELATION_EDGES_TO_FOLLOW_WHEN_DELETING
                ):
                    gather_nodes_to_excise(predecessor)

        last_failed_node = match_failure.last_failed_pattern_node
        logging.info("Relaxation: last failed pattern node is %s", last_failed_node)

        if last_failed_node in match_failure.pattern_node_to_graph_node_for_largest_match:
            # This means the supposed "last_failed_node" was in fact matched successfully,
            # so the match failure must be because all alignments found were judged
            # illegal. In that case, no relaxation can fix things.
            logging.info("Match found is illegal; no relaxation can help")
            return None

        gather_nodes_to_excise(last_failed_node)

        same_color_nodes: List[NodePredicate]

        # We only want to gather all the color nodes when we are trying to match an object
        # Otherwise we can accidentally remove important information for the pattern
        if (
            isinstance(last_failed_node, IsColorNodePredicate)
            and self._match_mode == MatchMode.OBJECT
        ):
            # We treat colors as a special case.
            # In a complex object (e.g. a dog), an object and a large number of its
            # sub-components will share the same color.
            # Usually if the color is irrelevant to a pattern in one place,
            # it is irrelevant in many other places.
            # Relaxing the pattern by peeling the color predicates off one by one can be very slow,
            # especially since such large patterns are slow to match in the first place.
            # Therefore, if we remove a color predicate in one place, we remove it
            # from the entire pattern.
            color = last_failed_node.color

            same_color_nodes = [
                node
                for node in pattern_as_digraph.nodes
                if isinstance(node, IsColorNodePredicate) and node.color == color
            ]
        elif isinstance(
            last_failed_node, IsOntologyNodePredicate
        ) and ontology.is_subtype_of(last_failed_node.property_value, COLOR):
            # same as above, but for the case where we are perceiving colors discretely.
            discrete_color = last_failed_node.property_value

            same_color_nodes = [
                node
                for node in pattern_as_digraph.nodes
                if isinstance(node, IsOntologyNodePredicate)
                and node.property_value == discrete_color
            ]
        else:
            same_color_nodes = []

        if same_color_nodes:
            logging.info("Deleting extra color nodes: %s", same_color_nodes)
        nodes_to_delete_directly.extend(same_color_nodes)

        logging.info("Nodes to delete directly: %s", nodes_to_delete_directly)

        pattern_as_digraph.remove_nodes_from(immutableset(nodes_to_delete_directly))

        if len(pattern_as_digraph) == 0:  # pylint:disable=len-as-condition
            # We deleted the whole graph, so no relaxation is possible.
            return None

        # The deletions above may have left disconncted "islands" in the pattern.
        # We remove these as well by deleting all connected components except the one
        # containing the successful portion of the pattern match.

        pattern_as_undirected_graph = pattern_as_digraph.to_undirected(as_view=True)

        if (
            len(  # pylint:disable=len-as-condition
                match_failure.largest_match_pattern_subgraph
            )
            > 0
        ):
            # We just need any node which did match so we know which connected component
            # to keep from the post-deletion graph
            probe_node = first(
                match_failure.largest_match_pattern_subgraph._graph.nodes  # pylint:disable=protected-access
            )

            to_delete_due_to_disconnection: List[NodePredicate] = []
            connected_components_containing_successful_pattern_matches = 0
            for connected_component in connected_components(pattern_as_undirected_graph):
                if probe_node not in connected_component:
                    # this is a component which is now disconnected from the successful partial match
                    to_delete_due_to_disconnection.extend(connected_component)
                else:
                    connected_components_containing_successful_pattern_matches += 1

            if connected_components_containing_successful_pattern_matches != 1:
                if graph_logger:
                    graph_logger.log_match_failure(
                        match_failure, logging.INFO, "Pattern match component failure"
                    )
                raise RuntimeError(
                    f"Expected the successfully matching portion of the pattern"
                    f" to belong to a single connected component, but it was in "
                    f"{connected_components_containing_successful_pattern_matches}"
                )

            if to_delete_due_to_disconnection:
                logging.info(
                    "Relaxation: deleted due to disconnection: %s",
                    to_delete_due_to_disconnection,
                )
                pattern_as_digraph.remove_nodes_from(to_delete_due_to_disconnection)
        else:
            # If nothing was successfully matched, it is less clear how to choose what portion
            # of the pattern to keep. For now, we are going to keep the largest connected component.
            # BEWARE: we have not confirmed the order of connected components is deterministic.
            components = sorted(
                list(connected_components(pattern_as_undirected_graph)),
                key=lambda x: len(x),  # pylint:disable=unnecessary-lambda
            )

            # We know there is some component because otherwise we would have bailed out
            # after we did the first deletion pass above.
            biggest_component = first(components)
            for component in components:
                if component != biggest_component:
                    logging.info(
                        "Relaxation: deleted due to being in smaller component: %s",
                        component,
                    )
                    pattern_as_digraph.remove_nodes_from(component)

        return PerceptionGraphPattern(
            pattern_as_digraph.copy(), dynamic=match_failure.pattern.dynamic
        )

    @matching_pattern_against_pattern.default
    def _matching_pattern_against_pattern_default(self) -> bool:
        if isinstance(self.graph_to_match_against, PerceptionGraphPattern):
            return True
        elif isinstance(self.graph_to_match_against, PerceptionGraph):
            return False
        else:
            raise RuntimeError(
                f"Don't know how to match against: {self.graph_to_match_against}"
            )


@attrs(frozen=True, slots=True, eq=False)
class PerceptionGraphPatternMatch:
    """
    Represents a match of a `PerceptionGraphPattern` against a `PerceptionGraph`.
    """

    matched_pattern: PerceptionGraphPattern = attrib(
        validator=instance_of(PerceptionGraphPattern), kw_only=True
    )
    graph_matched_against: PerceptionGraphProtocol = attrib(
        validator=instance_of(PerceptionGraphProtocol), kw_only=True
    )
    matched_sub_graph: PerceptionGraphProtocol = attrib(
        validator=instance_of(PerceptionGraphProtocol), kw_only=True
    )
    pattern_node_to_matched_graph_node: Mapping[
        "NodePredicate", PerceptionGraphNode
    ] = attrib(converter=_to_immutabledict, kw_only=True)
    """
    A mapping of pattern nodes from `matched_pattern` to the nodes
    in `matched_sub_graph` they were aligned to.
    """

    def confirm_pattern_match(self) -> None:
        """
        Update the pattern graph nodes by confirming a match for all matched nodes.

        This mutates the pattern graph nodes.
        """
        for node in self.matched_pattern:
            node.confirm_match(self.pattern_node_to_matched_graph_node[node])

    def confirm_graph_match(self) -> None:
        """
        Update the graph nodes by confirming a match for all matched nodes.

        This method does nothing when the graph is an actual perception graph. When the graph is a
        pattern graph, this mutates the graph nodes.
        """
        if isinstance(self.graph_matched_against, PerceptionGraphPattern):
            for node in self.matched_pattern:
                graph_node = cast(
                    NodePredicate, self.pattern_node_to_matched_graph_node[node]
                )
                graph_node.confirm_match(node)


class EdgePredicate(ABC):
    r"""
    Super-class for pattern graph edges.
    """

    @abstractmethod
    def __call__(
        self,
        source_object_perception: PerceptionGraphNode,
        edge_label: Union[EdgeLabel, TemporallyScopedEdgeLabel],
        dest_object_percption: PerceptionGraphNode,
    ) -> bool:
        """
        Returns whether this predicate matches the edge
        from *source_object* to *dest_object* with label *edge_label*.
        """

    @abstractmethod
    def dot_label(self) -> str:
        """
        Edge label to use when rendering patterns as graphs using *dot*.
        """

    def reverse_in_dot_graph(self) -> bool:
        """
        In the dot graph, should this edge be treated as reversed for layout purposes?
        (dot tries to put the sources of edges to the left of the destinations)
        """
        return False

    @abstractmethod
    def matches_predicate(self, edge_predicate: "EdgePredicate") -> bool:
        """
        Returns whether *edge_label* matches *self*
        """


_BEFORE_AND_AFTER = immutableset([TemporalScope.BEFORE, TemporalScope.AFTER])
_DURING_ONLY = immutableset([TemporalScope.DURING])
_AT_SOME_POINT_ONLY = immutableset([TemporalScope.AT_SOME_POINT])


@attrs(frozen=True, slots=True)
class HoldsAtTemporalScopePredicate(EdgePredicate):
    """
    `EdgePredicate` which matches an edge with a `TemporallyScopedEdgeLabel`
    whose attribute matches *wrapped_edge_predicate*
    and which has at least the temporal scope *temporal_scope*
    (but may have others).
    """

    wrapped_edge_predicate: EdgePredicate = attrib(validator=instance_of(EdgePredicate))
    temporal_scopes: ImmutableSet[TemporalScope] = attrib(converter=_to_immutableset)

    def __call__(
        self,
        source_object_perception: PerceptionGraphNode,
        edge_label: Union[EdgeLabel, TemporallyScopedEdgeLabel],
        dest_object_percption: PerceptionGraphNode,
    ) -> bool:
        if isinstance(edge_label, TemporallyScopedEdgeLabel):
            return (
                all(
                    scope in edge_label.temporal_specifiers
                    for scope in self.temporal_scopes
                )
                # This is callable. I don't know why pylint doesn't understand that.
                and self.wrapped_edge_predicate(  # pylint:disable=not-callable
                    source_object_perception, edge_label.attribute, dest_object_percption
                )
            )
        else:
            raise RuntimeError(
                f"Cannot apply HoldsAtTemporalScopePredicate to anything but "
                f"a TemporallyScopedEdgeLabel."
                f"This exception probably indicates that you are applying "
                f"a pattern intended for a dynamic situation to a static situation."
                f"Source: {source_object_perception}; Dest: {dest_object_percption};"
                f"Predicate: {self}; Label: {edge_label}"
            )

    def dot_label(self) -> str:
        # To reduce clutter, we only render temporal information for edges which are not
        # both before and after.
        if self.temporal_scopes == _BEFORE_AND_AFTER:
            return self.wrapped_edge_predicate.dot_label()
        else:
            return f"{self.wrapped_edge_predicate.dot_label()}@{','.join(scope.name for scope in self.temporal_scopes)}"

    def matches_predicate(self, edge_predicate: "EdgePredicate") -> bool:
        return (
            isinstance(edge_predicate, HoldsAtTemporalScopePredicate)
            and self.temporal_scopes == edge_predicate.temporal_scopes
            and self.wrapped_edge_predicate.matches_predicate(
                edge_predicate.wrapped_edge_predicate
            )
        )


@attrs(frozen=True, slots=True)
class AnyEdgePredicate(EdgePredicate):
    """
    `EdgePredicate` which matches any edge.
    """

    def __call__(
        self,
        source_object_perception: PerceptionGraphNode,
        edge_label: Union[EdgeLabel, TemporallyScopedEdgeLabel],
        dest_object_percption: PerceptionGraphNode,
    ) -> bool:
        return True

    def dot_label(self) -> str:
        return "*"

    def matches_predicate(self, edge_predicate: "EdgePredicate") -> bool:
        return isinstance(edge_predicate, AnyEdgePredicate)


@attrs(frozen=True, slots=True)
class RelationTypeIsPredicate(EdgePredicate):
    """
    `EdgePredicate` which matches a relation of the given type.
    """

    relation_type: OntologyNode = attrib(validator=instance_of(OntologyNode))

    def __call__(
        self,
        source_object_perception: PerceptionGraphNode,
        edge_label: Union[EdgeLabel, TemporallyScopedEdgeLabel],
        dest_object_percption: PerceptionGraphNode,
    ) -> bool:
        return edge_label == self.relation_type

    def dot_label(self) -> str:
        return str(self.relation_type)

    def reverse_in_dot_graph(self) -> bool:
        return self.relation_type == PART_OF

    def matches_predicate(self, edge_predicate: "EdgePredicate") -> bool:
        return (
            isinstance(edge_predicate, RelationTypeIsPredicate)
            and edge_predicate.relation_type == self.relation_type
        )


@attrs(frozen=True, slots=True)
class DirectionPredicate(EdgePredicate):
    """
    `EdgePredicate` which matches a `Direction` object
    annotating an edge between a `Region` and an `ObjectPerception`.
    """

    reference_direction: Direction[Any] = attrib(validator=instance_of(Direction))

    def __call__(
        self,
        source_object_perception: PerceptionGraphNode,
        edge_label: Union[EdgeLabel, TemporallyScopedEdgeLabel],
        dest_object_percption: PerceptionGraphNode,
    ) -> bool:
        return (
            isinstance(edge_label, Direction)
            and edge_label.positive == self.reference_direction.positive
        )

    def dot_label(self) -> str:
        return f"dir({sign(self.reference_direction.positive)})"

    @staticmethod
    def exactly_matching(direction: Direction[Any]) -> "DirectionPredicate":
        return DirectionPredicate(direction)

    def matches_predicate(self, edge_predicate: "EdgePredicate") -> bool:
        return isinstance(edge_predicate, DirectionPredicate) and (
            edge_predicate.reference_direction.positive
            == self.reference_direction.positive
        )


# Graph translation code shared between perception graph construction
# and pattern construction

_AxisMapper = Callable[[GeonAxis], Any]
_EdgeMapper = Callable[[Any], MutableMapping[str, Any]]


def _add_labelled_edge(
    graph: DiGraph,
    source: Any,
    target: Any,
    unmapped_label: Any,
    *,
    map_edge: _EdgeMapper,
    temporal_scopes: AbstractSet[TemporalScope] = immutableset(),
):
    graph.add_edge(source, target)
    mapped_edge = map_edge(unmapped_label)
    if temporal_scopes:
        mapped_edge[LABEL] = TemporallyScopedEdgeLabel.for_dynamic_perception(
            mapped_edge[LABEL], when=temporal_scopes
        )
    graph.edges[source, target].update(mapped_edge)


def _translate_axes(
    graph: DiGraph,
    owner: HasAxes,
    mapped_owner: Any,
    map_axis: _AxisMapper,
    map_edge: _EdgeMapper,
) -> None:
    mapped_primary_axis = map_axis(owner.axes.primary_axis)
    graph.add_node(mapped_primary_axis)
    _add_labelled_edge(
        graph, mapped_owner, mapped_primary_axis, PRIMARY_AXIS_LABEL, map_edge=map_edge
    )
    graph.add_nodes_from(
        map_axis(orienting_axis) for orienting_axis in owner.axes.orienting_axes
    )
    for orienting_axis in owner.axes.orienting_axes:
        mapped_axis = map_axis(orienting_axis)
        _add_labelled_edge(
            graph, mapped_owner, mapped_axis, HAS_AXIS_LABEL, map_edge=map_edge
        )

    # the relations between those axes becomes edges
    for axis_relation in owner.axes.axis_relations:
        mapped_arg1 = map_axis(axis_relation.first_slot)
        mapped_arg2 = map_axis(cast(GeonAxis, axis_relation.second_slot))
        _add_labelled_edge(
            graph,
            mapped_arg1,
            mapped_arg2,
            axis_relation.relation_type,
            map_edge=map_edge,
        )


def _translate_geon(
    graph: DiGraph,
    owner: MaybeHasGeon,
    *,
    mapped_owner: Any,
    map_geon: Callable[[Geon], Any],
    map_axis: _AxisMapper,
    map_edge: _EdgeMapper,
) -> None:
    if owner.geon:
        mapped_geon = map_geon(owner.geon)
        graph.add_node(mapped_geon)
        _add_labelled_edge(
            graph, mapped_owner, mapped_geon, HAS_GEON_LABEL, map_edge=map_edge
        )
        graph.add_node((mapped_geon[0].cross_section, mapped_geon[1]))
        _add_labelled_edge(
            graph,
            mapped_owner,
            (mapped_geon[0].cross_section, mapped_geon[1]),
            HAS_PROPERTY_LABEL,
            map_edge=map_edge,
        )
        _translate_axes(graph, owner.geon, mapped_geon, map_axis, map_edge)
        mapped_generating_axis = map_axis(owner.geon.generating_axis)
        _add_labelled_edge(
            graph,
            mapped_geon,
            mapped_generating_axis,
            GENERATING_AXIS_LABEL,
            map_edge=map_edge,
        )


def _translate_region(
    graph: DiGraph,
    region: Region[Any],
    *,
    map_node: Callable[[Any], Any],
    map_edge: _EdgeMapper,
    axes_info: Optional[AxesInfo[Any]] = None,
    temporal_scopes: AbstractSet[TemporalScope] = immutableset(),
) -> None:
    mapped_region = map_node(region)
    mapped_reference_object = map_node(region.reference_object)
    _add_labelled_edge(
        graph,
        mapped_region,
        mapped_reference_object,
        REFERENCE_OBJECT_LABEL,
        map_edge=map_edge,
        temporal_scopes=temporal_scopes,
    )
    if region.direction:
        axis_relative_to = region.direction.relative_to_concrete_axis(axes_info)
        mapped_axis_relative_to = map_node(axis_relative_to)
        _add_labelled_edge(
            graph,
            mapped_region,
            mapped_axis_relative_to,
            region.direction,
            map_edge=map_edge,
            temporal_scopes=temporal_scopes,
        )


# This is used to control the order in which pattern nodes are matched,
# which can have a significant impact on match speed.
# We try to match the most restrictive nodes first.
_PATTERN_PREDICATE_NODE_ORDER = [
    # If we have matchedObjects in the pattern we want to try and find these first.
    ObjectSemanticNodePerceptionPredicate,
    # Match the Stroke GNN property next
    StrokeGNNRecognitionPredicate,
    # The graph predicate types listed here
    AnyObjectPredicate,
    ObjectStrokePredicate,
    CategoricalPredicate,
    DistributionalContinuousPredicate,
    ContinuousPredicate,
    RgbColorPredicate,
    CielabColorPredicate,
    ObjectStroke,
    CategoricalNode,
    ContinuousNode,
    RgbColorNode,
    CielabColorNode,
    # Paths are rare, match them next
    IsPathPredicate,
    PathOperatorPredicate,
    # properties and colors tend to be highlight restrictive, so let's match them first
    IsOntologyNodePredicate,
    IsColorNodePredicate,
    AnyObjectPerception,
    GeonPredicate,
    CrossSectionPredicate,
    RegionPredicate,
    # the matcher tends to get bogged down when dealing with axes,
    # so we search those last one the other nodes have established the skeleton of a match.
    AxisPredicate,
]


def _pattern_matching_node_order(node_node_data_tuple) -> int:
    (node, _) = node_node_data_tuple
    return _PATTERN_PREDICATE_NODE_ORDER.index(node.__class__)


# This is used to control the order in which pattern nodes are matched,
# which can have a significant impact on match speed.
# This should match _PATTERN_PREDICATE_NODE_ORDER above.
_GRAPH_NODE_ORDER = [  # type: ignore
    ObjectSemanticNode,
    StrokeGNNRecognitionNode,
    ObjectClusterNode,
    ObjectStroke,
    CategoricalNode,
    ContinuousNode,
    RgbColorNode,
    CielabColorNode,
    SpatialPath,
    PathOperator,
    OntologyNode,
    RgbColorPerception,
    ObjectPerception,
    Geon,
    Region,
    GeonAxis,
    CrossSection,
]


def _graph_node_order(node_node_data_tuple) -> int:
    (node, _) = node_node_data_tuple
    if isinstance(node, tuple):
        # some node types are wrapped in tuples with unique ids to keep them distinct
        # (e.g. otherwise identical Geon objects).
        # We need to unwrap these before comparing types.
        node = node[0]

    return _GRAPH_NODE_ORDER.index(node.__class__)


GOVERNED = OntologyNode("governed")
"""
An object match governed in a preposition relationship
"""
MODIFIED = OntologyNode("modified")
"""
An object match modified in a preposition relationship
"""


_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


def _invert_to_immutabledict(mapping: Mapping[_KT, _VT]) -> ImmutableDict[_VT, _KT]:
    return immutabledict((v, k) for (k, v) in mapping.items())


@attrs
class GraphLogger:
    log_directory: Path = attrib(validator=instance_of(Path))
    enable_graph_rendering: bool = attrib(validator=instance_of(bool))
    serialize_graphs: bool = attrib(validator=instance_of(bool), default=False)
    call_count: int = attrib(init=False, default=0)

    def log_graph(
        self,
        graph: PerceptionGraphProtocol,
        level,
        msg: str,
        *args,
        match_correspondence_ids: Mapping[Any, str] = immutabledict(),
        graph_name: Optional[str] = None,
    ) -> None:
        self.call_count += 1
        if self.enable_graph_rendering:
            if not graph_name:
                graph_name = str(uuid4())

            filename = self.log_directory / f"{graph_name}"
            graph.render_to_file(
                graph_name, filename, match_correspondence_ids=match_correspondence_ids
            )
            logging.log(
                level, f"[{self.call_count}] Rendered to {filename}.pdf\n{msg}", *args
            )
            if self.serialize_graphs:
                serialized_file = self.log_directory / f"{graph_name}.serialized"
                logging.info("Serializing to %s", serialized_file)
                with open(str(serialized_file), "wb") as out:
                    pickle.dump(graph, out)
        else:
            logging.log(level, msg, *args)

    def log_match_failure(
        self,
        match_failure: PatternMatching.MatchFailure,
        level,
        msg: str,
        *args,
        graph_name: Optional[str] = None,
    ) -> None:
        if not graph_name:
            graph_name = str(uuid4())

        if self.enable_graph_rendering:
            graph_correspondence_ids: Mapping[Any, str] = immutabledict(
                (graph_node, str(i))
                for (i, (_, graph_node)) in enumerate(
                    match_failure.pattern_node_to_graph_node_for_largest_match.items()
                )
            )
            self.log_graph(
                match_failure.graph,
                level,
                msg + " [graph] ",
                *args,
                match_correspondence_ids=graph_correspondence_ids,
                graph_name=f"{graph_name}-graph",
            )

            pattern_correspondence_ids: Mapping[Any, str] = immutabledict(
                (pattern_node, str(i))
                for (i, (pattern_node, _)) in enumerate(
                    match_failure.pattern_node_to_graph_node_for_largest_match.items()
                )
            )
            self.log_graph(
                match_failure.pattern,
                level,
                msg + " [pattern] ",
                *args,
                match_correspondence_ids=pattern_correspondence_ids,
                graph_name=f"{graph_name}-pattern",
            )
        else:
            logging.log(level, msg, *args)

    def log_pattern_match(
        self,
        pattern_match: PerceptionGraphPatternMatch,
        level,
        msg: str,
        *args,
        graph_name: Optional[str] = None,
    ) -> None:
        if not graph_name:
            graph_name = str(uuid4())

        if self.enable_graph_rendering:
            graph_correspondence_ids: Mapping[Any, str] = immutabledict(
                (graph_node, str(i))
                for (i, (_, graph_node)) in enumerate(
                    pattern_match.pattern_node_to_matched_graph_node.items()
                )
            )
            self.log_graph(
                pattern_match.graph_matched_against,
                level,
                msg + " [graph] ",
                *args,
                match_correspondence_ids=graph_correspondence_ids,
                graph_name=f"{graph_name}-graph",
            )

            pattern_correspondence_ids: Mapping[Any, str] = immutabledict(
                (pattern_node, str(i))
                for (i, (pattern_node, _)) in enumerate(
                    pattern_match.pattern_node_to_matched_graph_node.items()
                )
            )
            self.log_graph(
                pattern_match.matched_pattern,
                level,
                msg + " [pattern] ",
                *args,
                match_correspondence_ids=pattern_correspondence_ids,
                graph_name=f"{graph_name}-pattern",
            )
        else:
            logging.log(level, msg, *args)


def _uniquify(
    item: PerceptionGraphNode, referring_node: PerceptionGraphNode
) -> Tuple[PerceptionGraphNode, PerceptionGraphNode]:
    """
    Utility method to force object which would otherwise compare as equal
    to be distinct when used as nodes in a digraph.
    """
    return item, referring_node


class _FrameTranslation(ABC, Generic[PerceptionT]):
    unique_counter: Incrementer = attrib(init=False, default=Incrementer(0))

    @abstractmethod
    def _translate_frame(self, frame: PerceptionT) -> DiGraph:
        """Gets the `DiGraph` corresponding to the `PerceptionT` frame."""
        raise NotImplementedError

    def translate_frame(self, frame: PerceptionT) -> PerceptionGraph:
        """Gets the `PerceptionGraph` corresponding to a the `PerceptionT` frame."""
        return PerceptionGraph(self._translate_frame(frame))

    @abstractmethod
    def _translate_frames(
        self, perceptual_representation: PerceptualRepresentation[PerceptionT]
    ) -> DiGraph:
        """Gets the dynamic `DiGraph` corresponding to the `PerceptionT` frames."""

    def translate_frames(
        self, perceptual_representation: PerceptualRepresentation[PerceptionT]
    ) -> PerceptionGraph:
        """Gets the dynamic `PerceptionGraph` corresponding to the `PerceptualRepresentation` with multiple frames."""
        # The PerceptionGraph constructor always copies the graph it's passed, so to avoid copying
        # the constructed digraph, we use this ugly hack. We construct a PerceptionGraph from an
        # empty graph (which should be faster) and use object.__setattr__ to set its graph attribute
        # to our constructed graph (working around the fact that PerceptionGraph is frozen).
        dynamic_digraph = self._translate_frames(perceptual_representation)
        new_perception_graph = PerceptionGraph(graph=DiGraph(), dynamic=True)
        object.__setattr__(new_perception_graph, "_graph", dynamic_digraph)
        return new_perception_graph

    @staticmethod
    def _map_node(
        obj: Any,
        *,
        referring_node_to_enforce_uniqueness: Optional[PerceptionGraphNode] = None,
    ):
        # If the object passed in is a GraphNode then is it already unique
        # and we can just return it
        if isinstance(obj, GraphNode):
            return obj
        # in some cases, a node will be combined with its referring node
        # to force otherwise identical objects to be treated separately.
        # We do this for properties, for example, so that if two things
        # are both animate, they end up with distinct animacy nodes in the graph
        # which could be e.g. treated differently during pattern relaxation.
        elif referring_node_to_enforce_uniqueness:
            return _uniquify(obj, referring_node_to_enforce_uniqueness)
        # Regions and Geons are normally treated as value objects,
        # but we want to maintain their distinctness in the perceptual graph
        # for the purpose of matching patterns, so we make their corresponding
        # graph nods compare by identity.
        elif isinstance(obj, (Region, Geon)):
            return obj, id(obj)
        else:
            return obj

    @staticmethod
    def _map_edge(label: Any) -> MutableMapping[str, Any]:
        return {LABEL: label}


@attrs
class _DevelopmentalPrimitivePerceptionFrameTranslation(
    _FrameTranslation[DevelopmentalPrimitivePerceptionFrame]
):
    def _translate_frame(self, frame: DevelopmentalPrimitivePerceptionFrame) -> DiGraph:
        """Gets the `DiGraph` corresponding to a `DevelopmentalPrimitivePerceptionFrame`."""
        graph = DiGraph()

        for perceived_object in frame.perceived_objects:
            # Every perceived object is a node in the graph.
            graph.add_node(perceived_object)
            # And so are each of its axes.
            _translate_axes(
                graph,
                perceived_object,
                perceived_object,
                self._map_node,
                map_edge=self._map_edge,
            )
            _translate_geon(
                graph,
                perceived_object,
                mapped_owner=perceived_object,
                map_geon=self._map_node,
                map_axis=self._map_node,
                map_edge=self._map_edge,
            )

        regions = [
            relation.second_slot
            for relation in frame.relations
            if isinstance(relation.second_slot, Region)
        ]

        for region in regions:
            _translate_region(
                graph,
                region,
                map_node=self._map_node,
                map_edge=self._map_edge,
                axes_info=frame.axis_info,
            )

        # Every relation is handled as a directed graph edge
        # from the first argument to the second
        for relation in frame.relations:
            if not relation.negated:
                # We currently ignore negated relations.
                # See https://github.com/isi-vista/adam/issues/707
                self._map_relation(graph, relation)

        dest_node: Any
        for property_ in frame.property_assertions:
            source_node = self._map_node(property_.perceived_object)
            if isinstance(property_, HasBinaryProperty):
                # TODO: fix this hack for me and you https://github.com/isi-vista/adam/issues/917
                if (
                    property_.binary_property in [IS_SPEAKER, IS_ADDRESSEE]
                    and source_node.debug_handle
                    and source_node.debug_handle != "learner"
                ):
                    dest_node = self._map_node(
                        property_.binary_property,
                        referring_node_to_enforce_uniqueness=source_node,
                    )
                    # this specification is really gross but ensures that our attribute nodes are unique from each other and
                    # other attribute nodes that are added doubly.
                    dest_node_2 = self._map_node(
                        property_.binary_property,
                        referring_node_to_enforce_uniqueness=dest_node,
                    )
                    graph.add_edge(source_node, dest_node, label=HAS_PROPERTY_LABEL)
                    graph.add_edge(source_node, dest_node_2, label=HAS_PROPERTY_LABEL)
                else:
                    dest_node = self._map_node(
                        property_.binary_property,
                        referring_node_to_enforce_uniqueness=source_node,
                    )
                    graph.add_edge(source_node, dest_node, label=HAS_PROPERTY_LABEL)

            elif isinstance(property_, HasColor):
                dest_node = self._map_node(
                    property_.color, referring_node_to_enforce_uniqueness=source_node
                )
                graph.add_edge(source_node, dest_node, label=HAS_PROPERTY_LABEL)

            else:
                raise RuntimeError(f"Don't know how to translate property {property_}")

        if frame.axis_info:
            for (object_, axis) in frame.axis_info.axes_facing.items():
                graph.add_edge(axis, object_, label=FACING_OBJECT_LABEL)

        return graph

    def _translate_frames(
        self,
        perceptual_representation: PerceptualRepresentation[
            DevelopmentalPrimitivePerceptionFrame
        ],
    ) -> DiGraph:
        check_arg(
            len(perceptual_representation.frames) == 2,
            "Can only create a DynamicPerceptionGraph from exactly two frames, "
            "but got %s",
            (len(perceptual_representation.frames),),
        )

        _dynamic_digraph = _merge_before_and_after_graphs(
            self._translate_frame(perceptual_representation.frames[0]),
            self._translate_frame(perceptual_representation.frames[1]),
        )

        # Translate path information
        if perceptual_representation.during:
            regions = []
            axes_info = perceptual_representation.frames[0].axis_info
            if perceptual_representation.during.objects_to_paths:
                for (
                    moving_object,
                    path_info,
                ) in perceptual_representation.during.objects_to_paths.items():
                    self._add_path_node(
                        _dynamic_digraph,
                        moving_object,
                        path_info,
                        axes_info=axes_info,
                        map_edge=self._map_edge,
                        map_node=self._map_node,
                    )
                    if isinstance(path_info.reference_source_object, Region):
                        regions.append(path_info.reference_source_object)
                    if isinstance(path_info.reference_destination_object, Region):
                        regions.append(path_info.reference_destination_object)

            # Below we ensure all regions appearing as relation and path arguments
            # are correctly translated.
            relations = []
            for relation in perceptual_representation.during.at_some_point:
                if not relation.negated:
                    # We currently ignore negated relations.
                    # See https://github.com/isi-vista/adam/issues/707
                    self._map_relation(
                        _dynamic_digraph, relation, temporal_scopes=_AT_SOME_POINT_ONLY
                    )
                    relations.append(relation)
            for relation in perceptual_representation.during.continuously:
                if not relation.negated:
                    # We currently ignore negated relations.
                    # See https://github.com/isi-vista/adam/issues/707
                    self._map_relation(
                        _dynamic_digraph, relation, temporal_scopes=_DURING_ONLY
                    )
                    relations.append(relation)
            for relation in relations:
                if isinstance(relation.second_slot, Region):
                    regions.append(relation.second_slot)
            for region in regions:
                _translate_region(
                    _dynamic_digraph,
                    region,
                    map_node=self._map_node,
                    map_edge=self._map_edge,
                    axes_info=axes_info,
                    temporal_scopes=_DURING_ONLY,
                )

        return _dynamic_digraph

    def _add_path_node(
        self,
        perception_digraph: DiGraph,
        moving_object: ObjectPerception,
        path: SpatialPath[ObjectPerception],
        *,
        map_node: Callable[[Any], Any],
        map_edge: _EdgeMapper,
        axes_info: Optional[AxesInfo[Any]] = None,
    ) -> None:
        edges_to_add: List[Tuple[Any, Any, Any]] = []
        edges_to_add.append((moving_object, path, HAS_PATH_LABEL))
        edges_to_add.append(
            (path, map_node(path.reference_source_object), REFERENCE_OBJECT_SOURCE_LABEL)
        )
        edges_to_add.append(
            (
                path,
                map_node(path.reference_destination_object),
                REFERENCE_OBJECT_DESTINATION_LABEL,
            )
        )
        if path.reference_source_object == path.reference_destination_object:
            if isinstance(path.reference_source_object, Region):
                _translate_region(
                    perception_digraph,
                    path.reference_source_object,
                    map_node=map_node,
                    map_edge=map_edge,
                    axes_info=axes_info,
                    temporal_scopes=_DURING_ONLY,
                )
        else:
            if isinstance(path.reference_source_object, Region):
                _translate_region(
                    perception_digraph,
                    path.reference_source_object,
                    map_node=map_node,
                    map_edge=map_edge,
                    axes_info=axes_info,
                    temporal_scopes=immutableset([TemporalScope.BEFORE]),
                )

            if isinstance(path.reference_destination_object, Region):
                _translate_region(
                    perception_digraph,
                    path.reference_destination_object,
                    map_node=map_node,
                    map_edge=map_edge,
                    axes_info=axes_info,
                    temporal_scopes=immutableset([TemporalScope.AFTER]),
                )
        if path.reference_axis:
            edges_to_add.append(
                (
                    path,
                    path.reference_axis
                    if isinstance(path.reference_axis, GeonAxis)
                    else path.reference_axis.to_concrete_axis(axes_info),
                    REFERENCE_AXIS_LABEL,
                )
            )
        if path.operator:
            edges_to_add.append(
                (path, _uniquify(path.operator, referring_node=path), HAS_PATH_OPERATOR)
            )
        # link path object to orientation_changed boolean if it did as a has-property
        if path.orientation_changed:
            edges_to_add.append(
                (
                    path,
                    _uniquify(ORIENTATION_CHANGED_PROPERTY, referring_node=path),
                    HAS_PROPERTY_LABEL,
                )
            )
        for prop in path.properties:
            edges_to_add.append(
                (path, _uniquify(prop, referring_node=path), HAS_PROPERTY_LABEL)
            )
        for (source, target, label) in edges_to_add:
            perception_digraph.add_edge(
                source,
                target,
                label=TemporallyScopedEdgeLabel.for_dynamic_perception(
                    label, TemporalScope.DURING
                ),
            )

    def _map_relation(
        self,
        graph: DiGraph,
        relation: Relation[ObjectPerception],
        *,
        temporal_scopes: Iterable[TemporalScope] = tuple(),
    ) -> None:
        label: Any
        if temporal_scopes:
            label = TemporallyScopedEdgeLabel.for_dynamic_perception(
                relation.relation_type, when=temporal_scopes
            )
        else:
            label = relation.relation_type
        graph.add_edge(
            self._map_node(relation.first_slot),
            self._map_node(relation.second_slot),
            label=label,
        )


@attrs
class _VisualPerceptionFrameTranslation(_FrameTranslation[VisualPerceptionFrame]):
    def _translate_frame(self, frame: VisualPerceptionFrame) -> DiGraph:
        """Gets the `DiGraph` corresponding to a `VisualPerceptionFrame`."""
        graph = DiGraph()

        # This will be used in the future when linking relationships between clusters
        cluster_perception_to_graph_node = dict()

        for perceived_cluster in frame.clusters:
            cluster_node = ObjectClusterNode(
                cluster_id=perceived_cluster.cluster_id,
                viewpoint_id=perceived_cluster.viewpoint_id,
                center_x=perceived_cluster.centroid_x,
                center_y=perceived_cluster.centroid_y,
                weight=1.0,
                std=perceived_cluster.std,
            )
            cluster_perception_to_graph_node[perceived_cluster] = cluster_node
            graph.add_node(cluster_node)

            for obj_property in perceived_cluster.properties:
                dest_node = self._map_node(obj_property)
                graph.add_edge(cluster_node, dest_node, label=HAS_PROPERTY_LABEL)

            for stroke in perceived_cluster.strokes:
                graph.add_edge(cluster_node, stroke, label=HAS_STROKE_LABEL)

            for source_stroke, adj_strokes in perceived_cluster.adjacent_strokes.items():
                for dest_stroke, is_adjacent in adj_strokes.items():
                    # We want to ensure we don't create a cycle in the DiGraph
                    # so we check that there isn't an edge in the opposite direction
                    # of what we are going to add
                    if (
                        not graph.has_edge(dest_stroke, source_stroke)
                        and is_adjacent
                        and dest_stroke != source_stroke
                    ):
                        graph.add_edge(
                            source_stroke, dest_stroke, label=ADJACENT_STROKE_LABEL
                        )

        return graph

    def _translate_frames(
        self,
        perceptual_representation: PerceptualRepresentation[VisualPerceptionFrame],
    ) -> DiGraph:
        """Gets the dynamic `DiGraph` corresponding to a `VisualPerceptionFrame`."""
        check_arg(
            len(perceptual_representation.frames) == 2,
            "Can only create a DynamicPerceptionGraph from exactly two frames, "
            "but got %s",
            (len(perceptual_representation.frames),),
        )

        if perceptual_representation.simulated_actions_features is None:
            raise ValueError(
                "Simulated actions features cannot be None when creating a multi-frame perception graph."
            )

        _dynamic_digraph = _merge_before_and_after_graphs(
            self._translate_frame(perceptual_representation.frames[0]),
            self._translate_frame(perceptual_representation.frames[1]),
        )

        decoded_object_to_cluster_node = {
            object_cluster_node.cluster_id: object_cluster_node
            for object_cluster_node in _dynamic_digraph.nodes
            if isinstance(object_cluster_node, ObjectClusterNode)
        }

        action_stroke_graph = perceptual_representation.simulated_actions_features[
            "stroke_graph"
        ]
        agent_node = decoded_object_to_cluster_node[
            perceptual_representation.simulated_actions_features["object_name"]
        ]
        involved_object_nodes = [
            decoded_object_to_cluster_node[object_name]
            for object_name in perceptual_representation.simulated_actions_features[
                "objects"
            ]
        ]
        involved_object_nodes.append(agent_node)

        action_decode_node = TrajectoryRecognitionNode(
            action_recognized=action_stroke_graph["action_name"].lower(),
            confidence=action_stroke_graph["confidence_score"],
            weight=action_stroke_graph["confidence_score"],
        )

        joint_point_nodes = self._extract_joint_points(action_stroke_graph)

        self._link_action_to_objects(
            _dynamic_digraph, action_decode_node, involved_object_nodes
        )
        self._link_action_to_agent(_dynamic_digraph, agent_node)
        self._link_agent_and_action_to_joint_points(
            _dynamic_digraph, agent_node, action_decode_node, joint_point_nodes
        )

        return _dynamic_digraph

    @staticmethod
    def _link_action_to_objects(
        perception_digraph: DiGraph,
        action_node: TrajectoryRecognitionNode,
        object_nodes: Sequence[ObjectClusterNode],
    ) -> None:
        for object_node in object_nodes:
            perception_digraph.add_edge(
                object_node,
                action_node,
                label=TemporallyScopedEdgeLabel.for_dynamic_perception(
                    INVOLVED_IN_ACTION, TemporalScope.DURING
                ),
            )

    @staticmethod
    def _link_action_to_agent(
        perception_digraph: DiGraph, agent_node: ObjectClusterNode
    ) -> None:
        # To help clarify the agent we mark them as 'VOLITIONAL' from the original symbolic system
        # This is mostly a way of representing the structured knowledge in the feature file.
        perception_digraph.add_edge(
            agent_node,
            VOLITIONALLY_INVOLVED,
            label=TemporallyScopedEdgeLabel.for_dynamic_perception(
                HAS_PROPERTY_LABEL, [TemporalScope.BEFORE, TemporalScope.AFTER]
            ),
        )

    @staticmethod
    def _link_agent_and_action_to_joint_points(
        perception_digraph: DiGraph,
        agent_node: ObjectClusterNode,
        action_node: TrajectoryRecognitionNode,
        joint_point_nodes: Sequence[JointPointNode],
    ) -> None:
        # Link the joint points to the Action
        for joint_node in joint_point_nodes:
            perception_digraph.add_edge(
                action_node,
                joint_node,
                label=TemporallyScopedEdgeLabel.for_dynamic_perception(
                    JOINT_POINT_TRAVERSAL, TemporalScope.DURING
                ),
            )

        # Special Case the first and last joint point temporally as 'before' and 'after'
        # This probably isn't needed but it's closer to temporally correct
        perception_digraph.add_edge(
            agent_node,
            joint_point_nodes[0],
            label=TemporallyScopedEdgeLabel.for_dynamic_perception(
                HAS_PROPERTY_LABEL, TemporalScope.BEFORE
            ),
        )
        perception_digraph.add_edge(
            agent_node,
            joint_point_nodes[-1],
            label=TemporallyScopedEdgeLabel.for_dynamic_perception(
                HAS_PROPERTY_LABEL, TemporalScope.AFTER
            ),
        )

    @staticmethod
    def _extract_joint_points(
        action_stroke_graph: Mapping[str, Any]
    ) -> Sequence[JointPointNode]:
        return [
            JointPointNode(
                scene_xyd_coord=DepthPoint(xyd_coord[0], xyd_coord[1], xyd_coord[2]),
                world_coord=Point(world_coord[0], world_coord[1], world_coord[2]),
                confidence=conf,
                weight=conf,
                temporal_index=temporal_index,
                joint_index=joint_num,
            )
            for temporal_index, (world_coords, xyd_coords, confs) in enumerate(
                zip(
                    action_stroke_graph["joint_points_world_coords"],
                    action_stroke_graph["joint_points_xyd_coords"],
                    action_stroke_graph["joint_points_confidence_score"],
                )
            )
            for joint_num, (world_coord, xyd_coord, conf) in enumerate(
                zip(world_coords, xyd_coords, confs)
            )
        ]


def _merge_before_and_after_graphs(before_graph, after_graph) -> DiGraph:
    """
    Merge two scenes together into a single temporally scopped graph.

    Args:
        before_graph: A graph of the initial scene state without temporal edges
        after_graph: A graph of the final scene state without temporal edges

    Returns:
        An acyclic DiGraph of the temporally scopped scene.
    """
    # First, we translate each of the two frames into PerceptionGraphs independently.
    # The edges of each graph are marked with the appropriate "temporal specifier"
    # which tells whether they belong to the "before" frame or the "after" frame.
    before_frame_graph = PerceptionGraph.add_temporal_scopes_to_edges(
        before_graph,
        TemporalScope.BEFORE,
    )

    after_frame_graph = PerceptionGraph.add_temporal_scopes_to_edges(
        after_graph,
        TemporalScope.AFTER,
    )

    # This will be what the PerceptionGraph we are building will wrap.
    # It starts with everything which is in the first frame's PerceptionGraph.
    _dynamic_digraph = before_frame_graph

    # We have to be more careful adding things from the second frame's PerceptionGraph.
    # We can freely add all the nodes because they don't contain temporal information
    # and adding the same node twice is harmless.
    _dynamic_digraph.add_nodes_from(after_frame_graph.nodes)

    # But now we need to walk edge-by-edge through the second graph,
    # adding edges which don't collide with our current edges,
    # but merging together edges which do (because we aren't using hypergraphs).
    # Note that because of the way perception graphs are constructed,
    # nodes representing objects will be reference-identical
    # between the two frame --> graph translations,
    # so we can use that to merge them below.
    for (source, target, after_label) in after_frame_graph.edges.data(LABEL):
        if _dynamic_digraph.has_edge(source, target):
            # This edge also appears in the first frame,
            # so we need to merge the edge metadata between the frames.

            # We know the edges in these graphs are wrapped with temporal scopes
            # because we applied the temporal scopes above.
            after_label = cast(TemporallyScopedEdgeLabel, after_label)
            before_label: TemporallyScopedEdgeLabel = _dynamic_digraph.edges[
                source, target
            ][LABEL]

            # We don't know how to merge edges which differ in anything
            # except temporal specifiers
            if before_label.attribute == after_label.attribute:
                _dynamic_digraph.edges[source, target][LABEL] = TemporallyScopedEdgeLabel(
                    before_label.attribute,
                    temporal_specifiers=chain(
                        before_label.temporal_specifiers,
                        after_label.temporal_specifiers,
                    ),
                )
            else:
                _dynamic_digraph.edges[source, target][LABEL] = TemporallyScopedEdgeLabel(
                    after_label.attribute,
                    temporal_specifiers=chain(after_label.temporal_specifiers),
                )
                logging.warning(
                    f"We currently don't know how to handle a change in label "
                    f"on an edge between the before frame and the after frame."
                    f"Source={source}; Target={target}; "
                    f"before label={before_label.attribute}; "
                    f"after label={after_label.attribute}."
                    f"As a hack, we delete the 'before' and preserve the 'after'."
                    f"See https://github.com/isi-vista/adam/issues/666"
                )
        else:
            # This edge does not also appear in the first frame,
            # so no merging is needed.
            _dynamic_digraph.add_edge(source, target, label=after_label)

    return _dynamic_digraph


def raise_graph_exception(exception_message: str, graph: PerceptionGraphProtocol):
    error_path = Path("error").absolute()
    try:
        graph.render_to_file("error", error_path)
    except RuntimeError:
        logging.info("Attempt to render offending graph failed")
    raise RuntimeError(
        f"{exception_message}\n"
        f"Violating graph: {graph}.\n"
        f"Offending graph was written to {error_path}.pdf"
    )


def edge_equals_ignoring_temporal_scope(
    edge_label: EdgeLabel, query: Union[EdgeLabel, Iterable[EdgeLabel]]
) -> bool:
    if not isinstance(query, Iterable):
        query = {query}
    if (
        isinstance(edge_label, TemporallyScopedEdgeLabel)
        and edge_label.attribute in query
    ):
        return True
    return edge_label in query


def get_features_from_semantic_node(
    semantic_root: SemanticNode, perception_graph: PerceptionGraph
) -> Sequence[str]:
    if semantic_root not in perception_graph:
        return []

    digraph = perception_graph.copy_as_digraph()

    return [
        f"{node}"
        for node in perception_graph.successors(semantic_root)
        if edge_equals_ignoring_temporal_scope(
            digraph.get_edge_data(semantic_root, node)[LABEL],
            {HAS_PROPERTY_LABEL, HAS_STROKE_LABEL},
        )
    ]
