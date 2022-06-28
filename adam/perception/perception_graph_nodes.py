from abc import ABC, abstractmethod
from typing import Union, Tuple, Any, Optional

from attr import attrs, attrib
from attr.validators import instance_of, in_, optional, deep_iterable
from immutablecollections import ImmutableSet
from immutablecollections.converter_utils import _to_immutableset

from adam.axis import GeonAxis
from adam.geon import Geon, CrossSection
from adam.math_3d import Point, DepthPoint
from adam.ontology import OntologyNode
from adam.ontology.phase1_spatial_relations import (
    Region,
    SpatialPath,
    PathOperator,
)
from adam.perception import ObjectPerception
from adam.semantics import SemanticNode


@attrs(slots=True, frozen=True)
class ObjectStroke:
    """Class to hold the coordinates of a Stroke."""

    normalized_coordinates: ImmutableSet[Point] = attrib(
        validator=deep_iterable(instance_of(Point)), converter=_to_immutableset
    )

    def dot_label(self) -> str:
        return (
            f"Stroke: [{', '.join(str(point) for point in self.normalized_coordinates)}]"
        )

    def __str__(self) -> str:
        return f"Stroke[{', '.join(f'({point.x:.2f}, {point.y:.2f})' for point in self.normalized_coordinates)}]"


# Perception graph nodes
@attrs(frozen=True, slots=True, eq=False)
class GraphNode(ABC):
    """Super-class for all perception graph nodes, useful for types."""

    weight: float = attrib(validator=instance_of(float))

    @abstractmethod
    def dot_label(self) -> str:
        raise NotImplementedError()


PerceptionGraphNode = Union[
    ObjectPerception,
    OntologyNode,
    Tuple[Region[Any], int],
    Tuple[Geon, int],
    GeonAxis,
    CrossSection,
    SemanticNode,
    SpatialPath[ObjectPerception],
    PathOperator,
    GraphNode,
    ObjectStroke,
]

# Some perception graph nodes are wrapped in tuples with counters
# to avoid them be treated as equal in NetworkX graphs.
# This type is the same as PerceptionGraphNode, except with such wrapping removed.
UnwrappedPerceptionGraphNode = Union[
    ObjectPerception,
    OntologyNode,
    Region[Any],
    Geon,
    GeonAxis,
    CrossSection,
    SemanticNode,
    SpatialPath[ObjectPerception],
    PathOperator,
    GraphNode,
    ObjectStroke,
]


@attrs(frozen=True, slots=True, eq=False)
class ObjectClusterNode(GraphNode):
    """A node representing a source of an object cluster perception."""

    cluster_id: str = attrib(validator=instance_of(str))
    viewpoint_id: int = attrib(validator=instance_of(int))
    center_x: Optional[float] = attrib(validator=optional(instance_of(float)))
    center_y: Optional[float] = attrib(validator=optional(instance_of(float)))
    std: Optional[float] = attrib(validator=optional(instance_of(float)))

    def dot_label(self):
        return (
            f"ObjectClusterNode(cluster_id={self.cluster_id!r})\n"
            f"(viewpoint_id={self.viewpoint_id}, "
            f"center=({self.center_x:.2g}, {self.center_y:.2g}), std={self.std:.2g})"
        )


@attrs(frozen=True, slots=True, eq=False)
class CategoricalNode(GraphNode):
    """A node representing a categorical value feature"""

    label: str = attrib(validator=instance_of(str))
    value: str = attrib(validator=instance_of(str))

    def dot_label(self):
        return f"CategoricalNode(label={self.label}, value={self.value})"


@attrs(frozen=True, slots=True, eq=False)
class ContinuousNode(GraphNode):
    """A node representing a continuous value feature."""

    label: str = attrib(validator=instance_of(str))
    value: float = attrib(validator=instance_of(float))

    def dot_label(self):
        return f"ContinuousNode(label={self.label}, value={self.value})"


@attrs(frozen=True, slots=True, eq=False)
class RgbColorNode(GraphNode):
    """A node representing an RGB perception value."""

    red: int = attrib(validator=in_(range(0, 256)))
    green: int = attrib(validator=in_(range(0, 256)))
    blue: int = attrib(validator=in_(range(0, 256)))

    def dot_label(self):
        return f"RgbColorNode({self})"

    def __str__(self) -> str:
        return f"#{hex(self.red)[2:]}{hex(self.green)[2:]}{hex(self.blue)[2:]}"


@attrs(frozen=True, slots=True, eq=False)
class StrokeGNNRecognitionNode(GraphNode):
    """A property node indicating Stroke GNN object recognition."""

    object_recognized: str = attrib(validator=instance_of(str))
    confidence: float = attrib(validator=instance_of(float))

    def dot_label(self):
        return (
            f"StrokeGNNRecognitionNode(object_recognized={self.object_recognized}, "
            f"confidence={self.confidence:.4f})"
        )

    def __str__(self) -> str:
        return f"StrokeGNNRecognized(recognized object={self.object_recognized} ({self.confidence:.2f}))"


@attrs(frozen=True, slots=True, eq=False)
class TrajectoryRecognitionNode(GraphNode):
    """A property node indicating Stroke GNN object recognition."""

    action_recognized: str = attrib(validator=instance_of(str))
    confidence: float = attrib(validator=instance_of(float))

    def dot_label(self):
        return (
            f"TrajectoryRecognitionNode(action_recognized={self.action_recognized}, "
            f"confidence={self.confidence:.4f})"
        )

    def __str__(self) -> str:
        return f"TrajectoryRecognitionNode(action recognized={self.action_recognized} ({self.confidence:.2f}))"


@attrs(frozen=True, slots=True, eq=False)
class JointPointNode(GraphNode):
    world_coord: Point = attrib(validator=instance_of(Point))
    scene_xyd_coord: DepthPoint = attrib(validator=instance_of(DepthPoint))
    temporal_index: int = attrib(validator=instance_of(int))
    joint_index: int = attrib(validator=instance_of(int))
    confidence: float = attrib(validator=instance_of(float))

    def dot_label(self) -> str:
        return (
            f"JointPointNode(world_coord={self.world_coord}, scene_xyd_coord={self.scene_xyd_coord},"
            f"confidence={self.confidence:.4f}, temporal_index={self.temporal_index}, joint_index={self.joint_index})"
        )

    def __str__(self) -> str:
        return f"JointPointNode([{self.joint_index}.{self.temporal_index}], world_coord={self.world_coord}, scene_xyd_coord={self.scene_xyd_coord} ({self.confidence:.2f}))"
