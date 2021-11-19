from abc import ABC
from typing import Union, Tuple, Any, Optional

from attr import attrs, attrib
from attr.validators import instance_of, in_, optional, deep_iterable
from immutablecollections import ImmutableSet
from immutablecollections.converter_utils import _to_immutableset

from adam.axis import GeonAxis
from adam.geon import Geon, CrossSection
from adam.math_3d import Point
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


@attrs(frozen=True, slots=True, eq=False)
# Perception graph nodes
class GraphNode(ABC):
    """Super-class for all perception graph nodes, useful for types."""

    weight: float = attrib(validator=instance_of(float))


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

    cluster_id: int = attrib(validator=instance_of(int))
    viewpoint_id: int = attrib(validator=instance_of(int))
    center_x: Optional[float] = attrib(validator=optional(instance_of(float)))
    center_y: Optional[float] = attrib(validator=optional(instance_of(float)))


@attrs(frozen=True, slots=True, eq=False)
class CategoricalNode(GraphNode):
    """A node representing a categorical value feature"""

    value: str = attrib(validator=instance_of(str))


@attrs(frozen=True, slots=True, eq=False)
class ContinuousNode(GraphNode):
    """A node representing a continuous value feature."""

    label: str = attrib(validator=instance_of(str))
    value: float = attrib(validator=instance_of(float))


@attrs(frozen=True, slots=True, eq=False)
class RgbColorNode(GraphNode):
    """A node representing an RGB perception value."""

    red: int = attrib(validator=in_(range(0, 255)))
    green: int = attrib(validator=in_(range(0, 255)))
    blue: int = attrib(validator=in_(range(0, 255)))
