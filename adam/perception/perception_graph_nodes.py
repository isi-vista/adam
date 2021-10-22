from abc import ABC
from typing import Union, Tuple, Any, Optional

from attr import attrs, attrib
from attr.validators import instance_of, in_, optional

from adam.axis import GeonAxis
from adam.geon import Geon, CrossSection
from adam.ontology import OntologyNode
from adam.ontology.phase1_spatial_relations import (
    Region,
    SpatialPath,
    PathOperator,
)
from adam.perception import ObjectPerception
from adam.semantics import ObjectSemanticNode


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
    ObjectSemanticNode,
    SpatialPath[ObjectPerception],
    PathOperator,
    GraphNode,
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
    ObjectSemanticNode,
    SpatialPath[ObjectPerception],
    PathOperator,
    GraphNode,
]


@attrs(frozen=True, slots=True, eq=False)
class ObjectNode(GraphNode):
    """A node representing a source of an object cluster perception."""

    _cluster_id: int = attrib(validator=instance_of(int))
    _viewpoint_id: int = attrib(validator=instance_of(int))
    center_x: Optional[float] = attrib(validator=optional(instance_of(float)))
    center_y: Optional[float] = attrib(validator=optional(instance_of(float)))


@attrs(frozen=True, slots=True, eq=False)
class CategoricalNode(GraphNode):
    """A node representing a categorical value feature"""

    value: str = attrib(validator=instance_of(str))


@attrs(frozen=True, slots=True, eq=False)
class ContinuousNode(GraphNode):
    """A node representing a continuous value feature."""

    value: float = attrib(validator=instance_of(float))


@attrs(frozen=True, slots=True, eq=False)
class RgbColorNode(GraphNode):
    """A node representing an RGB perception value."""

    red: int = attrib(validator=in_(range(0, 255)))
    green: int = attrib(validator=in_(range(0, 255)))
    blue: int = attrib(validator=in_(range(0, 255)))
