from abc import ABC, abstractmethod
from typing import Optional, Any, Tuple

from attr import attrs, attrib
from attr.validators import optional, instance_of, in_, deep_iterable
from immutablecollections import ImmutableSet
from immutablecollections.converter_utils import _to_tuple, _to_immutableset

from adam.axis import GeonAxis
from adam.geon import Geon, CrossSection
from adam.math_3d import Point
from adam.ontology import OntologyNode
from adam.ontology.phase1_spatial_relations import (
    Region,
    Distance,
    SpatialPath,
    PathOperator,
)
from adam.perception import ObjectPerception
from adam.perception.developmental_primitive_perception import RgbColorPerception
from adam.perception.perception_graph_nodes import (
    PerceptionGraphNode,
    UnwrappedPerceptionGraphNode,
    GraphNode,
    ObjectClusterNode,
    CategoricalNode,
    ContinuousNode,
    RgbColorNode,
    ObjectStroke,
    StrokeGNNRecognitionNode,
)

# Perception graph predicate nodes are defined below.
# These match the graph nodes defined above when using computer vision inputs
# Or ADAM objects when matching to the symbolic representation
from adam.semantics import ObjectSemanticNode
from adam.utilities import sign


class NodePredicate(ABC):
    r"""
    Super-class for pattern graph nodes.

    All `NodePredicate`\ s should compare non-equal to one another
    (if the are *attrs* classes, set *eq=False*).
    """

    weight: float = attrib(validator=instance_of(float), default=1.0)

    @abstractmethod
    def __call__(self, graph_node: PerceptionGraphNode) -> bool:
        """
        Determines whether a *graph_node* is matched by this predicate.
        """

    @abstractmethod
    def dot_label(self) -> str:
        """
        Node label to use when rendering patterns as graphs using *dot*.
        """

    @abstractmethod
    def is_equivalent(self, other: "NodePredicate") -> bool:
        """
        Compares two predicates and return true if they are equivalent
        """

    def matches_predicate(self, predicate_node: "NodePredicate") -> bool:
        """
        Determines whether a NodePredicate matches another Node Predicate
        """


@attrs(frozen=True, slots=True, eq=False)
class AnyGraphNodePredicate(NodePredicate):
    """
    Matches any node of type `GraphNode`.
    """

    def __call__(self, graph_node: PerceptionGraphNode) -> bool:
        return isinstance(graph_node, GraphNode)

    def dot_label(self) -> str:
        return "GraphNode(*)"

    def is_equivalent(self, other) -> bool:
        return isinstance(other, AnyGraphNodePredicate)

    def matches_predicate(self, predicate_node: NodePredicate) -> bool:
        return isinstance(predicate_node, AnyGraphNodePredicate)


@attrs(frozen=True, slots=True, eq=False)
class AnyObjectPredicate(NodePredicate):
    """
    Matches any node of type `ObjectClusterNode`.
    """

    def __call__(self, graph_node: PerceptionGraphNode) -> bool:
        return isinstance(graph_node, ObjectClusterNode)

    def dot_label(self) -> str:
        return "ObjectClusterNode(*)"

    def is_equivalent(self, other) -> bool:
        return isinstance(other, AnyObjectPredicate)

    def matches_predicate(self, predicate_node: NodePredicate) -> bool:
        return isinstance(predicate_node, AnyObjectPredicate)


@attrs(frozen=True, slots=True, eq=False)
class CategoricalPredicate(NodePredicate):
    """
    Matches a node where the Categorical value is the same.
    """

    value: str = attrib(validator=instance_of(str))

    def __call__(self, graph_node: PerceptionGraphNode) -> bool:
        if isinstance(graph_node, CategoricalNode):
            return self.value == graph_node.value
        return False

    @staticmethod
    def from_node(categorical_node: CategoricalNode) -> "CategoricalPredicate":
        return CategoricalPredicate(value=categorical_node.value)

    def dot_label(self) -> str:
        return f"CategoryFeature(value={self.value})"

    def is_equivalent(self, other) -> bool:
        return isinstance(other, CategoricalPredicate)

    def matches_predicate(self, predicate_node: NodePredicate) -> bool:
        if isinstance(predicate_node, CategoricalPredicate):
            return predicate_node.value == self.value
        return False


@attrs(frozen=True, slots=True, eq=False)
class ContinuousPredicate(NodePredicate):
    """
    Matches a node where the value is within the given tolerance
    """

    label: str = attrib(validator=instance_of(str))
    value: float = attrib(validator=instance_of(float))
    tolerance: float = attrib(validator=instance_of(float))

    def __call__(self, graph_node: PerceptionGraphNode) -> bool:
        if isinstance(graph_node, ContinuousNode):
            return (
                self.label == graph_node.label
                and self.value - self.tolerance
                <= graph_node.value
                <= self.value + self.tolerance
            )
        return False

    @staticmethod
    def from_node(node: ContinuousNode) -> "ContinuousPredicate":
        return ContinuousPredicate(
            label=node.label,
            value=node.value,
            tolerance=0.25,  # Maybe change this in the future ?
        )

    def dot_label(self) -> str:
        return f"ContinuousFeature(label={self.label}, value={self.value}, tolerance={self.tolerance})"

    def is_equivalent(self, other) -> bool:
        return isinstance(other, ContinuousPredicate)

    def matches_predicate(self, predicate_node: NodePredicate) -> bool:
        if isinstance(predicate_node, ContinuousPredicate):
            return (
                self.label == predicate_node.label
                and self.value - self.tolerance
                <= predicate_node.value
                <= self.value + self.tolerance
            )
        return False


@attrs(frozen=True, slots=True, eq=False)
class RgbColorPredicate(NodePredicate):
    """
    Matches a node where the RGB value matches exactly.
    """

    red: int = attrib(validator=in_(range(0, 255)))
    green: int = attrib(validator=in_(range(0, 255)))
    blue: int = attrib(validator=in_(range(0, 255)))

    def __call__(self, graph_node: PerceptionGraphNode) -> bool:
        if isinstance(graph_node, RgbColorNode):
            return (
                self.red == graph_node.red
                and self.blue == graph_node.blue
                and self.green == graph_node.green
            )
        return False

    def dot_label(self) -> str:
        return f"RGBColorFeature(red={self.red}, green={self.green}, blue={self.blue})"

    def is_equivalent(self, other) -> bool:
        return isinstance(other, RgbColorPredicate)

    def matches_predicate(self, predicate_node: NodePredicate) -> bool:
        if isinstance(predicate_node, RgbColorPredicate):
            return (
                self.red == predicate_node.red
                and self.blue == predicate_node.blue
                and self.green == predicate_node.green
            )
        return False

    @staticmethod
    def from_node(node: RgbColorNode) -> "RgbColorPredicate":
        return RgbColorPredicate(red=node.red, green=node.green, blue=node.blue)


@attrs(frozen=True, slots=True, eq=False)
class ObjectStrokePredicate(NodePredicate):
    """Matches an Object Stroke"""

    stroke_normalized_coordinates: ImmutableSet[Point] = attrib(
        validator=deep_iterable(instance_of(Point)), converter=_to_immutableset
    )

    def __call__(self, graph_node: PerceptionGraphNode) -> bool:
        # TODO: Add more strict alignment of stroke normalized values
        # https://github.com/isi-vista/adam/issues/1051
        return isinstance(graph_node, ObjectStroke)

    def dot_label(self) -> str:
        return f"ObjectStroke({', '.join(f'{point}' for point in self.stroke_normalized_coordinates)})"

    def is_equivalent(self, other) -> bool:
        return isinstance(other, ObjectStrokePredicate)

    def matches_predicate(self, predicate_node: "NodePredicate") -> bool:
        return isinstance(predicate_node, ObjectStrokePredicate)

    @staticmethod
    def from_node(node: ObjectStroke) -> "ObjectStrokePredicate":
        return ObjectStrokePredicate(
            stroke_normalized_coordinates=node.normalized_coordinates
        )


@attrs(frozen=True, slots=True, eq=False)
class StrokeGNNRecognitionPredicate(NodePredicate):
    """Matches a Stroke GNN recognition."""

    recognized_object: str = attrib(validator=instance_of(str))

    def __call__(self, graph_node: PerceptionGraphNode) -> bool:
        return (
            graph_node.object_recognized == self.recognized_object
            if isinstance(graph_node, StrokeGNNRecognitionNode)
            else False
        )

    def dot_label(self) -> str:
        return f"StrokeGNNRecognition(object={self.recognized_object})"

    def is_equivalent(self, other: "NodePredicate") -> bool:
        return isinstance(other, StrokeGNNRecognitionPredicate)

    def matches_predicate(self, predicate_node: "NodePredicate") -> bool:
        return (
            predicate_node.recognized_object == self.recognized_object
            if isinstance(predicate_node, StrokeGNNRecognitionPredicate)
            else False
        )


# In an effort to not break previous symbolic space for ADAM graphs the previous predicates
# have been left alone here
@attrs(frozen=True, slots=True, eq=False)
class AnyNodePredicate(NodePredicate):
    """
    Matches any node whatsoever.
    """

    def __call__(self, graph_node: PerceptionGraphNode) -> bool:
        return True

    def dot_label(self) -> str:
        return "*"

    def is_equivalent(self, other) -> bool:
        return isinstance(other, AndNodePredicate)

    def matches_predicate(self, predicate_node: "NodePredicate") -> bool:
        return isinstance(predicate_node, AnyNodePredicate)


@attrs(frozen=True, slots=True, eq=False)
class AnyObjectPerception(NodePredicate):
    """
    Matches any `ObjectPerception` node.
    """

    debug_handle: Optional[str] = attrib(validator=optional(instance_of(str)))

    def __call__(self, graph_node: PerceptionGraphNode) -> bool:
        return isinstance(graph_node, ObjectPerception)

    def dot_label(self) -> str:
        if self.debug_handle is not None:
            debug_handle_str = f"[{self.debug_handle}]"
        else:
            debug_handle_str = ""
        return f"*obj{debug_handle_str}"

    def is_equivalent(self, other) -> bool:
        return isinstance(other, AnyObjectPerception) or isinstance(
            other, ObjectSemanticNode
        )

    def matches_predicate(self, predicate_node: "NodePredicate") -> bool:
        return isinstance(predicate_node, AnyObjectPerception)


def unwrap_if_necessary(graph_node: PerceptionGraphNode) -> UnwrappedPerceptionGraphNode:
    """
    Some nodes might be wrapped in tuples together with IDs to prevent them
    from being compared equal and collapsed in perception graphs.
    This method removes that wrapping.
    """
    if isinstance(graph_node, tuple):
        return graph_node[0]
    else:
        return graph_node


@attrs(frozen=True, slots=True, eq=False)
class AxisPredicate(NodePredicate):
    """
    Represents constraints on an axis given in a `PerceptionGraphPattern`
    """

    curved: Optional[bool] = attrib(validator=optional(instance_of(bool)))
    directed: Optional[bool] = attrib(validator=optional(instance_of(bool)))
    aligned_to_gravitational: Optional[bool] = attrib(
        validator=optional(instance_of(bool))
    )

    def __call__(self, graph_node: PerceptionGraphNode) -> bool:
        unwrapped_graph_node = unwrap_if_necessary(graph_node)

        if isinstance(unwrapped_graph_node, GeonAxis):
            if self.curved is not None and self.curved != unwrapped_graph_node.curved:
                return False
            if (
                self.directed is not None
                and self.directed != unwrapped_graph_node.directed
            ):
                return False
            if (
                self.aligned_to_gravitational is not None
                and self.aligned_to_gravitational
                != unwrapped_graph_node.aligned_to_gravitational
            ):
                return False
            return True
        else:
            return False

    @staticmethod
    def from_axis(axis_to_match: GeonAxis) -> "AxisPredicate":
        return AxisPredicate(
            curved=axis_to_match.curved,
            directed=axis_to_match.directed,
            aligned_to_gravitational=axis_to_match.aligned_to_gravitational,
        )

    def dot_label(self) -> str:
        constraints = []

        if self.curved is not None:
            constraints.append(f"{sign(self.curved)}curved")
        if self.directed is not None:
            constraints.append(f"{sign(self.directed)}directed")
        if self.aligned_to_gravitational is not None:
            constraints.append(f"{sign(self.aligned_to_gravitational)}grav_aligned")

        return f"axis({', '.join(constraints)})"

    def is_equivalent(self, other) -> bool:
        if isinstance(other, AxisPredicate):
            return (
                self.aligned_to_gravitational == other.aligned_to_gravitational
                and self.curved == other.curved
                and self.directed == other.directed
            )
        return False

    def matches_predicate(self, predicate_node: "NodePredicate") -> bool:
        if isinstance(predicate_node, AxisPredicate):
            if self.curved != predicate_node.curved:
                return False
            if self.directed != predicate_node.directed:
                return False
            if self.aligned_to_gravitational != predicate_node.aligned_to_gravitational:
                return False
            else:
                return True
        else:
            return False


@attrs(frozen=True, slots=True, eq=False)
class GeonPredicate(NodePredicate):
    """
    Represents constraints on a `Geon` given in a `PerceptionGraphPattern`
    """

    template_geon: Geon = attrib(validator=instance_of(Geon))

    def __call__(self, graph_node: PerceptionGraphNode) -> bool:
        # geons might be wrapped in tuples with their id()
        # in order to simulate comparison by object ID.
        unwrapped_graph_node = unwrap_if_necessary(graph_node)

        if isinstance(unwrapped_graph_node, Geon):
            return (
                self.template_geon.cross_section == unwrapped_graph_node.cross_section
                and self.template_geon.cross_section_size
                == unwrapped_graph_node.cross_section_size
            )
        else:
            return False

    def dot_label(self) -> str:
        return f"geon({self.template_geon})"

    @staticmethod
    def exactly_matching(geon: Geon) -> "GeonPredicate":
        return GeonPredicate(geon)

    def is_equivalent(self, other) -> bool:
        if isinstance(other, GeonPredicate):
            return (
                self.template_geon.axes.axis_relations
                == other.template_geon.axes.axis_relations
                and self.template_geon.axes.orienting_axes
                == other.template_geon.axes.orienting_axes
                and self.template_geon.axes.primary_axis
                == other.template_geon.axes.primary_axis
                and self.template_geon.cross_section.curved
                == other.template_geon.cross_section.curved
                and self.template_geon.cross_section.has_reflective_symmetry
                == other.template_geon.cross_section.has_reflective_symmetry
                and self.template_geon.cross_section.has_rotational_symmetry
                == other.template_geon.cross_section.has_rotational_symmetry
                and self.template_geon.cross_section_size.name
                == other.template_geon.cross_section_size.name
                and self.template_geon.generating_axis.curved
                == other.template_geon.generating_axis.curved
                and self.template_geon.generating_axis.directed
                == other.template_geon.generating_axis.directed
                and self.template_geon.generating_axis.aligned_to_gravitational
                == other.template_geon.generating_axis.aligned_to_gravitational
            )
        return False

    def matches_predicate(self, predicate_node: "NodePredicate") -> bool:
        if isinstance(predicate_node, GeonPredicate):
            return (
                self.template_geon.cross_section
                == predicate_node.template_geon.cross_section
                and self.template_geon.cross_section_size
                == predicate_node.template_geon.cross_section_size
            )
        else:
            return False


@attrs(frozen=True, slots=True, eq=False)
class CrossSectionPredicate(NodePredicate):
    """
    Represents constraints on a `Geon` given in a `PerceptionGraphPattern`
    """

    cross_section: CrossSection = attrib(validator=instance_of(CrossSection))

    def __call__(self, graph_node: PerceptionGraphNode) -> bool:

        unwrapped_graph_node = unwrap_if_necessary(graph_node)

        if isinstance(unwrapped_graph_node, CrossSection):
            return self.cross_section == unwrapped_graph_node
        else:
            return False

    def dot_label(self) -> str:
        return f"cross-section({self.cross_section})"

    @staticmethod
    def exactly_matching(cs: CrossSection) -> "CrossSectionPredicate":
        return CrossSectionPredicate(cs)

    def is_equivalent(self, other) -> bool:
        if isinstance(other, CrossSectionPredicate):
            return (
                self.cross_section.curved == other.cross_section.curved
                and self.cross_section.has_reflective_symmetry
                == other.cross_section.has_reflective_symmetry
                and self.cross_section.has_rotational_symmetry
                == other.cross_section.has_rotational_symmetry
            )
        return False

    def matches_predicate(self, predicate_node: "NodePredicate") -> bool:
        if isinstance(predicate_node, CrossSectionPredicate):
            return self.cross_section == predicate_node.cross_section
        else:
            return False


@attrs(frozen=True, slots=True, eq=False)
class RegionPredicate(NodePredicate):
    """
    Represents constraints on a `Region` given in a `PerceptionGraphPattern`.
    """

    distance: Optional[Distance] = attrib(validator=optional(instance_of(Distance)))

    def __call__(self, graph_node: PerceptionGraphNode) -> bool:
        # regions might be wrapped in tuples with their id()
        # in order to simulate comparison by object ID.
        unwrapped_graph_node = unwrap_if_necessary(graph_node)
        return (
            isinstance(unwrapped_graph_node, Region)
            and self.distance == unwrapped_graph_node.distance
        )

    def dot_label(self) -> str:
        return f"dist({self.distance})"

    @staticmethod
    def matching_distance(region: Region[Any]) -> "RegionPredicate":
        return RegionPredicate(region.distance)

    def is_equivalent(self, other) -> bool:
        if isinstance(other, RegionPredicate):
            if self.distance is None and other.distance is None:
                return True
            elif self.distance is not None and other.distance is not None:
                return self.distance.name == other.distance.name
        return False

    def matches_predicate(self, predicate_node: "NodePredicate") -> bool:
        if isinstance(predicate_node, RegionPredicate):
            return self.distance == predicate_node.distance
        else:
            return False


@attrs(frozen=True, slots=True, eq=False)
class IsOntologyNodePredicate(NodePredicate):
    property_value: OntologyNode = attrib(validator=instance_of(OntologyNode))

    def __call__(self, graph_node: PerceptionGraphNode) -> bool:
        unwrapped_graph_node = unwrap_if_necessary(graph_node)
        return self.property_value == unwrapped_graph_node

    def dot_label(self) -> str:
        return f"{self.property_value.handle}"

    def is_equivalent(self, other) -> bool:
        if isinstance(other, IsOntologyNodePredicate):
            return self.property_value == other.property_value
        return False

    def matches_predicate(self, predicate_node: "NodePredicate") -> bool:
        if isinstance(predicate_node, IsOntologyNodePredicate):
            return self.property_value == predicate_node.property_value
        return False


@attrs(frozen=True, slots=True, eq=False)
class IsColorNodePredicate(NodePredicate):
    color: RgbColorPerception = attrib(validator=instance_of(RgbColorPerception))

    def __call__(self, graph_node: PerceptionGraphNode) -> bool:
        unwrapped_graph_node = unwrap_if_necessary(graph_node)
        if isinstance(unwrapped_graph_node, RgbColorPerception):
            return (
                (unwrapped_graph_node.red == self.color.red)
                and (unwrapped_graph_node.blue == self.color.blue)
                and (unwrapped_graph_node.green == self.color.green)
            )
        return False

    def matches_predicate(self, predicate_node: "NodePredicate") -> bool:
        if isinstance(predicate_node, IsColorNodePredicate):
            return (
                (predicate_node.color.red == self.color.red)
                and (predicate_node.color.blue == self.color.blue)
                and (predicate_node.color.green == self.color.green)
            )
        return False

    def dot_label(self) -> str:
        return f"{self.color.hex}"

    def is_equivalent(self, other) -> bool:
        if isinstance(other, IsColorNodePredicate):
            return self.color.hex == other.color.hex
        return False


@attrs(frozen=True, slots=True, eq=False)
class AndNodePredicate(NodePredicate):
    """
    `NodePredicate` which matches if all its *sub_predicates* match.
    """

    sub_predicates: Tuple[NodePredicate, ...] = attrib(converter=_to_tuple)

    def __call__(self, graph_node: PerceptionGraphNode) -> bool:
        return all(sub_predicate(graph_node) for sub_predicate in self.sub_predicates)

    def dot_label(self) -> str:
        return " & ".join(sub_pred.dot_label() for sub_pred in self.sub_predicates)

    def is_equivalent(self, other) -> bool:
        if isinstance(other, AndNodePredicate) and len(self.sub_predicates) == len(
            other.sub_predicates
        ):
            return all(
                any(pred1.is_equivalent(pred2) for pred1 in self.sub_predicates)
                for pred2 in other.sub_predicates
            )
        return False

    def matches_predicate(self, predicate_node: "NodePredicate") -> bool:
        raise NotImplementedError(
            "Matches Predicate between AndNodePredicate is not yet implemented"
        )


@attrs(frozen=True, slots=True, eq=False)
class ObjectSemanticNodePerceptionPredicate(NodePredicate):
    """
    `NodePredicate` which matches if the node is of this type
    """

    def __call__(self, graph_node: PerceptionGraphNode) -> bool:
        return isinstance(graph_node, ObjectSemanticNode)

    def dot_label(self) -> str:
        return "*[matched-obj]"

    def matches_predicate(self, predicate_node: "NodePredicate") -> bool:
        return isinstance(predicate_node, ObjectSemanticNodePerceptionPredicate)

    def is_equivalent(self, other) -> bool:
        return isinstance(other, ObjectSemanticNodePerceptionPredicate)


@attrs(frozen=True, slots=True, eq=False)
class IsPathPredicate(NodePredicate):
    """
    Matches any `SpatialPath` node.
    """

    def __call__(self, graph_node: PerceptionGraphNode) -> bool:
        return isinstance(graph_node, SpatialPath)

    def dot_label(self) -> str:
        return "* [path]"

    def is_equivalent(self, other) -> bool:
        return isinstance(other, IsPathPredicate)

    def matches_predicate(self, predicate_node: "NodePredicate") -> bool:
        return isinstance(predicate_node, IsPathPredicate)


@attrs(frozen=True, slots=True, eq=False)
class PathOperatorPredicate(NodePredicate):
    r"""
    Predicate to match against `PathOperator`\ s
    """
    reference_path_operator: PathOperator = attrib(validator=instance_of(PathOperator))

    def __call__(self, graph_node: PerceptionGraphNode) -> bool:

        return unwrap_if_necessary(graph_node) == self.reference_path_operator

    def dot_label(self) -> str:
        return self.reference_path_operator.name

    def is_equivalent(self, other) -> bool:
        return (
            isinstance(other, PathOperatorPredicate)
            and other.reference_path_operator == self.reference_path_operator
        )

    def matches_predicate(self, predicate_node: "NodePredicate") -> bool:
        return (
            isinstance(predicate_node, PathOperatorPredicate)
            and predicate_node.reference_path_operator == self.reference_path_operator
        )
