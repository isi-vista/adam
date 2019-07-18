"""
Visual Representation based on

David Marr, Vision, Chapter 5 "Presenting Shapes for Recognition".

Marr's represent objects in terms of hierarchical structures with object-centered coordinate
systems, which he calls "models".  Marr's exposition does not sharply distinguish a
model-as-a-representation-of-a-particular-object-instance from
model-as-a-representation-of-an-object-class.  In code we need to be a little more explicit, so
we call the former a `Marr3dObject` and the latter a `Marr3dModel`.

Unfortunately, Python's lack of forward declarations means the classes in this module are not
declared in the best order for reading.  I suggest reading them in the following order:

These describe the model-as-representation-of-a-particular-object-instance (e.g. the particular
truck I see right now):
- `Marr3dObject`
- `Cylinder`
- `AdjunctRelation`

These describe model-as-a-representation-of-an-object-class (e.g. trucks in general):
- `Marr3dModel`
- `CylinderRange`
- `AdjunctRelationRange`

These describe the structures which can be used to match instances to class (that is, to identify
what it is you are looking at):
- MarrModelIndex
- SpecificityIndex
- SpecificityIndexNode

"""
from typing import Any, Callable, Mapping, TypeVar

from attr import attrib, attrs
from attr.validators import instance_of
from immutablecollections import immutabledict
from vistautils.range import Range


# TODO: move this to vistautils
def _positive_float(_, _2, value: Any) -> None:
    if not (isinstance(value, float) and value > 0.0):
        raise ValueError(f"{value} is not a positive float")


_T = TypeVar("_T")


# TODO: move this to vistautils
def _in_range(_range: Range[_T]) -> Callable[[Any, Any, Any], None]:
    def validator(_, attribute, value) -> None:
        if value not in _range:
            raise ValueError(
                f"Attribute {attribute}'s value is not in required range {_range}"
            )

    return validator


_ZERO_ONE_CLOSED = Range.closed(0.0, 1.0)


@attrs(frozen=True)
class Cylinder:
    """
    A cylinder, irrespective of orientation.

    Marr's representation builds objects up from generalized cylinders; right now we only
    represent cylinders with elliptical cross-sections.
    """

    length_in_meters: float = attrib(validator=_positive_float, kw_only=True)
    major_diameter_in_meters: float = attrib(validator=_positive_float, kw_only=True)
    minor_diameter_in_meters: float = attrib(validator=_positive_float, kw_only=True)


@attrs(frozen=True)
class CylinderRange:
    """
    Represents a set of possible cylinders.

    This is used for object recognition.
    """

    length_range_in_meters: float = attrib(validator=_positive_float, kw_only=True)
    diameter_range_in_meters: float = attrib(validator=_positive_float, kw_only=True)


@attrs(frozen=True, kw_only=True)
class AdjunctRelation:
    r"""
    Specifies the location and orientation of an cylinder :math:`S` with respect to another
    cylinder :math:`A`.

    "Two three-dimensional vectors are required to specify the position in space of one axis
    relative to another... The first vector, written in cylindrical coordinates
    :math:`(p, r, \Theta)`, defines the starting point of :math:`S` relative to :math:`A`; the
    second vector, written in spherical coordinates :math:`(\iota, \phi, s)`, specified
    :math:`S` itself.  We shall call the combined specification
    :math:`(p, r, \Theta, \iota, \phi, s)` an adjunct relation for :math:`S` relative to
    :math:`A`." ~ Marr, Vision, p. 308
    """
    p_relative_to_reference_length: float = attrib(validator=_positive_float)
    r_relative_to_reference_width: float = attrib(validator=_positive_float)
    theta_in_degrees: float = attrib(validator=_positive_float)
    iota_in_degrees: float = attrib(validator=_positive_float)
    phi_in_degrees: float = attrib(validator=_positive_float)
    s_relative_to_reference_length: float = attrib(validator=_positive_float)


@attrs(frozen=True, kw_only=True)
class AdjunctRelationRange:
    r"""
    Represents a set of acceptable adjunct relations for object identification.

    "Because the precision with which 3-D model scan represent a shape varies, it is appropriate to
    represent the angles and lengths that occur in an adjunct relation in a system that is also
    capable of variable precision, For instance, one might wish to state that a particular axis,
    like the arm component of the human 3-D model... is connected rather precisely at one end of
    the torso (that is, that value of :math:`p` is exactly :math:`0`), but with
    :math:`\Theta` only coarsely specified and with very little restriction on :math:`\iota`.
    """
    p_relative_to_reference_length: Range[float] = attrib(validator=instance_of(Range))
    r_relative_to_reference_width: Range[float] = attrib(validator=instance_of(Range))
    theta_in_degrees: Range[float] = attrib(validator=instance_of(Range))
    iota_in_degrees: Range[float] = attrib(validator=instance_of(Range))
    phi_in_degrees: Range[float] = attrib(validator=instance_of(Range))
    s_relative_to_reference_length: Range[float] = attrib(validator=instance_of(Range))


@attrs(frozen=True, kw_only=True)
class Marr3dObject:
    r"""
    A Marr-ian representation of a particular object instance (e.g. a particular truck
    I see, not trucks in general).

    An object is defined by:

    - A *bounding_cylinder* specifying a `Cylinder` (which Marr calls an "Axis") which bounds the
      object.

    - *components*, a :class:`typing.Mapping` of sub-objects to `AdjunctRelation`\ s describing
      their orientation relative to the primary model.

    - a *principal_cylinder* which defines the `Cylinder` that sub-object positions and
      orientations are defined with respect to. This is normally the *model_axis* but may
      differ in some cases. To use Marr's example, for a "human being" object it is more
      convenient to specify the sub-components with respect to the "torso" axis than the
      bounding cylinder axis.

    TODO: add orientation information - https://github.com/isi-vista/adam/issues/21
    """
    bounding_cylinder: Cylinder = attrib(validator=instance_of(Cylinder))
    principal_cylinder: Cylinder = attrib(validator=instance_of(Cylinder))
    components: Mapping["Marr3dObject", AdjunctRelation] = attrib(converter=immutabledict)


@attrs(frozen=True, kw_only=True)
class Marr3dModel:
    r"""
    A Marr-ian representation of a recognition model for a class of instances (e.g. "trucks in
    general", not a particular truck).

    A model is defined by:

    - A *bounding_cylinder_range* specifying the constraints on the possible sizes of the
      *bounding_cylinder* of any `Marr3dObject` matching this model.  Marr calls this the
      "model axis."

    - a *principal_cylinder_relative_to_bounding_cylinder* which gives the `AdjunctRelation`
      specifying the size and orientation of the principal cylinder for this model relative
      to the bounding cylinder.   Marr refers to this as the "principal axis"; the sizes and
      orientations of all sub-components are relative to this axis/cylinder.

      This adjunct relation often specifies a cylinder equal to the bounding cylinder, but may
      differ in some cases. To use Marr's example, for a "human being" object it is more
      convenient to specify the sub-components with respect to the "torso" cylinder/axis than the
      bounding cylinder cylinder/axis.

    - *components*, a :class:`typing.Mapping` of sub-models to `AdjunctRelation`\ s describing their
      orientation relative to the primary model.

    """

    bounding_cylinder_range: CylinderRange = attrib(validator=instance_of(CylinderRange))
    principal_cylinder_relative_to_bounding_cylinder: AdjunctRelation = attrib(
        validator=instance_of(AdjunctRelation)
    )
    components: Mapping["Marr3dModel", AdjunctRelation] = attrib(converter=immutabledict)


# @attrs(frozen=True, auto_attribs=True)
# class SpecificityIndexNode:
#     node_model: Marr3dModel
#     child_models: ImmutableSet["SpecificityIndexNode"]
#
#     def dominated_models(self) -> AbstractSet[Marr3dModel]:
#         ret: List[Marr3dModel] = [self.node_model]
#         for child_model in self.child_models:
#             ret.extend(child_model.dominated_models())
#         return immutableset(ret)
#
#
# @attrs(frozen=True)
# class SpecificityIndex:
#     root_models: ImmutableSet[SpecificityIndexNode]
#
#     def all_compatible_leaf_models(
#         self, probe_model: Marr3dModel
#     ) -> AbstractSet[Marr3dModel]:
#         pass
#
#     def most_specific_compatible_nodes(
#         self, probe_model: Marr3dModel
#     ) -> AbstractSet[SpecificityIndexNode]:
#         pass
#
#
# @attrs(frozen=True)
# class MarrModelIndex:
#     specificity_index: SpecificityIndex = attrib(validator=instance_of(SpecificityIndex))
#     # for the moment, we do not support the adjunct and specificity indices (p. 320),
#     # although the submodel points on Marr3dModel partly account for the adjunct index


# convenience methods for those of us who don't think in metric
def feet_to_meters(feet: float) -> float:
    return feet * 0.3048


def inches_to_meters(inches: float) -> float:
    return feet_to_meters(inches / 12.0)
