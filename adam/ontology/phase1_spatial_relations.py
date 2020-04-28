from typing import Generic, List, Mapping, Optional, TypeVar, Union

from adam.axis import GeonAxis
from attr import attrib, attrs
from attr.validators import in_, instance_of, optional
from vistautils.preconditions import check_arg

from adam.axes import AxesInfo, AxisFunction, GRAVITATIONAL_AXIS_FUNCTION


@attrs(frozen=True, slots=True, repr=False)
class Distance:
    """
    A distance of the sort used by Landau and Jackendoff
    to specify spatial regions.
    """

    name: str = attrib(validator=instance_of(str))

    def __repr__(self) -> str:
        return self.name


INTERIOR = Distance("interior")
"""
Figure is within the ground.
"""
EXTERIOR_BUT_IN_CONTACT = Distance("exterior-but-in-contact")
"""
Figure is outside the ground but contacting it.
"""
PROXIMAL = Distance("proximal")
"""
Figure is "near" the ground.
"""
DISTAL = Distance("distal")
"""
Figure is "far" from the ground.
"""

LANDAU_AND_JACKENDOFF_DISTANCES = [INTERIOR, EXTERIOR_BUT_IN_CONTACT, PROXIMAL, DISTAL]
"""
Distances used by Landau and Jackendoff in describing spatial relations.
"""

ReferenceObjectT = TypeVar("ReferenceObjectT")
NewObjectT = TypeVar("NewObjectT")


# @attrs(frozen=True, repr=False)
# class Axis(Generic[ReferenceObjectT]):
#     name: str = attrib(validator=instance_of(str))
#     reference_object: Optional[ReferenceObjectT] = attrib(kw_only=True)
#
#     @staticmethod
#     def primary_of(reference_object: ReferenceObjectT) -> "Axis[ReferenceObjectT]":
#         return Axis("primary", reference_object=reference_object)
#
#     def copy_remapping_objects(
#         self, object_map: Mapping[ReferenceObjectT, NewObjectT]
#     ) -> "Axis[" "NewObjectT]":
#         return Axis(
#             name=self.name,
#             reference_object=object_map[self.reference_object]
#             if self.reference_object
#             else None,
#         )
#
#     def accumulate_referenced_objects(
#         self, object_accumulator: List[ReferenceObjectT]
#     ) -> None:
#         r"""
#         Adds all objects referenced by this `Axis` to *object_accumulator*.
#         """
#         if self.reference_object:
#             object_accumulator.append(self.reference_object)
#
#     def __repr__(self) -> str:
#         if self.reference_object:
#             return f"{self.name}({self.reference_object})"
#         else:
#             return self.name


@attrs(frozen=True, repr=False)
class Direction(Generic[ReferenceObjectT]):
    r"""
    Represents the direction one object may have relative to another.

    This is used to specify `Region`\ s.
    """
    positive: bool = attrib(validator=instance_of(bool))
    """
    We need to standardize on what "positive" direction means. 
    It is clear for vertical axes but less clear for other things. 
    """
    relative_to_axis: Union[GeonAxis, AxisFunction[ReferenceObjectT]] = attrib()

    def copy_remapping_objects(
        self, object_map: Mapping[ReferenceObjectT, NewObjectT]
    ) -> "Direction[NewObjectT]":
        new_relative_to_axis: Union[GeonAxis, AxisFunction[NewObjectT]]
        if isinstance(self.relative_to_axis, AxisFunction):
            new_relative_to_axis = self.relative_to_axis.copy_remapping_objects(
                object_map
            )
        else:
            new_relative_to_axis = self.relative_to_axis

        return Direction(positive=self.positive, relative_to_axis=new_relative_to_axis)

    def relative_to_concrete_axis(
        self, axes_info: Optional[AxesInfo[ReferenceObjectT]]
    ) -> GeonAxis:
        if isinstance(self.relative_to_axis, GeonAxis):
            return self.relative_to_axis
        else:
            return self.relative_to_axis.to_concrete_axis(axes_info)

    def __repr__(self) -> str:
        polarity = "+" if self.positive else "-"
        return f"{polarity}{self.relative_to_axis}"

    def __attrs_post_init__(self) -> None:
        check_arg(isinstance(self.relative_to_axis, (GeonAxis, AxisFunction)))


GRAVITATIONAL_UP = Direction(positive=True, relative_to_axis=GRAVITATIONAL_AXIS_FUNCTION)
GRAVITATIONAL_DOWN = Direction(
    positive=False, relative_to_axis=GRAVITATIONAL_AXIS_FUNCTION
)


@attrs(frozen=True, repr=False, cache_hash=True)
class Region(Generic[ReferenceObjectT]):
    """
    A region of space perceived by the learner.

    We largely follow

    Barbara Landau and Ray Jackendoff. "'What' and 'where' in spatial language
    and spatial cognition. Brain and Behavioral Sciences (1993) 16:2.

    who analyze spatial relations in term of a `Distance` and `Direction`
    with respect to some *reference_object*.

    At least one of *distance* and *direction* must be specified.
    """

    reference_object: ReferenceObjectT = attrib()
    distance: Optional[Distance] = attrib(
        validator=optional(in_(LANDAU_AND_JACKENDOFF_DISTANCES)), default=None
    )
    direction: Optional[Direction[ReferenceObjectT]] = attrib(
        validator=optional(instance_of(Direction)), default=None
    )

    def copy_remapping_objects(
        self, object_map: Mapping[ReferenceObjectT, NewObjectT]
    ) -> "Region[NewObjectT]":
        return Region(
            reference_object=object_map[self.reference_object],
            distance=self.distance,
            direction=self.direction.copy_remapping_objects(object_map)
            if self.direction
            else None,
        )

    def accumulate_referenced_objects(
        self, object_accumulator: List[ReferenceObjectT]
    ) -> None:
        r"""
        Adds all objects referenced by this `Region` to *object_accumulator*.
        """
        object_accumulator.append(self.reference_object)
        if self.direction:
            if isinstance(self.direction.relative_to_axis, AxisFunction):
                self.direction.relative_to_axis.accumulate_referenced_objects(
                    object_accumulator
                )

    def __attrs_post_init__(self) -> None:
        check_arg(
            self.distance or self.direction,
            "A region must have either a distance or direction specified.",
        )

    def __repr__(self) -> str:
        parts = [str(self.reference_object)]
        if self.distance:
            parts.append(f"distance={self.distance}")
        if self.direction:
            parts.append(f"direction={self.direction}")
        return f"Region({','.join(parts)})"


@attrs(frozen=True, slots=True)
class PathOperator:
    name: str = attrib(validator=instance_of(str))


VIA = PathOperator("via")
TO = PathOperator("to")
TOWARD = PathOperator("toward")
FROM = PathOperator("from")
AWAY_FROM = PathOperator("away-from")


@attrs(frozen=True)
class SpatialPath(Generic[ReferenceObjectT]):
    operator: Optional[PathOperator] = attrib(
        validator=optional(instance_of(PathOperator))
    )
    reference_object: Union[ReferenceObjectT, Region[ReferenceObjectT]] = attrib()
    reference_axis: Optional[Union[GeonAxis, AxisFunction[ReferenceObjectT]]] = attrib(
        validator=optional(instance_of(AxisFunction)), default=None, kw_only=True
    )
    orientation_changed: bool = attrib(
        validator=instance_of(bool), default=False, kw_only=True
    )

    def __attrs_post_init__(self) -> None:
        # you either need a path operator
        #  or an orientation change around an axis
        #  (e.g. for rotation without translation)
        # weird conditional to make mypy happy
        if (
            not self.reference_object
            and not self.reference_axis
            and not self.orientation_changed
        ):
            raise RuntimeError(
                "A path must have at least one of a reference objects, "
                "a reference axis, or an orientation change"
            )
        if self.reference_axis:
            check_arg(isinstance(self.reference_axis, (GeonAxis, AxisFunction)))

    def copy_remapping_objects(
        self, object_mapping: Mapping[ReferenceObjectT, NewObjectT]
    ) -> "SpatialPath[NewObjectT]":
        new_reference_axis: Optional[Union[GeonAxis, AxisFunction[NewObjectT]]]
        if isinstance(self.reference_axis, AxisFunction):
            new_reference_axis = self.reference_axis.copy_remapping_objects(
                object_mapping
            )
        elif self.reference_axis:
            new_reference_axis = self.reference_axis
        else:
            new_reference_axis = None

        return SpatialPath(
            self.operator,
            self.reference_object.copy_remapping_objects(object_mapping)
            if isinstance(self.reference_object, Region)
            else object_mapping[self.reference_object],
            reference_axis=new_reference_axis,
            orientation_changed=self.orientation_changed,
        )

    def accumulate_referenced_objects(
        self, object_accumulator: List[ReferenceObjectT]
    ) -> None:
        r"""
        Adds all objects referenced by this `Region` to *object_accumulator*.
        """
        if isinstance(self.reference_object, Region):
            self.reference_object.accumulate_referenced_objects(object_accumulator)
        else:
            object_accumulator.append(self.reference_object)
        if self.reference_axis and not isinstance(self.reference_axis, GeonAxis):
            self.reference_axis.accumulate_referenced_objects(object_accumulator)
