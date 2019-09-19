from typing import Any, Generic, Mapping, Optional, TypeVar, List

from attr import attrib, attrs
from attr.validators import in_, instance_of, optional
from vistautils.preconditions import check_arg


@attrs(frozen=True, slots=True, repr=False)
class Distance:
    """
    A distance of the sort used by Landau and Jackendoff
    to specify spatial regions.
    """

    name: str = attrib(validator=instance_of(str))

    def __repr__(self) -> str:
        return self.name


INTERIOR = Distance("interor")
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


@attrs(frozen=True, repr=False)
class Axis(Generic[ReferenceObjectT]):
    name: str = attrib(validator=instance_of(str))
    reference_object: Optional[ReferenceObjectT] = attrib(kw_only=True)

    @staticmethod
    def primary_of(reference_object: ReferenceObjectT) -> "Axis[ReferenceObjectT]":
        return Axis("primary", reference_object=reference_object)

    def copy_remapping_objects(
        self, object_map: Mapping[ReferenceObjectT, NewObjectT]
    ) -> "Axis[" "NewObjectT]":
        return Axis(
            name=self.name,
            reference_object=object_map[self.reference_object]
            if self.reference_object
            else None,
        )

    def __repr__(self) -> str:
        if self.reference_object:
            return f"{self.name}({self.reference_object})"
        else:
            return self.name


GRAVITATIONAL_AXIS: Axis[Any] = Axis("gravitational", reference_object=None)


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
    relative_to_axis: Axis[ReferenceObjectT] = attrib(validator=instance_of(Axis))

    def copy_remapping_objects(
        self, object_map: Mapping[ReferenceObjectT, NewObjectT]
    ) -> "Direction[NewObjectT]":
        return Direction(
            positive=self.positive,
            relative_to_axis=self.relative_to_axis.copy_remapping_objects(object_map),
        )

    def __repr__(self) -> str:
        polarity = "+" if self.positive else "-"
        return f"{polarity}{self.relative_to_axis}"


@attrs(frozen=True, repr=False)
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
        if self.direction and self.direction.relative_to_axis.reference_object:
            object_accumulator.append(self.direction.relative_to_axis.reference_object)

    def __attrs_post_init__(self) -> None:
        check_arg(
            self.distance or self.direction,
            "A region must have either a " "distance or direction specified.",
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
    reference_object: ReferenceObjectT = attrib()
    reference_axis: Optional[Axis[ReferenceObjectT]] = attrib(
        validator=optional(instance_of(Axis)), default=None, kw_only=True
    )
    orientation_changed: bool = attrib(
        validator=instance_of(bool), default=False, kw_only=True
    )

    def __attrs_post_init__(self) -> None:
        # you either need a path operator
        #  or an orientation change around an axis
        #  (e.g. for rotation without translation)
        check_arg(
            self.operator
            or all((self.reference_object, self.reference_axis, self.orientation_changed))
        )

    def copy_remapping_objects(
        self, object_mapping: Mapping[ReferenceObjectT, NewObjectT]
    ) -> "SpatialPath[NewObjectT]":
        return SpatialPath(
            self.operator,
            object_mapping[self.reference_object],
            reference_axis=self.reference_axis.copy_remapping_objects(object_mapping)
            if self.reference_axis
            else None,
            orientation_changed=self.orientation_changed,
        )
