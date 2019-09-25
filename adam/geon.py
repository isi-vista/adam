from abc import ABC, abstractmethod
from itertools import chain
from typing import Generic, Iterable, Mapping, TypeVar

from attr import attrib, attrs, evolve
from attr.validators import instance_of
from immutablecollections import (
    ImmutableDict,
    ImmutableSet,
    ImmutableSetMultiDict,
    immutableset,
    immutablesetmultidict,
)
from immutablecollections.converter_utils import (
    _to_immutabledict,
    _to_immutableset,
    _to_immutablesetmultidict,
)
from more_itertools import quantify

from adam.relation import Relation, flatten_relations


@attrs(frozen=True, slots=True, repr=False)
class CrossSectionSize:
    name: str = attrib(validator=instance_of(str))

    def __repr__(self) -> str:
        return self.name


@attrs(frozen=True, slots=True, repr=False)
class CrossSection:
    has_rotational_symmetry: bool = attrib(
        validator=instance_of(bool), default=False, kw_only=True
    )
    has_reflective_symmetry: bool = attrib(
        validator=instance_of(bool), default=False, kw_only=True
    )
    curved: bool = attrib(validator=instance_of(bool), default=False, kw_only=True)

    def __repr__(self) -> str:

        return (
            f"[{_sign(self.has_reflective_symmetry)}reflect-sym, "
            f"{_sign(self.has_rotational_symmetry)}rotate-sym, "
            f"{_sign(self.curved)}curved]"
        )


@attrs(frozen=True, slots=True, repr=False, cmp=False)
class GeonAxis:
    debug_name: str = attrib(validator=instance_of(str))
    curved: bool = attrib(validator=instance_of(bool), default=False)
    directed: bool = attrib(validator=instance_of(bool), default=True)
    aligned_to_gravitational = attrib(validator=instance_of(bool), default=False)

    def copy(self) -> "GeonAxis":
        return evolve(self)

    def __repr__(self) -> str:
        return (
            f"{self.debug_name}"
            f"[{_sign(self.curved)}curved, "
            f"{_sign(self.directed)}directed, "
            f"{_sign(self.curved)}aligned_to_gravity]"
        )


def directed(debug_name: str) -> GeonAxis:
    return GeonAxis(debug_name, directed=True)


def straight_up(debug_name: str) -> GeonAxis:
    return GeonAxis(debug_name, directed=True, aligned_to_gravitational=True)


def symmetric(debug_name: str) -> GeonAxis:
    return GeonAxis(debug_name, directed=False)


def symmetric_vertical(debug_name: str) -> GeonAxis:
    return GeonAxis(debug_name, directed=False, aligned_to_gravitational=True)


@attrs(slots=True, frozen=True)
class Axes:
    primary_axis: GeonAxis = attrib(validator=instance_of(GeonAxis), kw_only=True)
    orienting_axes: ImmutableSet[GeonAxis] = attrib(
        converter=_to_immutableset, kw_only=True
    )
    axis_relations: ImmutableSet[Relation[GeonAxis]] = attrib(
        converter=flatten_relations, default=immutableset(), kw_only=True
    )

    def __attrs_post_init__(self) -> None:
        num_gravitationally_aligned_axes = quantify(
            x.aligned_to_gravitational
            for x in chain((self.primary_axis,), self.orienting_axes)
        )
        if num_gravitationally_aligned_axes > 1:
            raise RuntimeError(
                f"A Geon cannot have multiple gravitationally aligned axes: {self}"
            )

    @property
    def all_axes(self) -> Iterable[GeonAxis]:
        return chain((self.primary_axis,), self.orienting_axes)

    def copy(self) -> "Axes":
        # world and learner axes are singletons
        if self is WORLD_AXES:
            return self
        elif self is LEARNER_AXES:
            return self
        else:
            return self.remap_axes({axis: axis.copy() for axis in self.all_axes})

    def remap_axes(self, axis_mapping: Mapping[GeonAxis, GeonAxis]) -> "Axes":
        return Axes(
            primary_axis=axis_mapping[self.primary_axis],
            orienting_axes=[axis_mapping[axis] for axis in self.orienting_axes],
            axis_relations=[
                relation.copy_remapping_objects(axis_mapping)
                for relation in self.axis_relations
            ],
        )


class HasAxes(ABC):
    @property
    @abstractmethod
    def axes(self) -> Axes:
        pass


@attrs(slots=True, frozen=True)
class Geon:
    cross_section: CrossSection = attrib(
        validator=instance_of(CrossSection), kw_only=True
    )
    cross_section_size: CrossSectionSize = attrib(
        validator=instance_of(CrossSectionSize), kw_only=True
    )
    axes: Axes = attrib(validator=instance_of(Axes), kw_only=True)
    generating_axis: GeonAxis = attrib(validator=instance_of(GeonAxis), kw_only=True)

    def copy(self) -> "Geon":
        axis_mapping = {axis: axis.copy() for axis in self.axes.all_axes}
        return Geon(
            cross_section=self.cross_section,
            cross_section_size=self.cross_section_size,
            axes=self.axes.remap_axes(axis_mapping),
            generating_axis=axis_mapping[self.generating_axis],
        )

    @generating_axis.default
    def _init_primary_axis(self) -> GeonAxis:
        return self.axes.primary_axis


def _sign(prop_val: bool) -> str:
    return "+" if prop_val else "-"


CONSTANT = CrossSectionSize("constant")
"""
Indicates the size of the cross-section of a geon 
remains roughly constant along its generating axis.
"""

SMALL_TO_LARGE = CrossSectionSize("small-to-large")
"""
Indicates the size of the cross-section of a geon 
increases along its generating axis.
"""

LARGE_TO_SMALL = CrossSectionSize("large-to-small")
"""
Indicates the size of the cross-section of a geon 
decreases along its generating axis.
"""

SMALL_TO_LARGE_TO_SMALL = CrossSectionSize("small-to-large-to-small")
"""
Indicates the size of the cross-section of a geon 
initially increases along the generating axis,
but then decreases.
"""


CIRCULAR = CrossSection(
    has_rotational_symmetry=True, has_reflective_symmetry=True, curved=True
)
SQUARE = CrossSection(
    has_rotational_symmetry=True, has_reflective_symmetry=True, curved=False
)
OVALISH = CrossSection(
    has_rotational_symmetry=False, has_reflective_symmetry=True, curved=True
)
RECTANGULAR = CrossSection(
    has_rotational_symmetry=False, has_reflective_symmetry=True, curved=False
)
IRREGULAR = CrossSection(
    has_reflective_symmetry=False, has_rotational_symmetry=False, curved=False
)

_ObjectT = TypeVar("_ObjectT", bound=HasAxes)


@attrs(frozen=True)
class AxesInfo(Generic[_ObjectT]):
    objects_to_axes: ImmutableDict[_ObjectT, Axes] = attrib(
        converter=_to_immutabledict, kw_only=True
    )
    axes_facing: ImmutableSetMultiDict[_ObjectT, GeonAxis] = attrib(
        converter=_to_immutablesetmultidict, default=immutablesetmultidict(), kw_only=True
    )


class AxisFunction(ABC, Generic[_ObjectT]):
    @abstractmethod
    def select_axis(self, axes_info: AxesInfo[_ObjectT]) -> GeonAxis:
        pass


@attrs(frozen=True)
class PrimaryAxisOfObject(Generic[_ObjectT], AxisFunction[_ObjectT]):
    _object: _ObjectT = attrib()

    def select_axis(
        self, axes_info: AxesInfo[_ObjectT]  # pylint:disable=unused-argument
    ) -> GeonAxis:
        return self._object.axes.primary_axis


GRAVITATIONAL_DOWN_TO_UP_AXIS = straight_up("gravitational-up")
SOUTH_TO_NORTH_AXIS = directed("south-to-north")
WEST_TO_EAST_AXIS = directed("west-to-east")
WORLD_AXES = Axes(
    primary_axis=GRAVITATIONAL_DOWN_TO_UP_AXIS,
    orienting_axes=[SOUTH_TO_NORTH_AXIS, WEST_TO_EAST_AXIS],
)
LEARNER_DOWN_TO_UP_AXIS = straight_up("learner-vertical")
LEARNER_LEFT_RIGHT_AXIS = directed("learner-left-to-right")
LEARNER_BACK_TO_FRONT_AXIS = directed("learner-back-to-front")
LEARNER_AXES = Axes(
    primary_axis=LEARNER_DOWN_TO_UP_AXIS,
    orienting_axes=[LEARNER_LEFT_RIGHT_AXIS, LEARNER_BACK_TO_FRONT_AXIS],
)
