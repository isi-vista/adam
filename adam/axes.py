from itertools import chain
from typing import Mapping, TypeVar, Generic, Any, Iterable, List

from attr import attrs, attrib
from attr.validators import instance_of
from immutablecollections import (
    ImmutableSetMultiDict,
    immutablesetmultidict,
    ImmutableSet,
    immutableset,
)
from immutablecollections.converter_utils import (
    _to_immutablesetmultidict,
    _to_immutableset,
)
from more_itertools import first, quantify
from typing_extensions import Protocol, runtime

from adam.axis import GeonAxis
from adam.relation import flatten_relations
from adam.remappable import CanRemapObjects


def directed(debug_name: str) -> GeonAxis:
    return GeonAxis(debug_name, directed=True)


def straight_up(debug_name: str) -> GeonAxis:
    return GeonAxis(debug_name, directed=True, aligned_to_gravitational=True)


def symmetric(debug_name: str) -> GeonAxis:
    return GeonAxis(debug_name, directed=False)


def symmetric_vertical(debug_name: str) -> GeonAxis:
    return GeonAxis(debug_name, directed=False, aligned_to_gravitational=True)


_ObjectT = TypeVar("_ObjectT")
_ObjectToT = TypeVar("_ObjectToT")


@attrs(frozen=True)
class AxesInfo(Generic[_ObjectT], CanRemapObjects[_ObjectT]):
    axes_facing: ImmutableSetMultiDict[_ObjectT, GeonAxis] = attrib(
        converter=_to_immutablesetmultidict, default=immutablesetmultidict()
    )

    def copy_remapping_objects(
        self, object_map: Mapping[_ObjectT, _ObjectToT]
    ) -> "AxesInfo[_ObjectToT]":
        return AxesInfo(
            immutablesetmultidict(
                (object_map[key], value) for (key, value) in self.axes_facing.items()
            )
        )


@runtime
class AxisFunction(Protocol, Generic[_ObjectT]):
    def to_concrete_axis(self, axes_info: AxesInfo[_ObjectT]) -> GeonAxis:
        pass

    def copy_remapping_objects(
        self, object_map: Mapping[_ObjectT, _ObjectToT]
    ) -> "AxisFunction[_ObjectToT]":
        pass

    def accumulate_referenced_objects(self, object_accumulator: List[_ObjectT]) -> None:
        pass


@attrs(frozen=True)
class PrimaryAxisOfObject(Generic[_ObjectT], AxisFunction[_ObjectT]):
    _object: _ObjectT = attrib()

    def to_concrete_axis(
        self, axes_info: AxesInfo[_ObjectT]  # pylint:disable=unused-argument
    ) -> GeonAxis:
        if not isinstance(self._object, HasAxes):
            raise RuntimeError(
                "Can only instantiate an axis function if the object is of a "
                "concrete type (e.g. perception or situation object)"
            )
        return self._object.axes.primary_axis

    def copy_remapping_objects(
        self, object_map: Mapping[_ObjectT, _ObjectToT]
    ) -> "PrimaryAxisOfObject[_ObjectToT]":
        return PrimaryAxisOfObject(object_map[self._object])


@attrs(frozen=True)
class FirstHorizontalAxisOfObject(Generic[_ObjectT], AxisFunction[_ObjectT]):
    _object: _ObjectT = attrib()

    def to_concrete_axis(
        self, axes_info: AxesInfo[_ObjectT]  # pylint:disable=unused-argument
    ) -> GeonAxis:
        if not isinstance(self._object, HasAxes):
            raise RuntimeError(
                "Can only instantiate an axis function if the object is of a "
                "concrete type (e.g. perception or situation object)"
            )
        return first(
            axis
            for axis in self._object.axes.all_axes
            if not axis.aligned_to_gravitational
        )

    def copy_remapping_objects(
        self, object_map: Mapping[_ObjectT, _ObjectToT]
    ) -> "FirstHorizontalAxisOfObject[_ObjectToT]":
        return FirstHorizontalAxisOfObject(object_map[self._object])


_GRAVITATIONAL_DOWN_TO_UP_AXIS = straight_up("gravitational-up")
_SOUTH_TO_NORTH_AXIS = directed("south-to-north")
_WEST_TO_EAST_AXIS = directed("west-to-east")
_LEARNER_DOWN_TO_UP_AXIS = straight_up("learner-vertical")
_LEARNER_LEFT_RIGHT_AXIS = directed("learner-left-to-right")
_LEARNER_BACK_TO_FRONT_AXIS = directed("learner-back-to-front")


@attrs(frozen=True)
class _GravitationalAxis(AxisFunction[Any]):
    def to_concrete_axis(
        self, axes_info: AxesInfo[Any]  # pylint:disable=unused-argument
    ) -> GeonAxis:
        return _GRAVITATIONAL_DOWN_TO_UP_AXIS

    def copy_remapping_objects(
        self, object_map: Mapping[_ObjectT, _ObjectToT]
    ) -> "_GravitationalAxis":
        return self


GRAVITATIONAL_AXIS_FUNCTION = _GravitationalAxis()


@attrs(slots=True, frozen=True)
class Axes:
    primary_axis: GeonAxis = attrib(validator=instance_of(GeonAxis), kw_only=True)
    orienting_axes: ImmutableSet[GeonAxis] = attrib(
        converter=_to_immutableset, kw_only=True
    )
    # TODO: fix typing issue below
    axis_relations: ImmutableSet["Relation[GeonAxis]"] = attrib(  # type: ignore
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


@runtime
class HasAxes(Protocol):
    axes: Axes


WORLD_AXES = Axes(
    primary_axis=_GRAVITATIONAL_DOWN_TO_UP_AXIS,
    orienting_axes=[_SOUTH_TO_NORTH_AXIS, _WEST_TO_EAST_AXIS],
)
LEARNER_AXES = Axes(
    primary_axis=_LEARNER_DOWN_TO_UP_AXIS,
    orienting_axes=[_LEARNER_LEFT_RIGHT_AXIS, _LEARNER_BACK_TO_FRONT_AXIS],
)
