from typing import Mapping, TypeVar, Generic, Any

from attr import attrs, attrib
from immutablecollections import (
    ImmutableSetMultiDict,
    immutablesetmultidict,
)
from immutablecollections.converter_utils import (
    _to_immutablesetmultidict,
)
from more_itertools import first
from typing_extensions import Protocol

from adam.axis import GeonAxis
from adam.object_axes import HasAxes
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
        converter=_to_immutablesetmultidict, default=immutablesetmultidict(), kw_only=True
    )

    def copy_remapping_objects(
        self, object_map: Mapping[_ObjectT, _ObjectToT]
    ) -> "AxesInfo[_ObjectToT]":
        return AxesInfo(
            immutablesetmultidict(
                (object_map[key], value) for (key, value) in self.axes_facing.items()
            )
        )


class AxisFunction(Protocol, Generic[_ObjectT], CanRemapObjects[_ObjectT]):
    def to_concrete_axis(self, axes_info: AxesInfo[_ObjectT]) -> GeonAxis:
        pass

    def copy_remapping_objects(
        self, object_map: Mapping[_ObjectT, _ObjectToT]
    ) -> "AxisFunction[_ObjectToT]":
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
