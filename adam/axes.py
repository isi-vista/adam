from itertools import chain
from typing import Any, Generic, Iterable, List, Mapping, Optional, TypeVar

from attr import attrib, attrs
from attr.validators import in_, instance_of
from immutablecollections import (
    ImmutableSet,
    ImmutableSetMultiDict,
    immutableset,
    immutablesetmultidict,
)
from immutablecollections.converter_utils import (
    _to_immutableset,
    _to_immutablesetmultidict,
)
from more_itertools import only, quantify
from typing_extensions import Protocol, runtime
from vistautils.range import Range

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


@attrs(frozen=True, cache_hash=True)
class AxesInfo(Generic[_ObjectT], CanRemapObjects[_ObjectT]):
    addressee: Optional[_ObjectT] = attrib(default=None)
    axes_facing: ImmutableSetMultiDict[_ObjectT, GeonAxis] = attrib(
        converter=_to_immutablesetmultidict, default=immutablesetmultidict()
    )

    def copy_remapping_objects(
        self, object_map: Mapping[_ObjectT, _ObjectToT]
    ) -> "AxesInfo[_ObjectToT]":
        return AxesInfo(
            addressee=object_map[self.addressee] if self.addressee else None,
            axes_facing=immutablesetmultidict(
                (object_map[key], value) for (key, value) in self.axes_facing.items()
            ),
        )


@runtime
class AxisFunction(Protocol, Generic[_ObjectT]):
    r"""
    A procedure for selecting a particular `GeonAxis`.

    This is used in defining the semantics of prepositions and verbs
    and for defining the spatial relations between parts of an object in
    `ObjectStructuralSchema`\ ta.
    """

    def to_concrete_axis(self, axes_info: Optional[AxesInfo[_ObjectT]]) -> GeonAxis:
        """
        Select a particular concrete axis.

        This function will be provided with an `AxesInfo` object in concrete situations
        which can be used to determing the relationship of object axes to the speaker
        and the learner.
        However, this information is not available in more abstract contexts,
        like `ObjectStructuralSchema`,
        and the `AxisFunction` should throw an exception if called in such a way.
        """

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
        self, axes_info: Optional[AxesInfo[_ObjectT]]  # pylint:disable=unused-argument
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
class HorizontalAxisOfObject(Generic[_ObjectT], AxisFunction[_ObjectT]):
    _object: _ObjectT = attrib()
    _index: int = attrib(validator=in_(Range.closed(0, 1)))

    def to_concrete_axis(
        self, axes_info: Optional[AxesInfo[_ObjectT]]  # pylint:disable=unused-argument
    ) -> GeonAxis:
        if not isinstance(self._object, HasAxes):
            raise RuntimeError(
                "Can only instantiate an axis function if the object is of a "
                "concrete type (e.g. perception or situation object)"
            )
        horizontal_axes = tuple(
            axis
            for axis in self._object.axes.all_axes
            if not axis.aligned_to_gravitational
        )
        return horizontal_axes[self._index]  # pylint:disable=invalid-sequence-index

    def copy_remapping_objects(
        self, object_map: Mapping[_ObjectT, _ObjectToT]
    ) -> "HorizontalAxisOfObject[_ObjectToT]":
        return HorizontalAxisOfObject(object_map[self._object], index=self._index)


@attrs(frozen=True)
class FacingAddresseeAxis(Generic[_ObjectT], AxisFunction[_ObjectT]):
    _object: _ObjectT = attrib()

    def to_concrete_axis(
        self, axes_info: Optional[AxesInfo[_ObjectT]]  # pylint:disable=unused-argument
    ) -> GeonAxis:
        if not axes_info:
            raise RuntimeError(
                "FacingAddresseeAxis cannot be applied if axis info not available"
            )
        if not isinstance(self._object, HasAxes):
            raise RuntimeError(
                "Can only instantiate an axis function if the object is of a "
                "concrete type (e.g. perception or situation object)"
            )
        addressee: Optional[_ObjectT] = axes_info.addressee
        if not addressee:
            raise RuntimeError("Addressee must be specified to use FacingAddresseeAxis")
        object_axes = immutableset(self._object.axes.all_axes)
        object_axes_facing_addressee = axes_info.axes_facing[addressee].intersection(
            object_axes
        )

        if object_axes_facing_addressee:
            if len(object_axes_facing_addressee) == 1:
                return only(object_axes_facing_addressee)
            else:
                raise RuntimeError("Cannot handle multiple axes facing the addressee.")
        else:
            raise RuntimeError(
                f"Could not find axis of {self._object} facing addressee {addressee}. Axis info is "
                f"{axes_info}.  Axes of object is {object_axes}"
            )

    def copy_remapping_objects(
        self, object_map: Mapping[_ObjectT, _ObjectToT]
    ) -> "FacingAddresseeAxis[_ObjectToT]":
        return FacingAddresseeAxis(object_map[self._object])


_GRAVITATIONAL_DOWN_TO_UP_AXIS = straight_up("gravitational-up")
_SOUTH_TO_NORTH_AXIS = directed("south-to-north")
_WEST_TO_EAST_AXIS = directed("west-to-east")
_LEARNER_DOWN_TO_UP_AXIS = straight_up("learner-vertical")
_LEARNER_LEFT_RIGHT_AXIS = directed("learner-left-to-right")
_LEARNER_BACK_TO_FRONT_AXIS = directed("learner-back-to-front")


@attrs(frozen=True)
class _GravitationalAxis(AxisFunction[Any]):
    def to_concrete_axis(
        self, axes_info: Optional[AxesInfo[Any]]  # pylint:disable=unused-argument
    ) -> GeonAxis:
        return _GRAVITATIONAL_DOWN_TO_UP_AXIS

    def copy_remapping_objects(
        self, object_map: Mapping[_ObjectT, _ObjectToT]  # pylint:disable=unused-argument
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
        if quantify(self.all_axes) != 3:
            raise RuntimeError(f"All objects must have three axes but got {self}")
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
        """
        Returns a deep copy of this set of axes.

        A copy is made of each contained axis.
        The correspondence of the copies to the previous axes can be tracked
        because the order of axes is maintained
        (so the first axis in the copy is a copy of the first axis in the original, etc.)
        """
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
