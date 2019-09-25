from itertools import chain
from typing import Iterable, Mapping

from attr import attrs, attrib
from attr.validators import instance_of
from immutablecollections import ImmutableSet, immutableset
from immutablecollections.converter_utils import _to_immutableset
from more_itertools import quantify
from typing_extensions import runtime, Protocol

from adam.axes import _GRAVITATIONAL_DOWN_TO_UP_AXIS, _SOUTH_TO_NORTH_AXIS, \
    _WEST_TO_EAST_AXIS, _LEARNER_DOWN_TO_UP_AXIS, _LEARNER_LEFT_RIGHT_AXIS, \
    _LEARNER_BACK_TO_FRONT_AXIS
from adam.axis import GeonAxis
from adam.relation import Relation, flatten_relations


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