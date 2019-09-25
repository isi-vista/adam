from itertools import chain

from attr import attrib, attrs
from attr.validators import instance_of
from immutablecollections import ImmutableSet, immutableset
from immutablecollections.converter_utils import _to_immutableset
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


def symmetric_verical(debug_name: str) -> GeonAxis:
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
