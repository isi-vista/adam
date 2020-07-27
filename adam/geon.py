from typing import Dict, Mapping, Optional

from typing_extensions import Protocol

from adam.axes import Axes, HasAxes
from adam.axis import GeonAxis
from adam.utilities import sign
from attr import attrib, attrs
from attr.validators import instance_of


@attrs(frozen=True, slots=True, repr=False)
class CrossSectionSize:
    name: str = attrib(validator=instance_of(str))

    def __repr__(self) -> str:
        return self.name


@attrs(frozen=True, slots=True, repr=False, cache_hash=True)
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
            f"[{sign(self.has_reflective_symmetry)}reflect-sym, "
            f"{sign(self.has_rotational_symmetry)}rotate-sym, "
            f"{sign(self.curved)}curved]"
        )


@attrs(slots=True, frozen=True, cache_hash=True)
class Geon(HasAxes):
    cross_section: CrossSection = attrib(
        validator=instance_of(CrossSection), kw_only=True
    )
    cross_section_size: CrossSectionSize = attrib(
        validator=instance_of(CrossSectionSize), kw_only=True
    )
    axes: Axes = attrib(validator=instance_of(Axes), kw_only=True)
    generating_axis: GeonAxis = attrib(validator=instance_of(GeonAxis), kw_only=True)

    def copy(
        self,
        *,
        axis_mapping: Optional[Mapping[GeonAxis, GeonAxis]] = None,
        output_axis_mapping: Optional[Dict[GeonAxis, GeonAxis]] = None,
    ) -> "Geon":
        """
        Makes an independent copy of this geon.

        This will also have its own axes.
        This geon's axes will be mapped using *axis_mapping* if specified.
        Otherwise, each axis will be copied.
        If *output_axis_mapping* is specified, it will be populated
        with the mapping between original and copied axes.
        This is somewhat of a hack,
        but the information is needed when instantiating
        perceivable sub-object relations from object schemata.
        """
        if axis_mapping is None:
            axis_mapping = {axis: axis.copy() for axis in self.axes.all_axes}
            if output_axis_mapping is not None:
                if output_axis_mapping:
                    raise RuntimeError(
                        f"output_axis_mapping must always be empty but got "
                        f"{output_axis_mapping}"
                    )
                output_axis_mapping.update(axis_mapping)
        return Geon(
            cross_section=self.cross_section,
            cross_section_size=self.cross_section_size,
            axes=self.axes.remap_axes(axis_mapping),
            generating_axis=axis_mapping[self.generating_axis],
        )

    @generating_axis.default
    def _init_primary_axis(self) -> GeonAxis:
        return self.axes.primary_axis


class MaybeHasGeon(Protocol):
    geon: Optional[Geon]


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
