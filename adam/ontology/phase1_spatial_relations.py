from typing import TypeVar, Optional, Generic

from attr import attrib, attrs
from attr.validators import instance_of, in_, optional
from vistautils.preconditions import check_arg


@attrs(frozen=True, slots=True)
class Distance:
    """
    A distance of the sort used by Landau and Jackendoff
    to specify spatial regions.
    """

    name: str = attrib(validator=instance_of(str))


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


@attrs(frozen=True, slots=True)
class Direction:
    r"""
    Represents the direction one object may have relative to another.

    This is used to specify `Region`\ s.
    """
    positive: bool = attrib(validator=instance_of(bool))
    """
    We need to standardize on what "positive" direction means. 
    It is clear for vertical axes but less clear for other things. 
    """
    relative_to_axis: str = attrib(validator=instance_of(str))
    """
    We store an arbitrary string for this pending determining the
    proper representation.

    https://github.com/isi-vista/adam/issues/137
    """


ReferenceObjectT = TypeVar("ReferenceObjectT")


@attrs(frozen=True)
class Region(Generic[ReferenceObjectT]):
    """
    A region of space perceived by the learner.

    We largely follow

    Barbara Landau and Ray Jackdendoff. "'What' and 'where' in spatial language
    and spatial cognition. Brain and Behavioral Sciences (1993) 16:2.

    who analyze spatial relations in term of a `Distance` and `Direction`
    with respect to some *reference_object*.

    At least one of *distance* and *direction* must be specified.
    """

    reference_object: ReferenceObjectT = attrib()
    distance: Optional[Distance] = attrib(
        validator=optional(in_(LANDAU_AND_JACKENDOFF_DISTANCES)), default=None
    )
    direction: Optional[Direction] = attrib(
        validator=optional(instance_of(Direction)), default=None
    )

    def __attrs_post_init__(self) -> None:
        check_arg(
            self.distance or self.direction,
            "A region must have either a " "distance or direction specified.",
        )
