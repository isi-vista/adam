"""
Contains math utilities for working in three-dimensional space.

We are not going to work with very large 3D models, so this is no optimized for speed.
"""
from attr import attrib, attrs
from attr.validators import instance_of


@attrs(frozen=True)
class Point:
    """
    A point in 3D space.
    """

    x: float = attrib(  # pylint:disable=invalid-name
        validator=instance_of(float), converter=float
    )
    y: float = attrib(  # pylint:disable=invalid-name
        validator=instance_of(float), converter=float
    )
    z: float = attrib(  # pylint:disable=invalid-name
        validator=instance_of(float), converter=float
    )


@attrs(frozen=True)
class DepthPoint:
    """
    A point in 3D space represented by a depth value for the z direction.

    Points are view agnostic by implementation but are assumed to exist
    from the space view-point when compared. Our system may use multiple views
    at different points which are not 1:1 comparable.
    """

    x_coord: float = attrib(validator=instance_of(float), converter=float)
    y_coord: float = attrib(validator=instance_of(float), converter=float)
    depth: float = attrib(validator=instance_of(float), converter=float)

    @property
    def x(self) -> float:  # pylint:disable=invalid-name
        return self.x_coord

    @property
    def y(self) -> float:  # pylint:disable=invalid-name
        return self.y_coord

    @property
    def d(self) -> float:  # pylint:disable=invalid-name
        return self.depth
