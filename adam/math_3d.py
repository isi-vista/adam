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
    def z(self) -> float:  # pylint:disable=invalid-name
        """Provided for ease of accessing the Depth value."""
        return self.depth
