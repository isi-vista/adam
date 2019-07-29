"""
Contains math utilities for working in three-dimensional space.

We are not going to work with very large 3D models, so this is no optimized for speed.
"""
from attr import attrs, attrib
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
