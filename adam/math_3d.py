"""
Contains math utilities for working in three-dimensional space
"""
from attr import attrs, attrib
from attr.validators import instance_of


@attrs(frozen=True)
class Point:
    """
    A point in 3D space.
    """

    x: float = attrib(validator=instance_of(float))  # pylint:disable=invalid-name
    y: float = attrib(validator=instance_of(float))  # pylint:disable=invalid-name
    z: float = attrib(validator=instance_of(float))  # pylint:disable=invalid-name
