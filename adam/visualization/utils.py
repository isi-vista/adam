import enum
import attr
from attr import attrs, attrib
from attr.validators import instance_of
from math import sqrt

class Shape(enum.Enum):
    CIRCULAR = "CIRCULAR"
    SQUARE = "SQUARE"
    OVALISH = "OVALISH"
    RECTANGULAR = "RECTANGULAR"
    IRREGULAR = "IRREGULAR"


@attrs(frozen=True, slots=True)
class Vector3:
    """(x, y, z) coordinates; frozen; Z is up"""
    x: float = attrib(converter=float, validator=instance_of(float))
    y: float = attrib(converter=float, validator=instance_of(float))
    z: float = attrib(converter=float, validator=instance_of(float))

    def dist(self, other: "Vector3") -> float:
        """Distance between this Vector3 and another"""
        return sqrt((other.x - self.x) ** 2 + (other.y - self.y) ** 2 + (other.z - self.z) ** 2)

    @staticmethod
    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    @staticmethod
    def scale(a: "Vector3", b: "Vector3") -> "Vector3":
        return Vector3(a.x * b.x, a.y * b.y, a.z * b.z)



@attrs(frozen=True, slots=True)
class BoundingBox:
    """
    Bounding Box.
    Corner names are assuming an orientation orthogonal to X/Y/Z axes, with Z as up (i.e. -gravity):
    Let's assume a unit cube with a corner on (0,0,0), where each line segment between vertices has length 1.

    left_back_bottom = (0, 0, 0)
    right_back_bottom = (1, 0, 0)
    left_forward_bottom = (0, 1, 0)
    right_forward_bottom = (1, 1, 0)

    left_back_top = (0, 0, 1)
    right_back_top = (1, 0, 1)
    left_forward_top = (0, 1, 1)
    right_forward_top = (1, 1, 1)

    A particular BoundingBox may have different overall dimensions, (i.e. longer in the canonical X dimension)
    but its corners will still correspond to these names.
    """
    left_back_bottom: Vector3 = attrib()
    right_forward_top: Vector3 = attrib()

    @classmethod
    def from_center_point(cls, center: Vector3) -> "BoundingBox":
        return BoundingBox.from_center_point_scaled(center, Vector3(1, 1, 1))

    @classmethod
    def from_center_point_scaled(cls, center: Vector3, scale_factor: Vector3):
        return cls(
            center + Vector3.scale(Vector3(0, 0, 0), scale_factor),
            center + Vector3.scale(Vector3(1, 1, 1), scale_factor)
        )
