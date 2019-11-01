import enum
from attr import attrs, attrib
from attr.validators import instance_of
from math import sqrt

import numpy

from typing import List, Tuple


class Shape(enum.Enum):
    CIRCULAR = "CIRCULAR"
    SQUARE = "SQUARE"
    OVALISH = "OVALISH"
    RECTANGULAR = "RECTANGULAR"
    IRREGULAR = "IRREGULAR"


# Will need to be converted to numpy arrays, at which point the necessary
# math operations will just come from there


@attrs(slots=True, frozen=True)
class Vector3:
    """(x, y, z) coordinates; frozen; z_val is up"""

    x_val: float = attrib(converter=float, validator=instance_of(float))
    y_val: float = attrib(converter=float, validator=instance_of(float))
    z_val: float = attrib(converter=float, validator=instance_of(float))

    def dist(self, other: "Vector3") -> float:
        """Distance between this Vector3 and another"""
        return sqrt(
            (other.x_val - self.x_val) ** 2
            + (other.y_val - self.y_val) ** 2
            + (other.z_val - self.z_val) ** 2
        )

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(
            self.x_val + other.x_val, self.y_val + other.y_val, self.z_val + other.z_val
        )

    def __neg__(self):
        return Vector3(-self.x_val, -self.y_val, -self.z_val)

    def __sub__(self, other):
        return self + -other

    @staticmethod
    def scale(a: "Vector3", b: "Vector3") -> "Vector3":
        return Vector3(a.x_val * b.x_val, a.y_val * b.y_val, a.z_val * b.z_val)

    @staticmethod
    def dot(a: "Vector3", b: "Vector3") -> float:
        return a.x_val * b.x_val + a.y_val * b.y_val + a.z_val * b.z_val

    def length(self) -> float:
        return sqrt(self.x_val ** 2 + self.y_val ** 2 + self.z_val ** 2)

    def normalized(self) -> "Vector3":
        length = self.length()
        return Vector3(self.x_val / length, self.y_val / length, self.z_val / length)

    def to_array(self) -> numpy.ndarray:
        return numpy.array([self.x_val, self.y_val, self.z_val])


@attrs(frozen=True, slots=True)
class BoundingBox:
    """
    Bounding Box.
    Corner names are assuming an orientation orthogonal to x_val/y_val/z_val axes, with z_val as up (i.e. -gravity):
    Let's assume a unit cube with a corner on (0,0,0), where each line segment between vertices has length 1.

    0: left_back_bottom = (0, 0, 0)
    1: right_back_bottom = (1, 0, 0)
    2: left_forward_bottom = (0, 1, 0)
    3: right_forward_bottom = (1, 1, 0)

    4: left_back_top = (0, 0, 1)
    5: right_back_top = (1, 0, 1)
    6: left_forward_top = (0, 1, 1)
    7: right_forward_top = (1, 1, 1)

    A particular BoundingBox may have different overall dimensions, (i.e. longer in the canonical x_val dimension)
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
            center + Vector3.scale(Vector3(1, 1, 1), scale_factor),
        )

    def normal(self, face):
        pass

    def all_corners(self) -> List[Vector3]:
        """return List of all corners of this BB"""
        return [
            Vector3(
                self.left_back_bottom.x_val,
                self.left_back_bottom.y_val,
                self.left_back_bottom.z_val,
            ),
            Vector3(
                self.right_forward_top.x_val,
                self.left_back_bottom.y_val,
                self.left_back_bottom.z_val,
            ),
            Vector3(
                self.left_back_bottom.x_val,
                self.right_forward_top.y_val,
                self.left_back_bottom.z_val,
            ),
            Vector3(
                self.right_forward_top.x_val,
                self.right_forward_top.y_val,
                self.left_back_bottom.z_val,
            ),
            Vector3(
                self.left_back_bottom.x_val,
                self.left_back_bottom.y_val,
                self.right_forward_top.z_val,
            ),
            Vector3(
                self.right_forward_top.x_val,
                self.left_back_bottom.y_val,
                self.right_forward_top.z_val,
            ),
            Vector3(
                self.left_back_bottom.x_val,
                self.right_forward_top.y_val,
                self.right_forward_top.z_val,
            ),
            Vector3(
                self.right_forward_top.x_val,
                self.right_forward_top.y_val,
                self.right_forward_top.z_val,
            ),
        ]

    def right_corner(self):
        return Vector3(
            self.right_forward_top.x_val,
            self.left_back_bottom.y_val,
            self.left_back_bottom.z_val,
        )

    def forward_corner(self):
        return Vector3(
            self.left_back_bottom.x_val,
            self.right_forward_top.y_val,
            self.left_back_bottom.z_val,
        )

    def up_corner(self):
        return Vector3(
            self.left_back_bottom.x_val,
            self.left_back_bottom.y_val,
            self.right_forward_top.z_val,
        )

    def all_face_normals(self) -> List[Vector3]:
        """In order: Right, Forward, Up.
           in the event that the BB is oriented to world axes, these would be:
           Right(1, 0, 0), Forward(0, 1, 0), Up(0, 0, 1). """
        return [
            (self.right_corner() - self.left_back_bottom).normalized(),
            (self.forward_corner() - self.left_back_bottom).normalized(),
            (self.up_corner() - self.left_back_bottom).normalized(),
        ]

    def min_max_projection(self, axis: Vector3) -> Tuple[float, float]:
        minimum = float("inf")
        maximum = float("-inf")
        for corner in self.all_corners():
            prod = Vector3.dot(corner, axis)
            if prod < minimum:
                minimum = prod
            elif prod > maximum:
                maximum = prod
        print(f"\nProjected {self} onto {axis}\nMin:{minimum}, Max:{maximum}")
        return minimum, maximum

    def colliding(self, other: "BoundingBox") -> bool:
        face_norms = self.all_face_normals() + other.all_face_normals()
        for face_norm in face_norms:
            self_min, self_max = self.min_max_projection(face_norm)
            other_min, other_max = other.min_max_projection(face_norm)
            if overlap((self_min, self_max), (other_min, other_max)):
                return True
        return False


def overlap(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    res = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    if res == 0.0:
        return False
    return True


if __name__ == "__main__":
    # test collision algorithm
    BB_1 = BoundingBox.from_center_point(Vector3(0, 0, 0))

    BB_2 = BoundingBox.from_center_point(Vector3(2, 2, 2))

    BB_COLLIDE = BoundingBox.from_center_point(Vector3(0, 0.5, 0))

    print(f"\nbb1({BB_1}) colliding with bb2({BB_2}): {BB_1.colliding(BB_2)}")

    print(
        f"\nbb1({BB_1}) colliding with bb_collide({BB_COLLIDE}): {BB_1.colliding(BB_COLLIDE)}"
    )
