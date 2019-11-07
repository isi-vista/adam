import enum
from attr import attrs, attrib

import numpy
from numpy import ndarray
from scipy.spatial.transform import Rotation

from typing import List, Tuple


class Shape(enum.Enum):
    CIRCULAR = "CIRCULAR"
    SQUARE = "SQUARE"
    OVALISH = "OVALISH"
    RECTANGULAR = "RECTANGULAR"
    IRREGULAR = "IRREGULAR"


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

    left_back_bottom: ndarray = attrib()

    scale: ndarray = attrib()

    rotation: Rotation = attrib()

    @classmethod
    def from_center_point(cls, center: ndarray) -> "BoundingBox":
        return BoundingBox.from_center_point_scaled(center, numpy.array([1.0, 1.0, 1.0]))

    @classmethod
    def from_center_point_scaled(cls, center: ndarray, scale_factor: ndarray):
        return BoundingBox.from_center_point_scaled_rotated(
            center, scale_factor, Rotation.from_quat([0, 0, 0, 1])
        )

    @classmethod
    def from_center_point_scaled_rotated(
        cls, center: ndarray, scale_factor: ndarray, rot: Rotation
    ):
        return cls(
            (center + numpy.array([-1.0, -1.0, -1.0]) * scale_factor), scale_factor, rot
        )

    @classmethod
    def from_scaled_corners(cls, left_back_bottom_corner, right_forward_top_corner, rot):
        return cls(left_back_bottom_corner, right_forward_top_corner, rot)

    def translate(self, offset: ndarray) -> "BoundingBox":
        "Return a new BB translated by offset"
        return BoundingBox.from_scaled_corners(
            self.left_back_bottom + offset, self.scale, self.rotation
        )

    def rotate(self, rot: Rotation):
        return BoundingBox.from_scaled_corners(
            self.left_back_bottom, self.scale, self.rotation * rot
        )

    def normal(self, face):
        pass

    def center(self) -> ndarray:
        """return center point of the BB"""
        return self.scale + self.left_back_bottom

    def all_corners(self) -> List[ndarray]:
        """return List of all corners of this BB"""
        # TODO: just return a 2d array of the corners
        return [
            self.rotation.apply(
                numpy.array(
                    [
                        self.left_back_bottom.item(0),
                        self.left_back_bottom.item(1),
                        self.left_back_bottom.item(2),
                    ]
                )
            ),
            self.rotation.apply(
                numpy.array(
                    [
                        self.left_back_bottom.item(0) + self.scale.item(0),
                        self.left_back_bottom.item(1),
                        self.left_back_bottom.item(2),
                    ]
                )
            ),
            self.rotation.apply(
                numpy.array(
                    [
                        self.left_back_bottom.item(0),
                        self.left_back_bottom.item(1) + self.scale.item(1),
                        self.left_back_bottom.item(2),
                    ]
                )
            ),
            self.rotation.apply(
                numpy.array(
                    [
                        self.left_back_bottom.item(0) + self.scale.item(0),
                        self.left_back_bottom.item(1) + self.scale.item(1),
                        self.left_back_bottom.item(2),
                    ]
                )
            ),
            self.rotation.apply(
                numpy.array(
                    [
                        self.left_back_bottom.item(0),
                        self.left_back_bottom.item(1),
                        self.left_back_bottom.item(2) + self.scale.item(2),
                    ]
                )
            ),
            self.rotation.apply(
                numpy.array(
                    [
                        self.left_back_bottom.item(0) + self.scale.item(0),
                        self.left_back_bottom.item(1),
                        self.left_back_bottom.item(2) + self.scale.item(2),
                    ]
                )
            ),
            self.rotation.apply(
                numpy.array(
                    [
                        self.left_back_bottom.item(0),
                        self.left_back_bottom.item(1) + self.scale.item(1),
                        self.left_back_bottom.item(2) + self.scale.item(2),
                    ]
                )
            ),
            self.rotation.apply(
                numpy.array(
                    [
                        self.left_back_bottom.item(0) + self.scale.item(0),
                        self.left_back_bottom.item(1) + self.scale.item(1),
                        self.left_back_bottom.item(2) + self.scale.item(2),
                    ]
                )
            ),
        ]

    def right_corner(self):
        return self.rotation.apply(
            numpy.array(
                [
                    self.left_back_bottom.item(0) + self.scale.item(0),
                    self.left_back_bottom.item(1),
                    self.left_back_bottom.item(2),
                ]
            )
        )

    def forward_corner(self):
        return self.rotation.apply(
            numpy.array(
                [
                    self.left_back_bottom.item(0),
                    self.left_back_bottom.item(1) + self.scale.item(1),
                    self.left_back_bottom.item(2),
                ]
            )
        )

    def up_corner(self):
        return self.rotation.apply(
            numpy.array(
                [
                    self.left_back_bottom.item(0),
                    self.left_back_bottom.item(1),
                    self.left_back_bottom.item(2) + self.scale.item(2),
                ]
            )
        )

    def zero_corner(self):
        return self.rotation.apply(self.left_back_bottom)

    def one_corner(self):
        return self.rotation.apply(self.left_back_bottom + self.scale * 2)

    def right(self):
        diff = self.right_corner() - self.zero_corner()
        norm = numpy.linalg.norm(diff)
        return diff / norm

    def forward(self):
        diff = self.forward_corner() - self.zero_corner()
        norm = numpy.linalg.norm(diff)
        return diff / norm

    def up(self):
        diff = self.up_corner() - self.zero_corner()
        norm = numpy.linalg.norm(diff)
        return diff / norm

    def all_face_normals(self) -> List[ndarray]:
        """In order: Right, Forward, Up.
           in the event that the BB is oriented to world axes, these would be:
           Right(1, 0, 0), Forward(0, 1, 0), Up(0, 0, 1). """
        return [self.right(), self.forward(), self.up()]

    def min_max_projection(self, axis: ndarray) -> Tuple[float, float]:
        return min_max_projection(self.all_corners(), axis)

    def colliding(self, other: "BoundingBox") -> bool:
        face_norms = self.all_face_normals()
        for face_norm in face_norms:
            self_min, self_max = self.min_max_projection(face_norm)
            other_min, other_max = other.min_max_projection(face_norm)
            if other_max < self_min or self_max < other_min:
                return False
        return True


def min_max_projection(corners: List[ndarray], axis: ndarray) -> Tuple[float, float]:
    minimum = float("inf")
    maximum = float("-inf")
    for c in corners:
        prod = numpy.dot(c, axis)
        if prod < minimum:
            minimum = prod
        elif prod > maximum:
            maximum = prod
    # print(f"\nProjected {self} onto {axis}\nMin:{minimum}, Max:{maximum}")
    return minimum, maximum
