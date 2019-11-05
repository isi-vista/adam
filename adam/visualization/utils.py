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
    right_forward_top: ndarray = attrib()

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
            (center + numpy.array([-1.0, -1.0, -1.0]) * scale_factor),
            (center + numpy.array([1.0, 1.0, 1.0]) * scale_factor),
            rot,
        )

    def normal(self, face):
        pass

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
                        self.right_forward_top.item(0),
                        self.left_back_bottom.item(1),
                        self.left_back_bottom.item(2),
                    ]
                )
            ),
            self.rotation.apply(
                numpy.array(
                    [
                        self.left_back_bottom.item(0),
                        self.right_forward_top.item(1),
                        self.left_back_bottom.item(2),
                    ]
                )
            ),
            self.rotation.apply(
                numpy.array(
                    [
                        self.right_forward_top.item(0),
                        self.right_forward_top.item(1),
                        self.left_back_bottom.item(2),
                    ]
                )
            ),
            self.rotation.apply(
                numpy.array(
                    [
                        self.left_back_bottom.item(0),
                        self.left_back_bottom.item(1),
                        self.right_forward_top.item(2),
                    ]
                )
            ),
            self.rotation.apply(
                numpy.array(
                    [
                        self.right_forward_top.item(0),
                        self.left_back_bottom.item(1),
                        self.right_forward_top.item(2),
                    ]
                )
            ),
            self.rotation.apply(
                numpy.array(
                    [
                        self.left_back_bottom.item(0),
                        self.right_forward_top.item(1),
                        self.right_forward_top.item(2),
                    ]
                )
            ),
            self.rotation.apply(
                numpy.array(
                    [
                        self.right_forward_top.item(0),
                        self.right_forward_top.item(1),
                        self.right_forward_top.item(2),
                    ]
                )
            ),
        ]

    def right_corner(self):
        return self.rotation.apply(
            numpy.array(
                [
                    self.right_forward_top.item(0),
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
                    self.right_forward_top.item(1),
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
                    self.right_forward_top.item(2),
                ]
            )
        )

    def right(self):
        diff = self.right_corner() - self.left_back_bottom
        norm = numpy.linalg.norm(diff)
        return diff / norm

    def forward(self):
        diff = self.forward_corner() - self.left_back_bottom
        norm = numpy.linalg.norm(diff)
        return diff / norm

    def up(self):
        diff = self.up_corner() - self.left_back_bottom
        norm = numpy.linalg.norm(diff)
        return diff / norm

    def all_face_normals(self) -> List[ndarray]:
        """In order: Right, Forward, Up.
           in the event that the BB is oriented to world axes, these would be:
           Right(1, 0, 0), Forward(0, 1, 0), Up(0, 0, 1). """
        return [self.right(), self.forward(), self.up()]

    def min_max_projection(self, axis: ndarray) -> Tuple[float, float]:
        minimum = float("inf")
        maximum = float("-inf")
        for c in self.all_corners():
            prod = numpy.dot(c, axis)
            if prod < minimum:
                minimum = prod
            elif prod > maximum:
                maximum = prod
        # print(f"\nProjected {self} onto {axis}\nMin:{minimum}, Max:{maximum}")
        return minimum, maximum

    def colliding(self, other: "BoundingBox") -> bool:
        face_norms = self.all_face_normals()
        for face_norm in face_norms:
            self_min, self_max = self.min_max_projection(face_norm)
            other_min, other_max = other.min_max_projection(face_norm)
            if other_max < self_min or self_max < other_min:
                return False
        return True
