import numpy as np
from numpy import ndarray

from typing import List

from adam.visualization.utils import BoundingBox, min_max_projection


def main() -> None:
    bb = BoundingBox.from_center_point(np.array([0, 0, 0]))
    midpoint_minimization(bb)

    assert False

    # standard form:
    # minimize: x    (1/2) x^(T) Px + q^(T)x

    # subject to: Gx element-wise< h
    #  " " "    : Ax = b

    # objective function:

    # constraints:


class PositionSolver:
    POSITION_CONSTRAINTS = {
        "X_RANGE": (-10.0, 10.0),
        "Y_RANGE": (-10.0, 5.0),
        "Z_RANGE_GROUND": (0.0, 0.0),
        "Z_RANGE_FLY": (2.0, 5.0),
        "Z_RANGE_OTHER": (0.0, 3.0),
    }

    CLOSE_DISTANCE = 0.25

    def __init__(self, main_point: ndarray, points: List[ndarray]) -> None:
        self.main_position = main_point
        self.positions = points


class RelativePositionSolver:
    pass


def midpoint_minimization(relative_bb: BoundingBox):
    # objective function:

    # quadratic components: none
    # P = matrix(np.diag(np.array([0, 0, 0])), tc="d")
    # linear coefficients: all ones
    # q = matrix(np.array([1, 1, 1]), tc="d")

    relative_bb_faces = relative_bb.all_face_normals()

    relative_bb_corners = relative_bb.all_corners()

    relative_bb_min_max_projections = [
        min_max_projection(relative_bb_corners, face) for face in relative_bb_faces
    ]

    print(relative_bb_min_max_projections)

    corner_matrix = np.array(
        [
            (-1, -1, -1),
            (1, -1, -1),
            (-1, 1, -1),
            (1, 1, -1),
            (-1, -1, 1),
            (1, -1, 1),
            (-1, 1, 1),
            (1, 1, 1),
        ]
    )

    print("np test situation")
    print(corner_matrix + np.array([1.0, 1.0, 1.0]))

    # should have a variable representing center of box
    box_a_center = np.array((3,))

    # idea of an objective function: minimizing distance from origin
    # keep in mind though... for abs() to be differentiable it has to be broken up into parts
    # f = abs(box_a_center[0]) + abs(box_a_center[1]) + abs(box_a_center[2])

    # constraints:
    # c_x_1 = box_a_center[0] <= 10.0
    #
    # c_x_2 = box_a_center[0] >= -10.0
    # c_y_1 = box_a_center[1] <= 5.0
    # c_y_2 = box_a_center[1] >= -5.0
    # c_z_1 = box_a_center[2] <= 2.0
    # c_z_2 = box_a_center[2] >= 0.0

    # intersection constraints:
    print(corner_matrix.size)

    center_shaped = np.ones((8, 3)) * box_a_center

    print(f"center_shaped type: {type(center_shaped)}")

    # corner0 = corner_shaped + corner_matrix[0]
    corners = center_shaped + corner_matrix
    print(f"type of corners: {type(corners)}")
    print(f"type of a corners element: {type(corners[0])}")

    print(corners[0].size)
    print(relative_bb_faces[0].size)
    print(f"corner shape: {corners[0].shape}")
    print(f"relative_bb_face shape: {relative_bb_faces[0].shape}")
    projections = [
        np.dot(corners[j], relative_bb_faces[i]) for i in range(3) for j in range(8)
    ]

    print(len(projections))
    print(f"{projections[0].shape}")
    print(f"{projections[0].size}")

    # collide_constraint = (
    #     cvxopt.modeling.min(projections[0], relative_bb_min_max_projections[0][1])
    #     - cvxopt.modeling.max(projections[1], relative_bb_min_max_projections[0][0])
    #     <= 0
    # )

    # prob = op(
    #     f, [c_x_1, c_x_2, c_y_1, c_y_2, c_z_1, c_z_2, collide_constraint], "test_prob"
    # )
    # prob.solve()
    # print(prob.status)

    # G =


if __name__ == "__main__":
    main()
