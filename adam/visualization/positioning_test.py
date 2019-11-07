import cvxpy
import numpy as np
from cvxpy import minimum, maximum
from adam.visualization.utils import BoundingBox, min_max_projection


def main() -> None:

    center = cvxpy.Variable(3, "center")  # x, y z

    relative_bb = BoundingBox.from_center_point(np.array([0.0, 0.0, 0.0]))
    relative_bb_faces = relative_bb.all_face_normals()
    relative_bb_corners = relative_bb.all_corners()
    relative_bb_min_max_projections = [
        min_max_projection(relative_bb_corners, face) for face in relative_bb_faces
    ]

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

    corner_shaped = np.broadcast_to(center, (8, 3))
    print(corner_shaped.shape)
    print(type(corner_shaped))

    resolved_corners = corner_shaped + corner_matrix
    print(resolved_corners.shape)

    print(
        f"shape of a single resolved corner: {resolved_corners[0].shape}, type: {type(resolved_corners[0])}"
    )

    print(
        f"shape of relative bb face normal: {relative_bb_faces[0].shape}, type: {type(relative_bb_faces[0])}"
    )

    test_dot = np.dot(resolved_corners[0], resolved_corners[1])
    # print(dir(test_dot))
    print(
        f"test_dot corner . corner shape: {test_dot.shape}, type: {type(test_dot)}, value: {test_dot}"
    )

    test_dot = np.dot(relative_bb_faces[0], relative_bb_faces[1])
    print(type(test_dot))
    print(
        f"test_dot face . face shape: {test_dot.shape}, type: {type(test_dot)}, value: {test_dot}"
    )

    test_dot = np.dot(resolved_corners[3], relative_bb_faces[1])
    print(type(test_dot))
    print(
        f"test dot for corner . axis: {test_dot.shape}, type: {type(test_dot)}, value: {test_dot}"
    )

    projections = [
        np.dot(resolved_corners[j], relative_bb_faces[i])
        for i in range(3)
        for j in range(8)
    ]
    print(f"shape of projection: {projections[0].shape}")
    print(f"projection curvature: {projections[0].curvature}")

    print(f"{projections[0].shape}, type {type(projections[0])}")
    print(
        f"{relative_bb_min_max_projections[0]}, type: {type(relative_bb_min_max_projections[0])}"
    )

    print(f"type of projections[0][0]: {type(projections[0][0])}")
    print(
        f"type of relative_bb_min_max_projections[0]: {type(relative_bb_min_max_projections[0])}"
    )

    # This is very wrong because we want to disincentivize collisions
    collide_constraint = (
        minimum(projections[0][0], relative_bb_min_max_projections[0][1])
        - maximum(projections[1][0], relative_bb_min_max_projections[0][0])
        >= 0
    )

    print(
        f"minimum(projections[0][0], relative_bb_min_max[0][1] curvature: {minimum(projections[0][0], relative_bb_min_max_projections[0][1]).curvature}"
    )
    print(
        f"maximum(projections[1][0], relative_bb_min_max_projections[0][0]) curvature: {maximum(projections[1][0], relative_bb_min_max_projections[0][0]).curvature}"
    )
    print(
        f"curvature from the difference: {(minimum(projections[0][0], relative_bb_min_max_projections[0][1]) - maximum(projections[1][0], relative_bb_min_max_projections[0][0])).curvature}"
    )

    constraints = [collide_constraint]

    obj = cvxpy.Minimize(
        cvxpy.abs(center[0]) + cvxpy.abs(center[1]) + cvxpy.abs(center[2])
    )
    prob = cvxpy.Problem(obj, constraints)
    print(f"is problem DCP: {prob.is_dcp()}")
    print(f"solvin it: {prob.solve()}")
    print(f"status: {prob.status}")
    print(f"prob value: {prob.value}")
    print(f"vars: {center.value}")


if __name__ == "__main__":
    main()
