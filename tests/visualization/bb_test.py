import numpy
from scipy.spatial.transform import Rotation
from adam.visualization.utils import BoundingBox


def test_bounding_boxes() -> None:
    # test collision algorithm
    bb_1 = BoundingBox.from_center_point((numpy.array([0.0, 0.0, 0.0])))

    bb_2 = BoundingBox.from_center_point(numpy.array([3.0, 3.0, 3.0]))

    bb_collide = BoundingBox.from_center_point(numpy.array([0.0, 0.75, 0.0]))

    assert not bb_1.colliding(bb_2)
    print(f"\nbb1({bb_1}) colliding with bb2({bb_2}): {bb_1.colliding(bb_2)}")

    print(
        f"\nbb1({bb_1}) colliding with bb_collide({bb_collide}): {bb_1.colliding(bb_collide)}"
    )

    assert bb_1.colliding(bb_collide)

    # check that a rotated box can register a collision

    bb_3 = BoundingBox.from_center_point(numpy.array([0.0, 0.0, 0.0]))
    bb_collide_2 = BoundingBox.from_center_point_scaled_rotated(
        numpy.array([3.0, 1.0, 0.0]),
        numpy.array([3.0, 1.0, 1.0]),
        Rotation.from_euler("z", 45, degrees=True),
    )

    print(
        f"\n\n{bb_3} colliding with BB_COLLIDE_2: \n{bb_collide_2}: \n\n{bb_3.colliding(bb_collide_2)}"
    )

    print("bb_3 corners")
    for corner in bb_3.all_corners():
        print(corner)
    print("bb_collide_2_corners")
    for corner in bb_collide_2.all_corners():
        print(corner)

    assert bb_3.colliding(bb_collide_2)

    for corner in bb_collide_2.all_corners():
        print(corner)

    print("check rotated BB face normals")
    for face_normal in bb_collide_2.all_face_normals():
        print(face_normal)

    print(f"bb1 center: {bb_1.center()}")
    print(f"bb2 center: {bb_2.center()}")
    print(f"bb_collide center: {bb_collide.center()}")

    # apply the same rotation to two non-intersecting BBs,
    # check that they still do not intersect!

    bb_2_rotated = bb_2.rotate(Rotation.from_euler("z", 45, degrees=True))
    print(f"bb_2 rotated 45 degrees on the z: {bb_2_rotated}")
    for corner in bb_2_rotated.all_corners():
        print(corner)

    print(f"bb_2_rotated right face: {bb_2_rotated.right()}")

    bb_4 = BoundingBox.from_center_point(numpy.array([6.0, 3.0, 3.0]))

    assert not bb_2.colliding(bb_4)

    bb4_rotated = bb_4.rotate(Rotation.from_euler("z", 45, degrees=True))
    print("rotating bb4")
    for corner in bb4_rotated.all_corners():
        print(corner)
    print("bb4 faces")
    for face in bb4_rotated.all_face_normals():
        print(face)

    assert not bb_2_rotated.colliding(bb4_rotated)

    # tests related to checking minimum separation of BBs or penetration distance of BBs

    diff = bb_1.minkowski_diff_distance(bb_2)
    print(f"minkowski diff distance between bb1 and bb2: {diff}")
    # should be 1
    assert diff < 0

    print(f"bb_1 ones corner: {bb_1.one_corner()}")
    print(f"bb_1 up corner: {bb_1.up_corner()}")
    print(f"bb_1 right corner: {bb_1.right_corner()}")
    print(f"bb_1 forward corner: {bb_1.forward_corner()}")
    diff = bb_1.minkowski_diff_distance(bb_1)
    assert diff == 0

    diff = bb_3.minkowski_diff_distance(bb_collide_2)
    print(f"minkowski diff distance between bb_3 and bb_collide_2: {diff}")
    assert diff > 0

    diff = bb_1.minkowski_diff_distance(bb_collide)
    print(f"minkowski diff distance between bb_1 and bb_collide: {diff}")
    assert diff > 0

    diff = bb_2_rotated.minkowski_diff_distance(bb4_rotated)
    print(f"minkowski diff distance between bb_2_rotated and bb_4_rotated: {diff}")
    assert diff < 0
