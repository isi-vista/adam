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

    bb_3 = BoundingBox.from_center_point(numpy.array([0.0, 0.0, 0.0]))
    bb_collide_2 = BoundingBox.from_center_point_scaled_rotated(
        numpy.array([3.0, 1.0, 0.0]),
        numpy.array([2.0, 1.0, 1.0]),
        Rotation.from_euler("z", 45, degrees=True),
    )

    print(
        f"\n\n{bb_3} colliding with BB_COLLIDE_2: \n{bb_collide_2}: \n\n{bb_3.colliding(bb_collide_2)}"
    )

    assert bb_3.colliding(bb_collide_2)

    for corner in bb_collide_2.all_corners():
        print(corner)

    print("check rotated BB face normals")
    for face_normal in bb_collide_2.all_face_normals():
        print(face_normal)
