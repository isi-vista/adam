import numpy as np
import pytest

from adam.visualization.positioning import (
    AxisAlignedBoundingBox,
    CollisionPenalty,
    DistanceFromOriginPenalty,
)

# re-use in-module test:
from adam.visualization.positioning import main as optimization_test


def test_optimization() -> None:
    # Check that the position optimization system runs without crashing
    np.random.seed(127)
    optimization_test()


def test_collision_constraint() -> None:
    # positioned to be colliding with one another:
    aabb0 = AxisAlignedBoundingBox.create_at_center_point(center=np.array([0, 0, 0]))
    aabb1 = AxisAlignedBoundingBox.create_at_center_point(
        center=np.array([0.5, 0.5, 0.5])
    )

    # positioned to not be colliding
    aabb_far = AxisAlignedBoundingBox.create_at_center_point(center=np.array([50, 0, 0]))

    collision_penalty = CollisionPenalty()

    res = collision_penalty.forward(aabb0, aabb1)
    assert res > 0

    res = collision_penalty.forward(aabb0, aabb_far)

    assert res <= 0


def test_distance_constraint() -> None:
    aabb0 = AxisAlignedBoundingBox.create_at_center_point(center=np.array([0, 0, 0]))
    aabb_far = AxisAlignedBoundingBox.create_at_center_point(center=np.array([50, 0, 0]))

    # this penalty is the Euclidean distance of the box from the origin
    distance_from_origin = DistanceFromOriginPenalty()

    aabb_far_dist = distance_from_origin(aabb_far)
    aabb0_dist = distance_from_origin(aabb0)

    # aabb_far should be further from the origin than aabb0
    assert aabb_far_dist > aabb0_dist

    # check the distances of the boxes
    assert aabb_far_dist.item() == pytest.approx(50.0)
    assert aabb0_dist.item() == pytest.approx(0.0)
