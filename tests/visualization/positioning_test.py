import numpy as np

from adam.visualization.positioning import (
    AxisAlignedBoundingBox,
    CollisionPenalty,
    WeakGravityPenalty,
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


def test_gravity_constraint() -> None:
    aabb_floating = AxisAlignedBoundingBox.create_at_center_point(
        center=np.array([0, 0, 5])
    )

    # boxes are 2 units tall by default, so this one is resting on the ground
    aabb_grounded = AxisAlignedBoundingBox.create_at_center_point(
        center=np.array([0, 0, 1])
    )

    gravity_penalty = WeakGravityPenalty()

    floating_result = gravity_penalty(aabb_floating)
    assert floating_result > 0

    grounded_result = gravity_penalty(aabb_grounded)
    assert grounded_result <= 0
