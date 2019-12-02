import numpy as np
import torch

from adam.visualization.positioning import (
    AxisAlignedBoundingBox,
    main,
    CollisionPenalty,
    DistanceFromOriginPenalty,
)


def test_positioning() -> None:
    np.random.seed(127)
    main()

    # positioned to be colliding with one another:
    aabb0 = AxisAlignedBoundingBox.create_at_center_point(
        center=np.ndarray(shape=(3,), buffer=np.array([0, 0, 0]))
    )
    aabb1 = AxisAlignedBoundingBox.create_at_center_point(
        center=np.ndarray(shape=(3,), buffer=np.array([0.5, 0.5, 0.5]))
    )

    # set to definitely not be near the origin:
    aabb_rand = AxisAlignedBoundingBox.create_at_random_position(
        min_distance_from_origin=6.0, max_distance_from_origin=10.0
    )
    aabb_scaled = AxisAlignedBoundingBox.create_at_random_position_scaled(
        min_distance_from_origin=6.0,
        max_distance_from_origin=10.0,
        object_scale=torch.tensor([2, 1, 1]),
    )

    collision_penalty = CollisionPenalty()

    res = collision_penalty.forward(aabb0, aabb1)
    assert res > 0

    res = collision_penalty.forward(aabb0, aabb_rand)
    assert res <= 0

    distance_from_origin = DistanceFromOriginPenalty()

    assert distance_from_origin.forward(aabb_scaled) > distance_from_origin.forward(aabb0)
