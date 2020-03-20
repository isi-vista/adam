import numpy as np
import torch

from math import isclose, pi
from immutablecollections import immutableset

from adam.visualization.positioning import (
    AxisAlignedBoundingBox,
    CollisionPenalty,
    WeakGravityPenalty,
    run_model,
    angle_between,
    InRegionPenalty,
)
from typing import Mapping, List, Tuple
from adam.perception import ObjectPerception, GROUND_PERCEPTION
from adam.axes import (
    Axes,
    straight_up,
    # directed,
    symmetric,
    symmetric_vertical,
)
from adam.ontology.phase1_spatial_relations import (
    Region,
    PROXIMAL,
    Direction,
    GRAVITATIONAL_UP,
    EXTERIOR_BUT_IN_CONTACT,
)
from adam.axes import HorizontalAxisOfObject


def test_running_model() -> None:
    # for code coverage purposes
    ball = ObjectPerception(
        "ball",
        axes=Axes(
            primary_axis=symmetric_vertical("ball-generating"),
            orienting_axes=immutableset(
                [symmetric("side-to-side0"), symmetric("side-to-side1")]
            ),
        ),
    )
    box = ObjectPerception(
        "box",
        axes=Axes(
            primary_axis=straight_up("top_to_bottom"),
            orienting_axes=immutableset(
                [symmetric("side-to-side0"), symmetric("side-to-side1")]
            ),
        ),
    )

    objs = immutableset([ball, box])
    relations: Mapping[ObjectPerception, List[Region[ObjectPerception]]] = {}
    scales: Mapping[str, Tuple[float, float, float]] = {
        "box": (1.0, 1.0, 1.0),
        "ball": (2.0, 1.0, 1.0),
    }
    run_model(
        objs,
        {},
        relations,
        scales,
        num_iterations=10,
        yield_steps=10,
        frozen_objects=immutableset([]),
    )


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

    ground_region = Region(GROUND_PERCEPTION, EXTERIOR_BUT_IN_CONTACT, GRAVITATIONAL_UP)

    floating_perception = ObjectPerception(
        "floating_thing",
        axes=Axes(
            primary_axis=symmetric_vertical("floating-thing-generating"),
            orienting_axes=immutableset(
                [symmetric("side-to-side0"), symmetric("side-to-side1")]
            ),
        ),
    )

    grounded_perception = ObjectPerception(
        "grounded_thing",
        axes=Axes(
            primary_axis=symmetric_vertical("grounded-thing-generating"),
            orienting_axes=immutableset(
                [symmetric("side-to-side0"), symmetric("side-to-side1")]
            ),
        ),
    )

    gravity_penalty = WeakGravityPenalty(
        {floating_perception: aabb_floating, grounded_perception: aabb_grounded},
        {floating_perception: [ground_region], grounded_perception: [ground_region]},
    )

    floating_result = gravity_penalty(aabb_floating, immutableset([ground_region]))
    assert floating_result > 0

    grounded_result = gravity_penalty(aabb_grounded, immutableset([ground_region]))
    assert grounded_result <= 0


def test_angle_between() -> None:
    # test that using a zero vector in angle calculation returns None
    assert (
        angle_between(
            torch.tensor([0, 0, 0]),  # pylint: disable=not-callable
            torch.tensor([1, 1, 1]),  # pylint: disable=not-callable
        )
        is None
    )
    # check angle between perpendicular vectors
    result = angle_between(
        torch.tensor([1, 0, 0], dtype=torch.float),  # pylint: disable=not-callable
        torch.tensor([0, 0, 1], dtype=torch.float),  # pylint: disable=not-callable
    )
    assert result is not None
    assert isclose(result.item(), pi / 2, rel_tol=0.05)
    # check parallel vectors
    result = angle_between(
        torch.tensor([1, 0, 0], dtype=torch.float),  # pylint: disable=not-callable
        torch.tensor([1, 0, 0], dtype=torch.float),  # pylint: disable=not-callable
    )
    assert result is not None and isclose(result.item(), 0, rel_tol=0.05)
    # check 180 degrees away
    result = angle_between(
        torch.tensor([1, 0, 0], dtype=torch.float),  # pylint: disable=not-callable
        torch.tensor([-1, 0, 0], dtype=torch.float),  # pylint: disable=not-callable
    )
    assert result is not None and isclose(result.item(), pi, rel_tol=0.05)


def test_in_region_constraint() -> None:
    ball = ObjectPerception(
        "ball",
        axes=Axes(
            primary_axis=symmetric_vertical("ball-generating"),
            orienting_axes=immutableset(
                [symmetric("side-to-side0"), symmetric("side-to-side1")]
            ),
        ),
    )
    box = ObjectPerception(
        "box",
        axes=Axes(
            primary_axis=straight_up("top_to_bottom"),
            orienting_axes=immutableset(
                [symmetric("side-to-side0"), symmetric("side-to-side1")]
            ),
        ),
    )

    aabb_ball = AxisAlignedBoundingBox.create_at_center_point(center=np.array([0, 0, 1]))

    aabb_box = AxisAlignedBoundingBox.create_at_center_point(center=np.array([0, -2, 1]))

    # specifying that the box should be to the right of the ball

    direction = Direction(
        positive=True, relative_to_axis=HorizontalAxisOfObject(ball, index=0)
    )

    region = Region(ball, PROXIMAL, direction)

    obj_percept_to_aabb = {ball: aabb_ball, box: aabb_box}

    in_region_relations = {box: [region]}

    in_region_penalty = InRegionPenalty(obj_percept_to_aabb, {}, {}, in_region_relations)

    box_penalty = in_region_penalty(box, immutableset([region]))
    assert box_penalty > 0

    # now with a box that *is* to the right of the ball

    box2 = ObjectPerception(
        "box2",
        axes=Axes(
            primary_axis=straight_up("top_to_bottom"),
            orienting_axes=immutableset(
                [symmetric("side-to-side0"), symmetric("side-to-side1")]
            ),
        ),
    )
    aabb_box2 = AxisAlignedBoundingBox.create_at_center_point(center=np.array([2, 0, 1]))

    obj_percept_to_aabb = {ball: aabb_ball, box2: aabb_box2}

    in_region_relations = {box2: [region]}

    in_region_penalty = InRegionPenalty(obj_percept_to_aabb, {}, {}, in_region_relations)

    box_penalty = in_region_penalty(box2, immutableset([region]))

    assert box_penalty == 0
