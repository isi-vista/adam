from itertools import combinations
from typing import Mapping, AbstractSet
from attr import attrs, attrib

import numpy as np
from numpy import ndarray
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from immutablecollections import immutabledict, immutableset, ImmutableDict
from vistautils.preconditions import check_arg

# see https://github.com/pytorch/pytorch/issues/24807 re: pylint issue
ORIGIN = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float)  # pylint: disable=not-callable
COLLISION_PENALTY = 10


@attrs(frozen=True, auto_attribs=True)
class AdamObject:
    name: str


def main() -> None:
    ball = AdamObject(name="ball")
    box = AdamObject(name="box")

    cardboard_box = AdamObject(name="cardboardBox")
    aardvark = AdamObject(name="aardvark")
    flamingo = AdamObject(name="flamingo")

    positioning_model = AdamObjectPositioningModel.for_objects(
        immutableset([ball, box, cardboard_box, aardvark, flamingo])
    )
    # we will start with an aggressive learning rate
    optimizer = optim.SGD(positioning_model.parameters(), lr=1.0)
    # but will decrease it whenever the loss plateaus
    learning_rate_schedule = ReduceLROnPlateau(
        optimizer,
        "min",
        # decrease the rate if the loss hasn't improved in
        # 3 epochs
        patience=3,
    )

    iterations = 25
    for _ in range(iterations):
        loss = positioning_model()
        print(f"Loss: {loss.item()}")
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        learning_rate_schedule.step(loss)

        positioning_model.dump_object_positions()


@attrs(frozen=True, slots=True)
class AxisAlignedBoundingBox:
    """
    Defines a 3D Box that is oriented to world axes given a center point (3,)
    and a scale (which defines the distance from the center to each corner of the box).
    e.g. a box centered at (0, 0, 0) and with a scale of (1, 1, 1) would have opposite
    corners at (-1, -1, -1) and (1, 1, 1), giving the box a volume of 2(^3)
    """

    center: torch.Tensor = attrib()  # tensor shape: (3,)
    scale: torch.Tensor = attrib()  # tensor shape: (3, 3) - diagonal matrix
    # rotation: torch.Tensor = attrib()

    def center_distance_from_point(self, point: torch.Tensor) -> torch.Tensor:
        return torch.dist(self.center, point, 2)

    @staticmethod
    def create_at_random_position(
        *, min_distance_from_origin: float, max_distance_from_origin: float
    ):
        return AxisAlignedBoundingBox.create_at_random_position_scaled(
            min_distance_from_origin=min_distance_from_origin,
            max_distance_from_origin=max_distance_from_origin,
            object_scale=torch.ones(3),
        )

    @staticmethod
    def create_at_center_point(*, center: ndarray):
        return AxisAlignedBoundingBox(
            Parameter(
                torch.tensor(center, dtype=torch.float),  # pylint: disable=not-callable
                requires_grad=True,
            ),
            torch.diag(torch.ones(3)),
        )

    @staticmethod
    def create_at_random_position_scaled(
        *,
        min_distance_from_origin: float,
        max_distance_from_origin: float,
        object_scale: torch.Tensor,
    ):
        check_arg(min_distance_from_origin > 0.0)
        check_arg(min_distance_from_origin < max_distance_from_origin)
        # we first generate a random point on the unit sphere by
        # generating a random vector in cube...
        center = np.random.randn(3, 1).squeeze()
        # and then normalizing.
        center /= np.linalg.norm(center)

        # then we scale according to the distances above
        scale_factor = np.random.uniform(
            min_distance_from_origin, max_distance_from_origin
        )
        center *= scale_factor
        return AxisAlignedBoundingBox(
            Parameter(
                torch.tensor(center, dtype=torch.float),  # pylint: disable=not-callable
                requires_grad=True,
            ),
            torch.diag(object_scale),
        )

    def get_corners(self) -> torch.Tensor:
        return self.center.expand(8, 3) + torch.tensor(  # pylint: disable=not-callable
            [
                [-1, -1, -1],
                [1, -1, -1],
                [-1, 1, -1],
                [1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [-1, 1, 1],
                [1, 1, 1],
            ],
            dtype=torch.float,
        ).matmul(self.scale)
        # see https://github.com/pytorch/pytorch/issues/24807 re: pylint issue

    # helper functions giving names to a few corners used in calculations:
    def right_corner(self) -> torch.Tensor:
        """
        Corner in the direction of the positive x axis from center (negative direction from other axes).
        (Assuming box is oriented to world axes)
        Returns: Tensor (3,)
        """
        return self.center + torch.tensor(  # pylint: disable=not-callable
            [1, -1, -1], dtype=torch.float
        ).matmul(self.scale)
        # see https://github.com/pytorch/pytorch/issues/24807 re: pylint issue

    def forward_corner(self) -> torch.Tensor:
        """
        Corner in the direction of the positive y axis from center (negative direction from other axes).
        (Assuming box is oriented to world axes)
        Returns: Tensor (3,)

        """
        return self.center + torch.tensor(  # pylint: disable=not-callable
            [-1, 1, -1], dtype=torch.float
        ).matmul(self.scale)

    def up_corner(self) -> torch.Tensor:
        """
        Corner in the direction of the positive z axis from center (negative direction from other axes).
        (Assuming box is oriented to world axes)
        Returns: Tensor (3,)

        """
        return self.center + torch.tensor(  # pylint: disable=not-callable
            [-1, -1, 1], dtype=torch.float
        ).matmul(self.scale)

    def minus_ones_corner(self) -> torch.Tensor:
        """
        Corner in the direction of the negative x, y, z axes from center
        Returns: Tensor (3,)

        """
        return self.center + torch.tensor(  # pylint: disable=not-callable
            [-1, -1, -1], dtype=torch.float
        ).matmul(self.scale)

    # functions returning normal vectors from three perpendicular faces of the box
    def right_face(self) -> torch.Tensor:
        """
        Normal vector for the right face of the box (toward positive x axis when aligned to world axes)
        Returns: Tensor (3,)
        """
        diff = self.right_corner() - self.minus_ones_corner()
        return diff / torch.norm(diff)

    def forward_face(self) -> torch.Tensor:
        """
        Normal vector for the forward face of the box (toward positive y axis when aligned to world axes)
        Returns: Tensor (3,)
        """
        diff = self.forward_corner() - self.minus_ones_corner()
        return diff / torch.norm(diff)

    def up_face(self) -> torch.Tensor:
        """
        Normal vector for the up face of the box (toward positive z axis when aligned to world axes)
        Returns: Tensor (3,)
        """
        diff = self.up_corner() - self.minus_ones_corner()
        return diff / torch.norm(diff)

    def get_face_norms(self) -> torch.Tensor:
        """
        Stacks the face norms from the right, forward, and up faces of the box
        Returns: Tensor (3,3)

        """
        # in axis-aligned case these are always the same
        return torch.stack([self.right_face(), self.forward_face(), self.up_face()])

    def corner_onto_axes_projections(self, axes: torch.Tensor) -> torch.Tensor:
        """
        Projects each of 8 corners onto each of three axes.
        Args:
            axes: (3,3) tensor -> the three axes we are projecting points onto

        Returns:
            (3, 8) tensor -> each point projected onto each of three dimensions

        """
        check_arg(axes.shape == (3, 3))
        corners = self.get_corners()
        return axes.matmul(corners.transpose(0, 1))


class AdamObjectPositioningModel(torch.nn.Module):
    def __init__(
        self, adam_object_to_bounding_box: Mapping[AdamObject, AxisAlignedBoundingBox]
    ) -> None:
        super().__init__()
        self.adam_object_to_bounding_box = adam_object_to_bounding_box
        self.object_bounding_boxes = adam_object_to_bounding_box.values()

        for (adam_object, bounding_box) in self.adam_object_to_bounding_box.items():
            self.register_parameter(adam_object.name, bounding_box.center)

        self.distance_to_origin_penalty = DistanceFromOriginPenalty()
        self.collision_penalty = CollisionPenalty()

    @staticmethod
    def for_objects(
        adam_objects: AbstractSet[AdamObject]
    ) -> "AdamObjectPositioningModel":
        objects_to_bounding_boxes: ImmutableDict[
            AdamObject, AxisAlignedBoundingBox
        ] = immutabledict(
            (
                adam_object,
                AxisAlignedBoundingBox.create_at_random_position(
                    min_distance_from_origin=5, max_distance_from_origin=10
                ),
            )
            for adam_object in adam_objects
        )
        return AdamObjectPositioningModel(objects_to_bounding_boxes)

    def forward(self) -> int:  # pylint: disable=arguments-differ
        distance_penalty = sum(
            self.distance_to_origin_penalty(box) for box in self.object_bounding_boxes
        )
        collision_penalty = sum(
            self.collision_penalty(box1, box2)
            for (box1, box2) in combinations(self.object_bounding_boxes, 2)
        )
        print(
            f"distance penalty: {distance_penalty}\ncollision penalty: {collision_penalty}"
        )
        return distance_penalty + collision_penalty

    def dump_object_positions(self) -> None:
        for (adam_object, bounding_box) in self.adam_object_to_bounding_box.items():
            print(f"{adam_object.name} = {bounding_box.center.data}")


class DistanceFromOriginPenalty(nn.Module):
    def __init__(self) -> None:  # pylint: disable=useless-super-delegation
        super().__init__()

    def forward(  # pylint: disable=arguments-differ
        self, bounding_box: AxisAlignedBoundingBox
    ) -> torch.Tensor:
        return bounding_box.center_distance_from_point(ORIGIN)


class CollisionPenalty(nn.Module):
    def __init__(self):  # pylint: disable=useless-super-delegation
        super().__init__()

    def forward(  # pylint: disable=arguments-differ
        self,
        bounding_box_1: AxisAlignedBoundingBox,
        bounding_box_2: AxisAlignedBoundingBox,
    ) -> torch.Tensor:

        # get face norms from one of the boxes:
        face_norms = bounding_box_2.get_face_norms()

        return CollisionPenalty.overlap_penalty(
            CollisionPenalty.get_min_max_overlaps(
                CollisionPenalty.get_min_max_corner_projections(
                    bounding_box_1.corner_onto_axes_projections(face_norms)
                ),
                CollisionPenalty.get_min_max_corner_projections(
                    bounding_box_2.corner_onto_axes_projections(face_norms)
                ),
            )
        )

    @staticmethod
    def get_min_max_corner_projections(projections: torch.Tensor):
        """
        Retrieve the minimum and maximum corner projection (min/max extent in that dimension) for each axis
        Args:
            projections: Tensor(3, 8) -> corner projections onto each of three dimensions

        Returns:
            Tensor(3, 2) -> (min, max) values for each of three dimensions

        """
        check_arg(projections.shape == (3, 8))

        min_indices = torch.min(projections, 1)
        max_indices = torch.max(projections, 1)
        # these are tuples of (values, indices), both of which are tensors

        # helper variable for representing dimension numbers
        # see https://github.com/pytorch/pytorch/issues/24807 re: pylint issue
        dims = torch.tensor([0, 1, 2], dtype=torch.int)  # pylint: disable=not-callable
        # select the indexed items (from a 24 element tensor)
        minima = torch.take(projections, min_indices[1] + (dims * 8))
        maxima = torch.take(projections, max_indices[1] + (dims * 8))
        # stack the minim
        return torch.stack((minima, maxima), 1)

    @staticmethod
    def get_min_max_overlaps(
        min_max_proj_0: torch.Tensor, min_max_proj_1: torch.Tensor
    ) -> torch.Tensor:
        """
        Given min/max corner projections onto 3 axes from two different objects,
        return an interval for each dimension representing the degree of overlap or
        separation between the two objects.
        Args:
            min_max_proj_0: Tensor(3,2) min_max_projections for box 0
            min_max_proj_1: Tensor(3,2) min_max projections for box 1

        Returns:
            (3, 2) tensor -> ranges (start, end) of overlap in each of three dimensions
        """
        check_arg(min_max_proj_0.shape == (3, 2))
        check_arg(min_max_proj_1.shape == (3, 2))

        # see https://github.com/pytorch/pytorch/issues/24807 re: pylint issue
        dims = torch.tensor([0, 1, 2], dtype=torch.int)  # pylint: disable=not-callable

        mins_0 = min_max_proj_0.gather(
            1, torch.tensor([[0], [0], [0]])  # pylint: disable=not-callable
        )
        mins_1 = min_max_proj_1.gather(
            1, torch.tensor([[0], [0], [0]])  # pylint: disable=not-callable
        )

        combined_mins = torch.stack((mins_0, mins_1), 1).squeeze()
        max_indices = torch.max(combined_mins, 1)
        maximum_mins = torch.take(combined_mins, max_indices[1] + (dims * 2))

        # should stick together the minimum parts and the maximum parts
        # with columns like:
        # [ min0x   min1x
        #   min0y   min1y
        #   min0z   min1z
        #                ]
        # then find the maximum element from each row

        # repeat the process for the min of the max projections
        maxs_0 = min_max_proj_0.gather(
            1, torch.tensor([[1], [1], [1]])  # pylint: disable=not-callable
        )
        maxs_1 = min_max_proj_1.gather(
            1, torch.tensor([[1], [1], [1]])  # pylint: disable=not-callable
        )
        combined_maxes = torch.stack((maxs_0, maxs_1), 1).squeeze()
        min_indices = torch.min(combined_maxes, 1)
        minimum_maxes = torch.take(combined_maxes, min_indices[1] + (dims * 2))

        return torch.stack((maximum_mins, minimum_maxes), 1)

    @staticmethod
    def overlap_penalty(min_max_overlaps: torch.Tensor) -> torch.Tensor:
        """
        Return penalty depending on degree of overlap between two 3d boxes.
        Args:
            min_max_overlaps: (3, 2) tensor -> intervals describing degree of overlap between the two boxes

        Returns: Tensor with a positive scalar of the collision penalty, or tensor with zero scalar
        for no collision.
        """
        check_arg(min_max_overlaps.shape == (3, 2))
        # subtract each minimum max from each maximum min:
        overlap_distance = min_max_overlaps[:, 0] - min_max_overlaps[:, 1]

        # if ANY element in overlap_distance is positive, the minimum positive value
        # then the two are not colliding

        for dim in range(3):
            if overlap_distance[dim] >= 0:
                return torch.zeros(1, dtype=torch.float)

        # otherwise the penetration distance is the maximum negative value
        # (the smallest translation that would disentangle the two

        # overlap is represented by a negative value, which we return as a positive penalty
        return overlap_distance.max() * -1 * COLLISION_PENALTY


if __name__ == "__main__":
    main()
