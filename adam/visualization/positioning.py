from itertools import combinations

import numpy as np
from immutablecollections import immutabledict, immutableset
from torch.nn import Parameter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from vistautils.preconditions import check_arg

from attr import attrs, attrib
from numpy import ndarray

from typing import List, Any, Set, Mapping

from adam.visualization.utils import BoundingBox, min_max_projection

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function


@attrs(frozen=True, slots=True)
class AxisAlignedBoundingBox:
    center: torch.Tensor = attrib()

    def center_distance_from_point(self, point: torch.Tensor) -> torch.Tensor:
        return torch.dist(self.center, point, 2)

    @staticmethod
    def create_at_random_position(
        *, min_distance_from_origin: float, max_distance_from_origin: float
    ):
        # np.random.seed(3030)

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
        return AxisAlignedBoundingBox(Parameter(torch.tensor(center, dtype=torch.float), requires_grad=True))

    def get_corners(self) -> torch.Tensor:
        return torch.ones((8, 3)) * self.center + torch.tensor([
            [-1, -1, -1],
            [1, -1, -1],
            [-1, 1, -1],
            [1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [-1, 1, 1],
            [1, 1, 1],
        ], dtype=torch.float)

    def right_corner(self):
        return self.center + torch.tensor([
            1, -1, -1
        ])

    def forward_corner(self):
        return self.center + torch.tensor([
            -1, 1, -1
        ])

    def up_corner(self):
        return self.center + torch.tensor([
            -1, -1, 1
        ])

    def zero_corner(self):
        return self.center + torch.tensor([
            -1, -1, -1
        ])

    def right_face(self):
        diff = self.right_corner() - self.zero_corner()
        return diff / torch.norm(diff)

    def forward_face(self):
        diff = self.forward_corner() - self.zero_corner()
        return diff / torch.norm(diff)

    def up_face(self):
        diff = self.up_corner() - self.zero_corner()
        return diff / torch.norm(diff)

    def get_face_norms(self) -> torch.Tensor:
        # in axis-aligned case these are always the same
        return torch.stack([
            self.right_face(),
            self.forward_face(),
            self.up_face()
        ])

    @staticmethod
    def get_min_max_projections(projections: torch.Tensor):
        """

        Args:
            projections: (3, 8) tensor -> corner projections onto each of three dimensions

        Returns:
            (3, 2) tensor -> (min, max) values for each of three dimensions

        """
        return torch.tensor([
            [torch.min(projections[i]), torch.max(projections[i])] for i in range(3)
        ], dtype=torch.float)

    def get_projections(self, axes: torch.Tensor):
        """

        Args:
            axes: (3,3) tensor -> the three axes we are projecting points onto

        Returns:
            (3, 8) tensor -> each point projected onto each of three dimensions

        """
        check_arg(axes.shape == (3,3))
        corners = self.get_corners()
        return torch.tensor([
            corners[j].dot(axes[i]) for i in range(3) for j in range(8)
        ], dtype=torch.float).reshape(3, 8)

    @staticmethod
    def min_max_overlaps(min_max_proj_0: torch.Tensor, min_max_proj_1: torch.Tensor) -> torch.Tensor:
        """

        Args:
            min_max_proj_0: min_max_projections for box 0
            min_max_proj_1: min_max projections for box 1
                (each is a (3, 2) tensor)

        Returns:
            (3, 2) tensor -> ranges (start, end) of overlap in each of three dimensions
        """
        return torch.tensor(
            [
                [
                    torch.max(
                        min_max_proj_0[i][0], min_max_proj_1[i][0]
                    ),
                    torch.min(
                        min_max_proj_0[i][1], min_max_proj_1[i][1]
                    ),
                ]
                for i in range(3)
            ], dtype=torch.float
        )

    @staticmethod
    def overlap_penalty(min_max_overlaps: torch.Tensor) -> torch.Tensor:
        """

        Args:
            min_max_overlaps: (3, 2) tensor -> intervals of how much the boxes overlap in each dimension

        Returns: Tensor with a scalar of the collision penalty size

        """
        # print(f"overlap penalty: min_max_overlaps shape: {min_max_overlaps.shape}")

        overlap_distance = torch.tensor(
            [min_max_overlaps[i][0] - min_max_overlaps[i][1] for i in range(3)],
            dtype=torch.float
        )
        # print(overlap_distance.shape)

        # if ANY element in overlap_distance is positive, the minimum positive value
        # then the two are not colliding

        # otherwise the penetration distance is the maximum negative value
        # (the smallest translation that would disentangle the two

        # if not colliding
        for dim in range(3):
            if overlap_distance[dim] >= 0:
                return overlap_distance[dim] * 0

        smallest_colliding_dim = 0
        for dim in range(3):
            if 0 > overlap_distance[dim] >= overlap_distance[smallest_colliding_dim]:
                smallest_colliding_dim = dim

        return overlap_distance[smallest_colliding_dim] * -1


class PositionSolver:
    # constants to eventually use
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


class CollisionPenalty(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, bounding_box_1: AxisAlignedBoundingBox,
                bounding_box_2: AxisAlignedBoundingBox) -> torch.Tensor:

        # get face norms from one of the boxes:
        face_norms = bounding_box_2.get_face_norms()

        return AxisAlignedBoundingBox.overlap_penalty(
            AxisAlignedBoundingBox.min_max_overlaps(
                AxisAlignedBoundingBox.get_min_max_projections(
                    bounding_box_1.get_projections(face_norms)
                ),
                AxisAlignedBoundingBox.get_min_max_projections(
                    bounding_box_2.get_projections(face_norms)
                )
            )
        )



ORIGIN = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float)


@attrs(frozen=True, auto_attribs=True)
class AdamObject:
    name: str


class DistanceFromOriginPenalty(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, bounding_box: AxisAlignedBoundingBox) -> torch.Tensor:
        return bounding_box.center_distance_from_point(ORIGIN)


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
    def for_objects(adam_objects: Set[AdamObject]) -> "AdamObjectPositioningModel":
        objects_to_bounding_boxes = immutabledict(
            (
                adam_object,
                AxisAlignedBoundingBox.create_at_random_position(
                    min_distance_from_origin=5, max_distance_from_origin=10
                ),
            )
            for adam_object in adam_objects
        )
        return AdamObjectPositioningModel(objects_to_bounding_boxes)

    def forward(self) -> torch.Tensor:
        distance_penalty = sum(
            self.distance_to_origin_penalty(box) for box in self.object_bounding_boxes
        )
        collision_penalty = sum(
            self.collision_penalty(box1, box2)
            for (box1, box2) in combinations(self.object_bounding_boxes, 2)
        )
        print(f"distance penalty: {distance_penalty}\ncollision penalty: {collision_penalty}")
        return collision_penalty

    def dump_object_positions(self) -> None:
        for (adam_object, bounding_box) in self.adam_object_to_bounding_box.items():
            print(f"{adam_object.name} = {bounding_box.center.data}")


def main() -> None:
    ball = AdamObject(name="ball")
    box = AdamObject(name="box")

    positioning_model = AdamObjectPositioningModel.for_objects(immutableset([ball, box]))
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


if __name__ == "__main__":
    main()
