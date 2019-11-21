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
        return AxisAlignedBoundingBox(Parameter(torch.tensor(center), requires_grad=True))


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
        # get all corners for current center
        corners = torch.ones((8, 3)) * self.center + self.corner_matrix
        print(f"corners.shape: {corners.shape}")
        # project corners onto faces from static bb (output shape (3, 8, 1) )

        projections = torch.tensor(
            [corners[j].dot(self.static_bb_faces[i]) for i in range(3) for j in range(8)],
            dtype=torch.float,
        ).reshape((3, 8, 1))
        print(f"corner projections shape: {projections.shape}")
        print(f"corner projections: {projections}")

        face_min_max = torch.tensor(
            [[torch.min(projections[i]), torch.max(projections[i])] for i in range(3)],
            dtype=torch.float,
        )
        print(f"face min-maxes shape: {face_min_max.shape}")  # (3, 2)
        print(f"face min-maxes: {face_min_max}")

        # compute overlap between the parameter's min/maxes and the static ones

        overlap_ranges = torch.tensor(
            [
                [
                    torch.max(
                        face_min_max[i][0], self.static_bb_min_max_projections[i][0]
                    ),
                    torch.min(
                        face_min_max[i][1], self.static_bb_min_max_projections[i][1]
                    ),
                ]
                for i in range(3)
            ],
            dtype=torch.float,
        )

        print(f"shape of overlap ranges: {overlap_ranges.shape}")
        print(f" overlap ranges: {overlap_ranges}")

        # condense this down in to a scalar for each of the 3 (x, y, z) dimensions:
        # to represent the degree of overlap / separation

        overlap_distance = torch.tensor(
            [overlap_ranges[i][0] - overlap_ranges[i][1] for i in range(3)],
            dtype=torch.float,
        )

        print(f"shape of overlap distance: {overlap_distance.shape}")  # should be (3, 1)
        print(f"overlap distance: {overlap_distance}")

        # if ANY element in overlap_distance is positive, the minimum positive value
        # is the separation distance

        # otherwise the penetration distance is the maximum negative value
        # (the smallest translation that would disentangle the two

    def forward(self, *input: Any, **kwargs: Any) -> Any:
        return self.center.dist(torch.tensor([0, 0, 0], dtype=torch.float, device="cpu"))
        # bb = BoundingBox.from_center_point(self.center.data.numpy())
        # dist = bb.minkowski_diff_distance(self.static_bb)
        # if dist <= 0:
        #     return torch.tensor([0.0], requires_grad=True)
        # else:
        #     return torch.tensor([dist], requires_grad=True)


ORIGIN = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)


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
        return distance_penalty + collision_penalty

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

    iterations = 40
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
