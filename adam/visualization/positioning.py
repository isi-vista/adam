import numpy as np
from attr import attrs, attrib
from numpy import ndarray

from typing import List, Any

from adam.visualization.utils import BoundingBox, min_max_projection

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function


def main() -> None:
    bb = BoundingBox.from_center_point(np.array([0, 0, 0]))
    midpoint_minimization(bb)


@attrs(frozen=True, slots=True)
class AxisAlignedBoundingBox:
    center: torch.Tensor = attrib()

    def center_distance_from_point(self, point: torch.Tensor) -> torch.Tensor:
        return torch.dist(self.center, point, 2)


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
    def __init__(self, static_bb):
        super().__init__()
        device = torch.device("cpu")
        dtype = torch.float
        torch.random.manual_seed(2014)

        self.static_bb = static_bb
        self.static_bb_faces = torch.tensor(
            self.static_bb.all_face_normals(), dtype=dtype
        )
        print(f"static bb faces: {self.static_bb_faces}")

        static_bb_corners = torch.tensor(self.static_bb.all_corners(), dtype=dtype)
        print(f"static bb corners: {static_bb_corners}")

        self.static_bb_min_max_projections = torch.tensor(
            [
                min_max_projection(static_bb_corners, face)
                for face in self.static_bb_faces
            ],
            dtype=dtype,
        )

        print(f"static_bb min/max projections: {self.static_bb_min_max_projections}")

        # box variable, initialized to a 3d position w/ normal distribution around 0,0,0 stdev (3, 1, 1)
        self.center = torch.normal(
            torch.tensor([0, 0, 0], device=device, dtype=dtype),
            torch.tensor([3, 1, 1], device=device, dtype=dtype),
        )
        # self.center = torch.zeros(3, dtype=dtype, device=device)
        self.center.requires_grad_(True)
        self.center = nn.Parameter(self.center)
        print(self.center)

        self.corner_matrix = torch.tensor(
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
            dtype=dtype,
        )
        self.minkowski()

    def minkowski(self):
        # TODO: see if requires_grad=True is needed for each of these intermediate tensors
        print(f"center: {self.center}")
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


ORIGIN = torch.tensor([0.0, 0.0, 0.0])


class DistanceFromOriginPenalty(nn.Module):
    def __init__(self, bounding_box: AxisAlignedBoundingBox) -> None:
        super().__init__()
        self.bounding_box = bounding_box

    def forward(self, *input: Any, **kwargs: Any) -> torch.Tensor:
        return self.bounding_box.center_distance_from_point(ORIGIN)


def midpoint_minimization(relative_bb: BoundingBox):

    model = CollisionPenalty(relative_bb).to("cpu")

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    loss_fn = nn.MSELoss()

    y = torch.tensor([0.0])  # no collision

    # training:
    print("\n\nTraining\n")

    iterations = 10
    for _ in range(iterations):
        model.train()  # set model to train mode
        # error = NonIntersection.apply(center, relative_bb)
        print(f"params: {model.state_dict()}")
        y_hat = model()
        print(f"error: {y_hat}")
        print(type(y_hat))
        # loss = loss_fn(y, y_hat)
        loss = y_hat ** 2

        # automagically gets something passed to it?
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(loss.item())


if __name__ == "__main__":
    main()
