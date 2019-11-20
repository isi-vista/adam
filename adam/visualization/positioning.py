import numpy as np
from numpy import ndarray

from typing import List, Any

from adam.visualization.utils import BoundingBox

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function


def main() -> None:
    bb = BoundingBox.from_center_point(np.array([0, 0, 0]))
    midpoint_minimization(bb)


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


class NonIntersection(Function):
    @staticmethod
    def forward(ctx: Any, *args, **kwargs) -> Any:
        center = args[0]
        reference_bb = args[1]
        # context can be used to store tensors that can be accessed during backward step
        ctx.save_for_backward(center)

        bb = BoundingBox.from_center_point(center.data.numpy())

        dist = bb.minkowski_diff_distance(reference_bb)

        # TODO: INCLUDE distance from regular-space origin as well in objective
        print(f"Minkowski diff distance\n{dist}\n")

        # not colliding or right next to one another
        if dist <= 0:
            return torch.tensor([0.0])
        else:
            # colliding
            return torch.tensor([dist])

        # needs to return the badness of the positioning

        # need to use `mark_non_differentiable()` to tell the engine if an output is non-differentiable

        # need to use `save_for_backward()` to save any input for later use by `backward()`

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_output = grad_outputs[0]
        # formula for differentiating the operation. takes as many outputs as the forward step returned,
        # returns as many tensors as there were inputs to forward()
        center, = ctx.saved_tensors
        grad_center = None
        print(f"gradient output: {grad_output[0]}")

        if ctx.needs_input_grad[0]:
            print(center * (grad_output ** 2))
            grad_center = center * (grad_output ** 2)

        return grad_center, None

        # each argument is the gradient w/r/t the given output, each return val should b


def midpoint_minimization(relative_bb: BoundingBox):

    # relative_bb_faces = relative_bb.all_face_normals()
    #
    # relative_bb_corners = relative_bb.all_corners()
    #
    # relative_bb_min_max_projections = [
    #     min_max_projection(relative_bb_corners, face) for face in relative_bb_faces
    # ]

    device = torch.device("cpu")
    dtype = torch.float

    # box variable, initialized to a 3d position w/ normal distribution around 0,0,0 stdev (3, 1, 1)
    center = torch.normal(
        torch.tensor([0, 0, 0], device=device, dtype=dtype),
        torch.tensor([3, 1, 1], device=device, dtype=dtype),
    )
    center.requires_grad_(True)

    optimizer = optim.SGD([center], lr=0.01, momentum=0.5)

    # training:
    iterations = 10
    for _ in range(iterations):
        optimizer.zero_grad()
        predictions = NonIntersection.apply(center, relative_bb)

        # automagically gets something passed to it?
        predictions.backward()
        optimizer.step()

        print(f"error: {predictions.data[0]}")
        print(f"center_var: {center}")


if __name__ == "__main__":
    main()
