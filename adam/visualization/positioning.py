"""
This module defines a bounding box type and implements a constraint solver
that can position multiple bounding boxes s/t they do not overlap
(in addition to other constraints).
"""
from itertools import combinations
from typing import Mapping, AbstractSet, Tuple, Optional, List
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
ORIGIN = torch.zeros(3, dtype=torch.float)  # pylint: disable=not-callable
GRAVITY_PENALTY = torch.tensor([1], dtype=torch.float)  # pylint: disable=not-callable
BELOW_GROUND_PENALTY = 2 * GRAVITY_PENALTY
COLLISION_PENALTY = 5 * GRAVITY_PENALTY


@attrs(frozen=True, auto_attribs=True)
class AdamObject:
    """Used for testing purposes, to attach a name to a bounding box"""

    name: str
    initial_position: Optional[Tuple[float, float, float]]


def main() -> None:
    ball = AdamObject(name="ball", initial_position=None)
    box = AdamObject(name="box", initial_position=None)

    cardboard_box = AdamObject(name="cardboardBox", initial_position=None)
    aardvark = AdamObject(name="aardvark", initial_position=None)
    flamingo = AdamObject(name="flamingo", initial_position=None)

    positioning_model = AdamObjectPositioningModel.for_objects_random_positions(
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

    iterations = 100
    for iteration in range(iterations):
        print(f"====== Iteration {iteration} ======")
        positioning_model.dump_object_positions(prefix="\t")

        loss = positioning_model()
        print(f"\tLoss: {loss.item()}")
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        learning_rate_schedule.step(loss)

    print("========= Final Positions ========")
    positioning_model.dump_object_positions(prefix="\t")


def run_model(
    objs: List[AdamObject], num_iterations: int = 200, yield_steps: Optional[int] = None
) -> List[torch.Tensor]:
    """
    Construct a positioning model given a list of objects to position, return their final position values
    Args:
        objs: list of AdamObjects requested to be positioned
        num_iterations: total number of SGD iterations.
        yield_steps: If provided, the current positions of all objects will be returned after this many steps

    Returns: List of (3,) tensors corresponding to the positions of the objs list

    """
    positioning_model = AdamObjectPositioningModel.for_objects(immutableset(objs))

    # we will start with an aggressive learning rate
    optimizer = optim.SGD(positioning_model.parameters(), lr=1.0)
    # but will decrease it whenever the loss plateaus
    learning_rate_schedule = ReduceLROnPlateau(
        optimizer,
        "min",
        # decrease the rate if the loss hasn't improved in
        # 5 epochs
        patience=5,
    )

    iterations = num_iterations
    for i in range(iterations):
        print(f"====== Iteration {i} =======")
        loss = positioning_model()
        print(f"\tLoss: {loss.item()}")
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        learning_rate_schedule.step(loss)

        positioning_model.dump_object_positions(prefix="\t")
        if yield_steps and i % yield_steps == 0:
            yield [positioning_model.get_object_position(obj).data for obj in objs]

    return [positioning_model.get_object_position(obj).data for obj in objs]


@attrs(frozen=True, slots=True)
class AxisAlignedBoundingBox:
    """
    Defines a 3D Box that is oriented to world axes.

    This box is defined by a center point of shape (3,),
    and a scale (also of shape (3,)) which defines how a unit cube
    with the given center will be scaled in each dimension to create
    this box.

    For example: a box centered at (0, 0, 0) and with a scale of (1, 1, 1) would have opposite
    corners at (-1, -1, -1) and (1, 1, 1), giving the box a volume of 2(^3)
    """

    center: torch.Tensor = attrib()  # tensor shape: (3,)
    scale: torch.Tensor = attrib()  # tensor shape: (3, 3) - diagonal matrix
    # rotation: torch.Tensor = attrib()

    def center_distance_from_point(self, point: torch.Tensor) -> torch.Tensor:
        return torch.dist(self.center, point, 2)

    def distance_from_lowest_corner_to_ground(self) -> torch.Tensor:
        """
        Returns distance of z value of minimum (w/r/t Z coordinate) corner of the box from origin's Z value.
        Returns: (1,) tensor

        """
        corners = self.get_corners()
        min_corner_z = torch.min(
            torch.gather(
                corners,
                1,
                torch.repeat_interleave(torch.tensor([[2]]), torch.tensor([8]), dim=0),
            )
        )
        return min_corner_z - ORIGIN[2]

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

    def _minus_ones_corner(self) -> torch.Tensor:
        """
        Corner in the direction of the negative x, y, z axes from center
        Returns: Tensor (3,)

        """
        return self.center + torch.tensor(  # pylint: disable=not-callable
            [-1, -1, -1], dtype=torch.float
        ).matmul(self.scale)

    # functions returning normal vectors from three perpendicular faces of the box
    def right_face_normal_vector(self) -> torch.Tensor:
        """
        Normal vector for the right face of the box (toward positive x axis when aligned to world axes)
        Returns: Tensor (3,)
        """
        diff = (
            self.center
            + torch.tensor(  # pylint: disable=not-callable
                [1, -1, -1], dtype=torch.float
            ).matmul(self.scale)
            - self._minus_ones_corner()
        )
        return diff / torch.norm(diff)

    def forward_face_normal_vector(self) -> torch.Tensor:
        """
        Normal vector for the forward face of the box (toward positive y axis when aligned to world axes)
        Returns: Tensor (3,)
        """
        diff = (
            self.center
            + torch.tensor(  # pylint: disable=not-callable
                [-1, 1, -1], dtype=torch.float
            ).matmul(self.scale)
            - self._minus_ones_corner()
        )
        return diff / torch.norm(diff)

    def up_face_normal_vector(self) -> torch.Tensor:
        """
        Normal vector for the up face of the box (toward positive z axis when aligned to world axes)
        Returns: Tensor (3,)
        """
        diff = (
            self.center
            + torch.tensor(  # pylint: disable=not-callable
                [-1, -1, 1], dtype=torch.float
            ).matmul(self.scale)
            - self._minus_ones_corner()
        )
        return diff / torch.norm(diff)

    def face_normal_vectors(self) -> torch.Tensor:
        """
        Stacks the face norms from the right, forward, and up faces of the box
        Returns: Tensor (3,3)

        """
        # in axis-aligned case these are always the same
        return torch.stack(
            [
                self.right_face_normal_vector(),
                self.forward_face_normal_vector(),
                self.up_face_normal_vector(),
            ]
        )

    def corners_onto_axes_projections(self, axes: torch.Tensor) -> torch.Tensor:
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


class AdamObjectPositioningModel(torch.nn.Module):  # type: ignore
    """
    Model that combines multiple constraints on AxisAlignedBoundingBoxes.
    """

    def __init__(
        self, adam_object_to_bounding_box: Mapping[AdamObject, AxisAlignedBoundingBox]
    ) -> None:
        super().__init__()
        self.adam_object_to_bounding_box = adam_object_to_bounding_box
        self.object_bounding_boxes = adam_object_to_bounding_box.values()

        for (adam_object, bounding_box) in self.adam_object_to_bounding_box.items():
            self.register_parameter(adam_object.name, bounding_box.center)

        self.collision_penalty = CollisionPenalty()
        self.below_ground_penalty = BelowGroundPenalty()
        self.weak_gravity_penalty = WeakGravityPenalty()

    @staticmethod
    def for_objects(
        adam_objects: AbstractSet[AdamObject]
    ) -> "AdamObjectPositioningModel":
        objects_to_bounding_boxes: ImmutableDict[
            AdamObject, AxisAlignedBoundingBox
        ] = immutabledict(
            (
                adam_object,
                AxisAlignedBoundingBox.create_at_center_point(
                    center=np.array(adam_object.initial_position)
                ),
            )
            for adam_object in adam_objects
        )
        return AdamObjectPositioningModel(objects_to_bounding_boxes)

    @staticmethod
    def for_objects_random_positions(
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

    def forward(self):  # pylint: disable=arguments-differ
        collision_penalty = sum(
            self.collision_penalty(box1, box2)
            for (box1, box2) in combinations(self.object_bounding_boxes, 2)
        )
        below_ground_penalty = sum(
            self.below_ground_penalty(box) for box in self.object_bounding_boxes
        )
        weak_gravity_penalty = sum(
            self.weak_gravity_penalty(box) for box in self.object_bounding_boxes
        )
        print(
            f"collision penalty: {collision_penalty}"
            f"\nout of bounds penalty: {below_ground_penalty}"
            f"\ngravity penalty: {weak_gravity_penalty}"
        )
        return collision_penalty + below_ground_penalty + weak_gravity_penalty

    def dump_object_positions(self, *, prefix: str = "") -> None:
        for (adam_object, bounding_box) in self.adam_object_to_bounding_box.items():
            print(f"{prefix}{adam_object.name} = {bounding_box.center.data}")

    def get_object_position(self, obj: AdamObject) -> torch.Tensor:
        """
        Retrieves the (center) position of an AdamObject contained in this model.
        Args:
            obj: AdamObject whose position is requested

        Returns: (3,) tensor of the requested object's position.

        Raises KeyError if an AdamObject not contained in this model is queried.
        """
        return self.adam_object_to_bounding_box[obj].center.data


class BelowGroundPenalty(nn.Module):  # type: ignore
    """
    Model that penalizes boxes lying outside of the scene (i.e. below the ground plane) or off-camera)
    """

    def __init(self) -> None:  # pylint: disable=useless-super-delegation
        super().__init__()

    def forward(self, *inputs):  # pylint: disable=arguments-differ
        bounding_box: AxisAlignedBoundingBox = inputs[0]
        distance_above_ground = bounding_box.distance_from_lowest_corner_to_ground()
        if distance_above_ground >= 0:
            return 0
        else:
            return -distance_above_ground


class WeakGravityPenalty(nn.Module):  # type: ignore
    """
    Model that penalizes boxes that are not resting on the ground.
    """

    # TODO: exempt birds from this constraint
    # TODO: exempt things resting on top of other objects from this constraint

    def __init__(self) -> None:  # pylint: disable=useless-super-delegation
        super().__init__()

    def forward(self, *inputs):  # pylint: disable=arguments-differ
        bounding_box: AxisAlignedBoundingBox = inputs[0]
        distance_above_ground = bounding_box.distance_from_lowest_corner_to_ground()
        if distance_above_ground <= 0:
            return 0.0
        else:
            # a linear penalty leads to a constant gradient, just like real gravity
            return GRAVITY_PENALTY * distance_above_ground


class CollisionPenalty(nn.Module):  # type: ignore
    """
    Model that penalizes boxes that are colliding with other boxes.
    """

    def __init__(self):  # pylint: disable=useless-super-delegation
        super().__init__()

    def forward(self, *inputs):  # pylint: disable=arguments-differ
        bounding_box_1: AxisAlignedBoundingBox = inputs[0]
        bounding_box_2: AxisAlignedBoundingBox = inputs[1]

        # get face norms from one of the boxes:
        face_norms = bounding_box_2.face_normal_vectors()

        return CollisionPenalty.overlap_penalty(
            CollisionPenalty.get_min_max_overlaps(
                CollisionPenalty.get_min_max_corner_projections(
                    bounding_box_1.corners_onto_axes_projections(face_norms)
                ),
                CollisionPenalty.get_min_max_corner_projections(
                    bounding_box_2.corners_onto_axes_projections(face_norms)
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
            (3, 2) tensor -> ranges (start, end) of overlap OR separation in each of three dimensions.
            If (start - end) is positive, this indicates that the boxes do not overlap along this dimension,
            otherwise, a negative value indicates an overlap along that dimension.
        """
        check_arg(min_max_proj_0.shape == (3, 2))
        check_arg(min_max_proj_1.shape == (3, 2))

        # see https://github.com/pytorch/pytorch/issues/24807 re: pylint issue
        dims = torch.tensor([0, 1, 2], dtype=torch.int)  # pylint: disable=not-callable

        mins_0 = min_max_proj_0.gather(1, torch.zeros((3, 1), dtype=torch.long))
        mins_1 = min_max_proj_1.gather(1, torch.zeros((3, 1), dtype=torch.long))

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
        maxs_0 = min_max_proj_0.gather(1, torch.ones((3, 1), dtype=torch.long))
        maxs_1 = min_max_proj_1.gather(1, torch.ones((3, 1), dtype=torch.long))
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

        # as long as at least one dimension's overlap distance is positive (not overlapping),
        # then the boxes are not colliding
        for dim in range(3):
            if overlap_distance[dim] >= 0:
                return torch.zeros(1, dtype=torch.float)

        # otherwise the penetration distance is the maximum negative value
        # (the smallest translation that would disentangle the two

        # overlap is represented by a negative value, which we return as a positive penalty
        return overlap_distance.max() * -1 * COLLISION_PENALTY


if __name__ == "__main__":
    main()
