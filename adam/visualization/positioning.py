"""
This module defines a bounding box type and implements a constraint solver
that can position multiple bounding boxes s/t they do not overlap
(in addition to other constraints).

main() function used for testing purposes. Primary function made available to outside callers
is run_model()
"""
from itertools import combinations
from typing import Mapping, AbstractSet, Optional, List, Iterable
from attr import attrs, attrib

import numpy as np
from numpy import ndarray
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import PairwiseDistance

from immutablecollections import immutabledict, immutableset, ImmutableDict, ImmutableSet
from vistautils.preconditions import check_arg

from adam.axes import (
    Axes,
    HorizontalAxisOfObject,
    straight_up,
    # directed,
    symmetric,
    symmetric_vertical,
    FacingAddresseeAxis,
)

from adam.ontology.phase1_spatial_relations import (
    PROXIMAL,
    DISTAL,
    EXTERIOR_BUT_IN_CONTACT,
    Region,
)
from adam.ontology.phase1_spatial_relations import (
    Direction,
    GRAVITATIONAL_UP,
    GRAVITATIONAL_DOWN,
)
from adam.perception import ObjectPerception

# see https://github.com/pytorch/pytorch/issues/24807 re: pylint issue


ORIGIN = torch.zeros(3, dtype=torch.float)  # pylint: disable=not-callable
GRAVITY_PENALTY = torch.tensor([1], dtype=torch.float)  # pylint: disable=not-callable
BELOW_GROUND_PENALTY = 2 * GRAVITY_PENALTY
COLLISION_PENALTY = 1 * GRAVITY_PENALTY

LOSS_EPSILON = 1.0e-04

ANGLE_PENALTY = torch.tensor([5], dtype=torch.float)  # pylint: disable=not-callable
DISTANCE_PENALTY = torch.tensor([1], dtype=torch.float)  # pylint: disable=not-callable

PROXIMAL_MIN_DISTANCE = torch.tensor(  # pylint: disable=not-callable
    [0.5], dtype=torch.float
)
PROXIMAL_MAX_DISTANCE = torch.tensor(  # pylint: disable=not-callable
    [2], dtype=torch.float
)

DISTAL_MIN_DISTANCE = torch.tensor([4], dtype=torch.float)  # pylint: disable=not-callable

EXTERIOR_BUT_IN_CONTACT_EPS = torch.tensor(  # pylint: disable=not-callable
    [1e-5], dtype=torch.float
)


def main() -> None:

    top_to_bottom = straight_up("top-to-bottom")
    side_to_side_0 = symmetric("side-to-side-0")
    side_to_side_1 = symmetric("side-to-side-1")

    box = ObjectPerception(
        "box",
        geon=None,
        axes=Axes(
            primary_axis=top_to_bottom,
            orienting_axes=immutableset([side_to_side_0, side_to_side_1]),
        ),
    )

    generating_axis = symmetric_vertical("ball-generating")
    orienting_axis_0 = symmetric("ball-orienting-0")
    orienting_axis_1 = symmetric("ball-orienting-1")

    # ball situated on top of box
    ball = ObjectPerception(
        "ball",
        geon=None,
        axes=Axes(
            primary_axis=generating_axis,
            orienting_axes=immutableset([orienting_axis_0, orienting_axis_1]),
        ),
    )

    in_region_relations: Mapping[ObjectPerception, List[Region[ObjectPerception]]] = {
        ball: [Region[ObjectPerception](box, EXTERIOR_BUT_IN_CONTACT, GRAVITATIONAL_UP)]
    }

    # other objects have no particular constraints:

    positioning_model = PositioningModel.for_objects_random_positions(
        immutableset([ball, box]), in_region_relations=in_region_relations
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


@attrs(frozen=True, auto_attribs=True)
class PositionsMap:
    """Convenience type: list of positions corresponding to objects in a scene."""

    name_to_position: Mapping[str, torch.Tensor]

    def __len__(self) -> int:
        return len(self.name_to_position)


def run_model(
    top_level_objects: ImmutableSet[ObjectPerception],
    in_region_map: Mapping[ObjectPerception, List[Region[ObjectPerception]]],
    *,
    num_iterations: int = 200,
    yield_steps: Optional[int] = None,
) -> Iterable[PositionsMap]:
    r"""
    Construct a positioning model given a list of objects to position, return their position values.
    The model will return final positions either after the given number of iterations, or if the model
    converges in a position where it is unable to find a gradient to continue.
    Args:
        top_level_objects: set of top-level objects requested to be positioned
        in_region_map: in-region relations for all top-level and sub-objects in this scene
        *num_iterations*: total number of SGD iterations.
        *yield_steps*: If provided, the current positions of all objects will be returned after this many steps

    Returns: PositionsMap: Map of object name -> Tensor (3,) of its position

    """
    positioning_model = PositioningModel.for_objects_random_positions(
        top_level_objects, in_region_relations=in_region_map
    )

    # we will start with an aggressive learning rate
    optimizer = optim.SGD(positioning_model.parameters(), lr=1.0)
    # but will decrease it whenever the loss plateaus
    learning_rate_schedule = ReduceLROnPlateau(
        optimizer,
        "min",
        # decrease the rate if the loss hasn't improved in
        # 10 epochs
        patience=10,
    )

    iterations = num_iterations
    for i in range(iterations):
        print(f"====== Iteration {i} =======")
        loss = positioning_model()
        # if we lose any substantial gradient, stop the search
        if loss < LOSS_EPSILON:
            break
        print(f"\tLoss: {loss.item()}")
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        learning_rate_schedule.step(loss)

        positioning_model.dump_object_positions(prefix="\t")
        if yield_steps and i % yield_steps == 0:
            yield positioning_model.get_objects_positions()

    return positioning_model.get_objects_positions()


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

    def nearest_center_face_distance_from_point(
        self, point: torch.Tensor
    ) -> torch.Tensor:
        """ returns the distance from the closest face center to the given point

        Args:
            point: tensor (3,) of a coordinate to check distance from this box's face centers
        Returns: tensor (1,)

        """
        centers = self.get_face_centers()
        return torch.min(PairwiseDistance().forward(centers, point.expand(6, 3)))

    def nearest_center_face_point(self, point: torch.Tensor) -> torch.Tensor:
        """
        Returns the closest face center of the box to the given point
        Args:
            point: tensor (3,) x,y,z coordinate: an arbitrary point in 3d space

        Returns: tensor (3,) x,y,z coordinate: the center face of the box closest to the function argument

        """
        # TODO: need to get the row INDEX for the minimum face distance
        face_centers = self.get_face_centers()
        return face_centers[
            torch.argmin(PairwiseDistance().forward(face_centers, point.expand(6, 3)))
        ]

    def z_coordinate_of_lowest_corner(self) -> torch.Tensor:
        """
        Get the position of the lowest corner of the box.

        The third coordinate is interpreted as the height relative to the ground at z=0.
        Returns: (1,) tensor

        """
        # all corners are considered (in case of a rotated box)
        corners = self.get_corners()
        # find the minimum z value
        min_corner_z = torch.min(
            # gather just the z coordinate from the tensor of all corners
            # turning (8,3) into (8,1)
            torch.gather(
                corners,
                1,
                # create a (8, 1) tensor filled with elements corresponding to the index of the Z coordinate
                # these indices are used by torch.gather() to retrieve the correct elements from the corners tensor
                torch.repeat_interleave(
                    torch.tensor([[2]]),  # pylint: disable=not-callable
                    torch.tensor([8]),  # pylint: disable=not-callable
                    dim=0,
                ),
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

    def get_face_centers(self) -> torch.Tensor:
        """
        Returns the center point of each of the box's 6 faces.
        Returns: tensor (6, 3)
        """
        corners = self.get_corners()
        return torch.stack(  # pylint: disable=not-callable
            [
                torch.div(corners[0] + corners[6], 2),  # left face
                torch.div(corners[0] + corners[5], 2),  # backward face
                torch.div(corners[1] + corners[7], 2),  # right face
                torch.div(corners[2] + corners[7], 2),  # forward face
                torch.div(corners[0] + corners[3], 2),  # bottom face
                torch.div(corners[4] + corners[7], 2),  # top face
            ],
            dim=0,
        )

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


class PositioningModel(torch.nn.Module):  # type: ignore
    """
    Model that combines multiple constraints on AxisAlignedBoundingBoxes.
    """

    def __init__(
        self,
        object_perception_to_bounding_box: Mapping[
            ObjectPerception, AxisAlignedBoundingBox
        ],
        in_region_relations: Mapping[ObjectPerception, List[Region[ObjectPerception]]],
    ) -> None:
        super().__init__()
        self.object_perception_to_bounding_box = object_perception_to_bounding_box
        self.in_region_relations = in_region_relations
        self.object_bounding_boxes = object_perception_to_bounding_box.values()

        for (
            object_perception,
            bounding_box,
        ) in self.object_perception_to_bounding_box.items():
            # suppress mypy error about supplying a Tensor where it expects a Parameter
            self.register_parameter(  # type: ignore
                object_perception.debug_handle, bounding_box.center
            )

        self.collision_penalty = CollisionPenalty()
        self.below_ground_penalty = BelowGroundPenalty()
        self.weak_gravity_penalty = WeakGravityPenalty()
        self.in_region_penalty = InRegionPenalty(
            object_perception_to_bounding_box, in_region_relations
        )

    @staticmethod
    def for_objects_random_positions(
        object_perceptions: AbstractSet[ObjectPerception],
        *,
        in_region_relations: Mapping[ObjectPerception, List[Region[ObjectPerception]]],
    ) -> "PositioningModel":
        objects_to_bounding_boxes: ImmutableDict[
            ObjectPerception, AxisAlignedBoundingBox
        ] = immutabledict(
            (
                object_perception,
                AxisAlignedBoundingBox.create_at_random_position(
                    min_distance_from_origin=10, max_distance_from_origin=20
                ),
            )
            for object_perception in object_perceptions
        )
        return PositioningModel(objects_to_bounding_boxes, in_region_relations)

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
        in_region_penalty = sum(
            self.in_region_penalty(
                object_perception,
                immutableset(self.in_region_relations[object_perception]),
            )
            for object_perception in self.object_perception_to_bounding_box
            if object_perception in self.in_region_relations
        )
        print(
            f"collision penalty: {collision_penalty}"
            f"\nout of bounds penalty: {below_ground_penalty}"
            f"\ngravity penalty: {weak_gravity_penalty}"
            f"\nin-region penalty: {in_region_penalty}"
        )
        return (
            collision_penalty
            + below_ground_penalty
            + weak_gravity_penalty
            + in_region_penalty
        )

    def dump_object_positions(self, *, prefix: str = "") -> None:
        for (
            object_perception,
            bounding_box,
        ) in self.object_perception_to_bounding_box.items():
            print(
                f"{prefix}{object_perception.debug_handle} = {bounding_box.center.data}"
            )

    def get_object_position(self, obj: ObjectPerception) -> torch.Tensor:
        """
        Retrieves the (center) position of an AdamObject contained in this model.
        Args:
            obj: AdamObject whose position is requested

        Returns: (3,) tensor of the requested object's position.

        Raises KeyError if an AdamObject not contained in this model is queried.
        """
        return self.object_perception_to_bounding_box[obj].center.data

    def get_objects_positions(self) -> PositionsMap:
        """
        Retrieves positions of all AdamObjects contained in this model.
        Returns: PositionsList

        """
        return PositionsMap(
            immutabledict(
                (object_perception.debug_handle, bounding_box.center.data)
                for object_perception, bounding_box in self.object_perception_to_bounding_box.items()
            )
        )


class BelowGroundPenalty(nn.Module):  # type: ignore
    """
    Model that penalizes boxes lying outside of the scene (i.e. below the ground plane) or off-camera)
    """

    def __init(self) -> None:  # pylint: disable=useless-super-delegation
        super().__init__()

    def forward(  # type: ignore
        self, bounding_box: AxisAlignedBoundingBox
    ):  # pylint: disable=arguments-differ
        distance_above_ground = bounding_box.z_coordinate_of_lowest_corner()
        if distance_above_ground >= 0:
            return 0
        else:
            return -distance_above_ground


class WeakGravityPenalty(nn.Module):  # type: ignore
    """
    Model that penalizes boxes that are not resting on the ground.
    """

    # TODO: exempt birds from this constraint https://github.com/isi-vista/adam/issues/485

    def __init__(self) -> None:  # pylint: disable=useless-super-delegation
        super().__init__()

    def forward(  # type: ignore
        self, bounding_box: AxisAlignedBoundingBox
    ):  # pylint: disable=arguments-differ
        distance_above_ground = bounding_box.z_coordinate_of_lowest_corner()
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

    def forward(  # type: ignore
        self,
        bounding_box_1: AxisAlignedBoundingBox,
        bounding_box_2: AxisAlignedBoundingBox,
    ):  # pylint: disable=arguments-differ

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


class InRegionPenalty(nn.Module):  # type: ignore
    """ Model that penalizes boxes for not adhering to relational (distance and direction)
        constraints with other boxes -- for being outside of the Region it is supposed to occupy
    """

    def __init__(
        self,
        object_perception_to_bounding_box: Mapping[
            ObjectPerception, AxisAlignedBoundingBox
        ],
        in_region_relations: Mapping[ObjectPerception, List[Region[ObjectPerception]]],
    ) -> None:  # pylint: disable=useless-super-delegation
        super().__init__()
        self.object_perception_to_bounding_box = object_perception_to_bounding_box
        self.in_region_relations = in_region_relations

    def forward(  # type: ignore
        self,
        target_object: ObjectPerception,
        designated_region: ImmutableSet[Region[ObjectPerception]],
    ):  # pylint: disable=arguments-differ

        print(f"{target_object.debug_handle} positioned w/r/t {designated_region}")

        # return 0 if object has no relative positions to apply
        if not designated_region:
            print(f"{target_object.debug_handle} has no relative positioning constraints")
            return torch.zeros(1)

        return sum(
            self.penalty(
                self.object_perception_to_bounding_box[target_object],
                self.object_perception_to_bounding_box[region.reference_object],
                region,
            )
            # positioning w/r/t the ground is handled by other constraints
            for region in designated_region
            if region.reference_object.debug_handle != "the ground"
        )

    def penalty(
        self,
        target_box: AxisAlignedBoundingBox,
        reference_box: AxisAlignedBoundingBox,
        region: Region[ObjectPerception],
    ):
        """
        Assign a penalty for target_box if it does not comply with its relation to reference_box according to
        the degree of difference between the expected angle between the boxes and the expected distance between
        the two objects.
        Args:
            target_box: box to be penalized if positioned outside of region
            reference_box: box referred to by region
            region: region that target_box should be in

        Returns: Tensor(1,) with penalty

        """
        assert region.direction is not None
        assert region.distance is not None
        # get direction that box 1 should be in w/r/t box 2
        # TODO: allow for addressee directions
        direction_vector = self.direction_to_worldspace_vector(
            region.direction, reference_box
        )

        current_direction_from_reference_to_target = (
            target_box.center - reference_box.nearest_center_face_point(target_box.center)
        )

        angle = angle_between(
            direction_vector, current_direction_from_reference_to_target
        )
        print(f"DIRECTION VECTOR: {direction_vector}")
        print(f"REF TO TARG VECTOR: {current_direction_from_reference_to_target}")
        print(f"ANGLE (rads): {angle}")
        if not angle or torch.isnan(angle):
            angle = torch.zeros(1, dtype=torch.float)

        distance = target_box.nearest_center_face_distance_from_point(
            reference_box.center
        )

        # distal has a minimum distance away from object to qualify
        if region.distance == DISTAL:
            if distance < DISTAL_MIN_DISTANCE:
                distance_penalty = DISTAL_MIN_DISTANCE - distance
            else:
                distance_penalty = torch.zeros(1)
        # proximal has a min/max range
        elif region.distance == PROXIMAL:
            if PROXIMAL_MIN_DISTANCE <= distance <= PROXIMAL_MAX_DISTANCE:
                distance_penalty = torch.zeros(1)
            elif distance < PROXIMAL_MIN_DISTANCE:
                distance_penalty = PROXIMAL_MIN_DISTANCE - distance
            else:
                distance_penalty = distance - PROXIMAL_MAX_DISTANCE

        # exterior but in contact has a tiny epsilon of acceptable distance
        # assuming that collisions are handled elsewhere
        elif region.distance == EXTERIOR_BUT_IN_CONTACT:
            if distance > EXTERIOR_BUT_IN_CONTACT_EPS:
                distance_penalty = distance
            else:
                distance_penalty = torch.zeros(1)
        else:
            raise RuntimeError(
                "Currently unable to support Interior distances w/ positioning solver"
            )

        print(
            f"Angle penalty: {angle * ANGLE_PENALTY} + distance penalty: {distance_penalty * DISTANCE_PENALTY}"
        )
        return angle * ANGLE_PENALTY + distance_penalty * DISTANCE_PENALTY

    def direction_to_worldspace_vector(
        self,
        direction: Direction[ObjectPerception],
        direction_reference: AxisAlignedBoundingBox,
        addressee_reference: Optional[AxisAlignedBoundingBox] = None,
    ) -> torch.Tensor:
        """
        Convert a direction to a (3,) tensor to represent the direction concretely in world-space.
        Args:
            direction: Direction object
            direction_reference: AABB corresponding to the object referenced by the direction parameter
            addressee_reference: AABB corresponding the an addressee referenced by the direction parameter

        Returns: (3,) Tensor. A vector describing a direction in world-space.

        """
        # special case: gravity
        if direction == GRAVITATIONAL_UP:
            return torch.tensor(  # pylint: disable=not-callable
                [0, 0, 1], dtype=torch.float
            )
        elif direction == GRAVITATIONAL_DOWN:
            return torch.tensor(  # pylint: disable=not-callable
                [0, 0, -1], dtype=torch.float
            )

        # horizontal axes mapped to world axes (as one of many possible visualizations)
        # We make the executive decision to map a horizontal relationship onto the X axis,
        # the better to view from a static camera position.
        # TODO: change to reflect distance from box extents https://github.com/isi-vista/adam/issues/496
        if isinstance(direction.relative_to_axis, HorizontalAxisOfObject):
            if direction.positive:
                return torch.tensor(  # pylint: disable=not-callable
                    [1, 0, 0], dtype=torch.float
                )

            else:
                return torch.tensor(  # pylint: disable=not-callable
                    [-1, 0, 0], dtype=torch.float
                )

        # in this case, calculate a vector facing toward or away from the addressee
        if (
            isinstance(direction.relative_to_axis, FacingAddresseeAxis)
            and addressee_reference is not None
        ):
            if direction.positive:
                # pointing toward addressee
                return addressee_reference.center - direction_reference.center
            else:
                # pointing away from addressee
                return direction_reference.center - addressee_reference.center

        raise NotImplementedError(f"direction_to_world called with {direction}")


def angle_between(vector0: torch.Tensor, vector1: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Returns angle between two vectors (tensors (3,) )
    Args:
        vector0: tensor (3,)
        vector1: tensor (3,)

    Returns: tensor (1,) with the angle (in radians) between the two vectors.
           Will return NaN if either of the inputs is zero.
    """
    if torch.nonzero(vector0).size(0) == 0 or torch.nonzero(vector1).size(0) == 0:
        return None
    unit_vector0 = vector0 / torch.norm(vector0, 2)
    unit_vector1 = vector1 / torch.norm(vector1, 2)
    return unit_vector0.dot(unit_vector1).acos()


if __name__ == "__main__":
    main()
