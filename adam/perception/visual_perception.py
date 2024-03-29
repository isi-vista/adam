import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence, Union, List, Mapping, Dict, Optional, Any, Tuple

import yaml
from attr import attrs, attrib
from attr.validators import (
    instance_of,
    deep_iterable,
    deep_mapping,
    optional,
)
from immutablecollections import ImmutableSet, ImmutableDict
from immutablecollections.converter_utils import (
    _to_immutableset,
    _to_immutabledict,
    immutabledict,
    immutableset,
)

from adam.math_3d import Point
from adam.ontology import OntologyNode
from adam.ontology.during import DuringAction
from adam.ontology.phase1_ontology import WHITE, BLACK, RED, GREEN, BLUE
from adam.perception import (
    PerceptualRepresentationFrame,
    PerceptualRepresentation,
    PerceptionT,
    _PerceptionT2,
    ObjectPerception,
)
from adam.perception.perception_graph_nodes import (
    GraphNode,
    RgbColorNode,
    CategoricalNode,
    ContinuousNode,
    ObjectStroke,
    StrokeGNNRecognitionNode,
)

CATEGORY_PROPERTY_KEYS: List[str] = ["texture"]
CONTINUOUS_PROPERTY_KEYS: List[str] = []
STROKE_PROPERTY_KEYS: List[str] = ["stroke_mean_x", "stroke_mean_y", "stroke_std"]
RELATIVE_DISTANCE_PROPERTY_KEYS: List[str] = [
    "x_offset",
    "y_offset",
    "euclidean_distance",
]
RELATIVE_SIZE_PROPERTY_KEYS: List[str] = ["width_greater_than", "height_greater_than"]


def object_name_to_id(object_name):
    return "".join(char for char in object_name if char.isdigit())


def to_property_dict(
    mapping: Mapping[str, Sequence[GraphNode]]
) -> ImmutableDict[str, ImmutableSet[GraphNode]]:
    """
    Convert to dict of {str: list} to immutable equivalent.
    """
    return immutabledict({k: immutableset(v) for (k, v) in mapping.items()})


@attrs(slots=True, frozen=True)
class ClusterPerception:
    """Class to hold an object cluster's perception properties."""

    cluster_id: str = attrib(validator=instance_of(str))
    viewpoint_id: int = attrib(validator=instance_of(int), converter=int)
    sub_object_id: int = attrib(
        validator=instance_of(int), converter=int
    )  # 0 means not a sub-object
    strokes: ImmutableSet[ObjectStroke] = attrib(
        validator=deep_iterable(instance_of(ObjectStroke)), converter=_to_immutableset
    )
    adjacent_strokes: ImmutableDict[
        ObjectStroke, ImmutableDict[ObjectStroke, bool]
    ] = attrib(
        validator=deep_mapping(
            instance_of(ObjectStroke),
            deep_mapping(instance_of(ObjectStroke), instance_of(bool)),
        ),
        converter=_to_immutabledict,
    )
    properties: ImmutableSet[Union[GraphNode, OntologyNode]] = attrib(
        validator=deep_iterable(instance_of((GraphNode, OntologyNode))),
        converter=_to_immutableset,
    )
    relative_properties: ImmutableDict[str, ImmutableSet[GraphNode]] = attrib(
        validator=deep_mapping(instance_of(str), deep_iterable(instance_of(GraphNode))),
        converter=to_property_dict,
    )
    centroid_x: Optional[float] = attrib(
        validator=optional(instance_of(float)), default=None
    )
    centroid_y: Optional[float] = attrib(
        validator=optional(instance_of(float)), default=None
    )
    std: Optional[float] = attrib(validator=optional(instance_of(float)), default=None)
    width: Optional[float] = attrib(validator=optional(instance_of(float)), default=None)
    height: Optional[float] = attrib(validator=optional(instance_of(float)), default=None)
    box_area: Optional[float] = attrib(
        validator=optional(instance_of(float)), default=None
    )


def color_as_category(color_properties: Sequence[int]) -> OntologyNode:
    """Convert RGB values into color categories."""
    red = color_properties[0]
    green = color_properties[1]
    blue = color_properties[2]

    if red > 240 and green > 240 and blue > 240:
        return WHITE
    if red > 128 and green < 128 and blue < 128:
        return RED
    if red < 128 and green > 128 and blue < 128:
        return GREEN
    if red < 128 and green < 128 and blue > 128:
        return BLUE
    return BLACK


@attrs(slots=True, frozen=True, repr=False)
class VisualPerceptionFrame(PerceptualRepresentationFrame):
    """
    A static snapshot of a visually processed representation of an image.
    This is the default perceptual representation for phase 3 phase of the ADAM project.
    """

    clusters: Sequence[ClusterPerception] = attrib(
        validator=deep_iterable(instance_of(ClusterPerception))
    )
    touching: ImmutableSet[Tuple[str, str]] = attrib(
        validator=deep_iterable(
            member_validator=deep_iterable(
                member_validator=instance_of(str),
            )
        ),
        converter=_to_immutableset,
    )

    @staticmethod
    def from_mapping(
        perception_mapping: Mapping[str, Any], *, color_is_rgb: bool = False
    ) -> "VisualPerceptionFrame":
        clusters = []

        for cluster_map in perception_mapping["objects"]:
            color_property = cluster_map["color"]
            strokes_map = cluster_map["stroke_graph"]
            strokes = [
                ObjectStroke(
                    normalized_coordinates=[
                        Point(x, y, 0) for x, y in zip(stroke[0], stroke[1])
                    ]
                )
                for stroke in strokes_map["strokes_normalized_coordinates"]
            ]

            adjacency_matrix: Dict[
                ObjectStroke, ImmutableDict[ObjectStroke, bool]
            ] = dict()
            for stroke_id, column in enumerate(strokes_map["adjacency_matrix"]):
                adjacency_matrix[strokes[stroke_id]] = immutabledict(
                    (strokes[stroke_id_2], bool(val))
                    for stroke_id_2, val in enumerate(column)
                )

            properties: List[Union[GraphNode, OntologyNode]] = [
                RgbColorNode(
                    red=int(color_property[0]),
                    green=int(color_property[1]),
                    blue=int(color_property[2]),
                    weight=1.0,
                )
                if color_is_rgb
                else color_as_category(color_property)
            ]
            properties.extend(
                CategoricalNode(label=entry, value=cluster_map[entry], weight=1.0)
                for entry in CATEGORY_PROPERTY_KEYS
                if cluster_map[entry]
            )
            properties.extend(
                ContinuousNode(label=entry, value=cluster_map[entry], weight=1.0)
                for entry in CONTINUOUS_PROPERTY_KEYS
                if cluster_map[entry]
            )
            # properties.extend(
            #     ContinuousNode(
            #         label=f"stroke-{entry}", value=strokes_map[entry], weight=1.0
            #     )
            #     for entry in STROKE_PROPERTY_KEYS
            #     if strokes_map[entry]
            # )
            if "concept_name" in strokes_map:
                properties.append(
                    StrokeGNNRecognitionNode(
                        object_recognized=strokes_map["concept_name"],
                        confidence=strokes_map["confidence_score"],
                        weight=strokes_map["confidence_score"],
                    )
                )
            if "concept_names" in strokes_map:
                for concept_name in strokes_map["concept_names"]:
                    properties.append(
                        StrokeGNNRecognitionNode(
                            object_recognized=concept_name,
                            confidence=strokes_map["confidence_score"],
                            weight=strokes_map["confidence_score"],
                        )
                    )

            relative_properties: Dict[str, List[GraphNode]] = defaultdict(list)
            if cluster_map["relative_distance"] is not None:
                for other_object_name, relative_distance in cluster_map[
                    "relative_distance"
                ].items():
                    other_object_id = object_name_to_id(other_object_name)
                    for key in RELATIVE_DISTANCE_PROPERTY_KEYS:
                        value = relative_distance[key]
                        relative_properties[other_object_id].append(
                            ContinuousNode(label=key, value=value, weight=1.0)
                        )
            if cluster_map["relative_size"] is not None:
                for other_object_name, relative_size in cluster_map[
                    "relative_size"
                ].items():
                    other_object_id = object_name_to_id(other_object_name)
                    for key in RELATIVE_SIZE_PROPERTY_KEYS:
                        # Have to convert value to `str` to fit CategoricalNode
                        value = relative_size[key]
                        relative_properties[other_object_id].append(
                            CategoricalNode(label=key, value=str(value), weight=1.0)
                        )
            relative_properties = dict(relative_properties)

            clusters.append(
                ClusterPerception(
                    cluster_id=object_name_to_id(cluster_map["object_name"]),
                    viewpoint_id=cluster_map["viewpoint_id"],
                    sub_object_id=cluster_map.get("subobject_id", 0),
                    strokes=strokes,
                    adjacent_strokes=adjacency_matrix,
                    relative_properties=relative_properties,
                    properties=properties,
                    centroid_x=strokes_map["stroke_mean_x"],
                    centroid_y=strokes_map["stroke_mean_y"],
                    std=strokes_map["stroke_std"],
                    width=cluster_map["size"]["width"],
                    height=cluster_map["size"]["height"],
                    box_area=cluster_map["size"]["box_area"],
                )
            )

        touching: List[Tuple[str, str]] = []
        for object1_name, object2_name in perception_mapping["touching"]:
            object1_id: str = object_name_to_id(object1_name)
            object2_id: str = object_name_to_id(object2_name)
            touching.append((object1_id, object2_id))

        return VisualPerceptionFrame(clusters=clusters, touching=touching)

    @staticmethod
    def from_json_str(
        json_str: str, *, color_is_rgb: bool = False
    ) -> "VisualPerceptionFrame":
        json_perception = json.loads(json_str)
        return VisualPerceptionFrame.from_mapping(
            json_perception, color_is_rgb=color_is_rgb
        )

    @staticmethod
    def from_json(
        json_path: Path, *, color_is_rgb: bool = False
    ) -> "VisualPerceptionFrame":
        with open(json_path, encoding="utf-8") as json_file:
            json_perception = json.load(json_file)

        return VisualPerceptionFrame.from_mapping(
            json_perception, color_is_rgb=color_is_rgb
        )

    @staticmethod
    def from_yaml_str(
        yaml_str: str, *, color_is_rgb: bool = False
    ) -> "VisualPerceptionFrame":
        return VisualPerceptionFrame.from_mapping(
            yaml.safe_load(yaml_str), color_is_rgb=color_is_rgb
        )

    @staticmethod
    def from_yaml(
        yaml_path: Path, *, color_is_rgb: bool = False
    ) -> "VisualPerceptionFrame":
        with open(yaml_path, encoding="utf-8") as yaml_file:
            yaml_perception = yaml.safe_load(yaml_file)

        return VisualPerceptionFrame.from_mapping(
            yaml_perception, color_is_rgb=color_is_rgb
        )


# A new class is used as I know I'll need to handle dynamic scenes differently in the future than we did in Phases 1-2
# and I don't want to have to fix the class names in the future.
@attrs(slots=True, frozen=True)
class VisualPerceptionRepresentation(PerceptualRepresentation[PerceptionT]):
    """A class to hold a representation for Phase 3 visual perception systems."""

    frames: Tuple[PerceptionT, ...] = attrib(converter=tuple)
    """
    The frames making up the description of a situation.

    Usually for a static situation, this will be a single frame,
    but there could be two or three for complex actions.
    """
    # mypy is confused by the instance_of with a generic class
    during: Optional[DuringAction[ObjectPerception]] = attrib(  # type: ignore
        validator=optional(instance_of(DuringAction)), default=None, kw_only=True
    )
    simulated_actions_features: Optional[Mapping[str, Any]] = attrib(
        default=None,
        kw_only=True,
    )

    @staticmethod
    def single_frame(
        perception_frame: _PerceptionT2,
    ) -> "VisualPerceptionRepresentation[_PerceptionT2]":
        """
        Convenience method for generating a `PerceptualRepresentation` which is a single frame.

        Args:
            perception_frame: a `PerceptualRepresentationFrame`.

        Returns:
            A `PerceptualRepresentation` wrapping the provided frame.

        """
        return VisualPerceptionRepresentation((perception_frame,))  # type: ignore

    @staticmethod
    def multi_frame(
        frames: Sequence[_PerceptionT2], action_features: Mapping[str, Any]
    ) -> "VisualPerceptionRepresentation[_PerceptionT2]":
        """
        Convenience method for generating a `PerceptualReprsentation` which is multiple frames.

        Args:
            frames: a sequence of `PerceptualRepresentationFrame`.
            action_features: A mapping of str to feature representations

        Returns:
            A `PerceptualRepresentation` wrapping the provided frames and action features.
        """
        return VisualPerceptionRepresentation(
            frames=tuple(frames), simulated_actions_features=action_features  # type: ignore
        )
