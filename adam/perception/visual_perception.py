import json
from pathlib import Path
from typing import Sequence, Union, List, Mapping, Dict, Optional, Any

from attr import attrs, attrib
from attr.validators import instance_of, deep_iterable, deep_mapping, optional
from immutablecollections import ImmutableSet, ImmutableDict, immutabledict
from immutablecollections.converter_utils import _to_immutableset, _to_immutabledict

from adam.math_3d import Point
from adam.ontology import OntologyNode
from adam.ontology.phase1_ontology import WHITE, BLACK, RED, GREEN, BLUE
from adam.perception import PerceptualRepresentationFrame
from adam.perception.perception_graph_nodes import (
    GraphNode,
    RgbColorNode,
    CategoricalNode,
    ContinuousNode,
    ObjectStroke,
)

CATEGORY_PROPERTY_KEYS: List[str] = ["texture"]
CONTINUOUS_PROPERTY_KEYS: List[str] = []
STROKE_PROPERTY_KEYS: List[str] = ["mean", "std"]


@attrs(slots=True, frozen=True)
class ClusterPerception:
    """Class to hold an object cluster's perception properties."""

    cluster_id: int = attrib(validator=instance_of(int))
    viewpoint_id: int = attrib(validator=instance_of(int))
    sub_object_id: int = attrib(validator=instance_of(int))  # 0 means not a sub-object
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
    centroid_x: Optional[float] = attrib(
        validator=optional(instance_of(float)), default=None
    )
    centroid_y: Optional[float] = attrib(
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

    @staticmethod
    def from_mapping(
        json_perception: Mapping[str, Any], *, color_is_rgb: bool = False
    ) -> "VisualPerceptionFrame":
        clusters = []
        for cluster_map in json_perception["objects"]:
            color_property = cluster_map["color"]
            strokes_map = cluster_map["stroke graph"]
            strokes = [
                ObjectStroke(normalized_coordinates=[Point(x, y, 0) for x, y in stroke])
                for stroke in strokes_map["strokes normalized coordinates"]
            ]

            adjacency_matrix: Dict[
                ObjectStroke, ImmutableDict[ObjectStroke, bool]
            ] = dict()
            for stroke_id, column in enumerate(strokes_map["adjacency matrix"]):
                adjacency_matrix[strokes[stroke_id]] = immutabledict(
                    (strokes[stroke_id_2], bool(val))
                    for stroke_id_2, val in enumerate(column)
                )

            properties: List[Union[GraphNode, OntologyNode]] = [
                RgbColorNode(
                    red=color_property[0],
                    green=color_property[1],
                    blue=color_property[2],
                    weight=1.0,
                )
                if color_is_rgb
                else color_as_category(color_property)
            ]
            properties.extend(
                CategoricalNode(value=cluster_map[entry], weight=1.0)
                for entry in CATEGORY_PROPERTY_KEYS
            )
            properties.extend(
                ContinuousNode(label=entry, value=cluster_map[entry], weight=1.0)
                for entry in CONTINUOUS_PROPERTY_KEYS
            )
            properties.extend(
                ContinuousNode(
                    label=f"stroke-{entry}", value=strokes_map[entry], weight=1.0
                )
                for entry in STROKE_PROPERTY_KEYS
            )
            clusters.append(
                ClusterPerception(
                    cluster_id=cluster_map["object name"],
                    viewpoint_id=cluster_map["viewpoint id"],
                    sub_object_id=cluster_map["sub-object id"],
                    strokes=strokes,
                    adjacent_strokes=adjacency_matrix,
                    properties=properties,
                )
            )

        return VisualPerceptionFrame(clusters=clusters)

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
