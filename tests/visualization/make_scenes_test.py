from adam.visualization.make_scenes import SceneCreator
from adam.curriculum.phase1_curriculum import GAILA_PHASE_1_CURRICULUM
from adam.geon import CrossSection
from adam.visualization.utils import Shape
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    ObjectPerception,
)
from typing import Tuple


def test_scenes_creation() -> Tuple[float, float, float]:
    for i, (_, obj_graph) in enumerate(
        SceneCreator.create_scenes(GAILA_PHASE_1_CURRICULUM)
    ):

        def test_for_each(obj: ObjectPerception) -> None:
            print(obj.debug_handle)

        def test_for_each_nested_really_nested(obj: ObjectPerception, other: str) -> str:
            print(obj.debug_handle)
            return other

        def test_cs_to_shape(cs: CrossSection) -> Shape:
            return SceneCreator.cross_section_to_geo(cs)

        SceneCreator.graph_for_each(obj_graph, test_for_each)
        SceneCreator.graph_for_each_top_level(
            obj_graph, test_for_each, test_for_each_nested_really_nested
        )
        for obj in obj_graph:
            if obj.elt.geon:
                test_cs_to_shape(obj.elt.geon.cross_section)

        point = SceneCreator.random_position()
        # automated test shouldn't go through every single scene
        if i > 5:
            break
    return point
