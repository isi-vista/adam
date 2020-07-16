from adam.curriculum.phase1_curriculum import build_gaila_phase_1_curriculum
from adam.language.language_utils import phase2_language_generator
from adam.learner import LanguageMode
from adam.visualization.make_scenes import SceneCreator
from adam.geon import CrossSection
from adam.visualization.utils import Shape, cross_section_to_geon
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    ObjectPerception,
)
from typing import Tuple


def test_scenes_creation() -> Tuple[
    Tuple[float, float, float], Tuple[float, float, float]
]:
    for i, scene_elements in enumerate(
        SceneCreator.create_scenes(
            build_gaila_phase_1_curriculum(
                None, None, phase2_language_generator(LanguageMode.ENGLISH)
            )
        )
    ):

        def test_for_each(obj: ObjectPerception) -> None:
            print(obj.debug_handle)

        def test_for_each_nested_really_nested(obj: ObjectPerception, other: str) -> str:
            print(obj.debug_handle)
            return other

        def test_cs_to_shape(cs: CrossSection) -> Shape:
            return cross_section_to_geon(cs)

        SceneCreator.graph_for_each(scene_elements.object_graph, test_for_each)
        SceneCreator.graph_for_each_top_level(
            scene_elements.object_graph, test_for_each, test_for_each_nested_really_nested
        )
        for obj in scene_elements.object_graph:
            if obj.perceived_obj.geon:
                test_cs_to_shape(obj.perceived_obj.geon.cross_section)

        point = SceneCreator.random_leaf_position()

        root_point = SceneCreator.random_root_position()

        # automated test shouldn't go through every single scene
        if i > 5:
            break
    return point, root_point
