from adam.curriculum.phase1_curriculum import build_gaila_phase_1_curriculum
from adam.visualization.make_scenes import SceneCreator

from typing import Tuple


def test_scenes_creation() -> Tuple[float, float, float]:
    for i, (_, obj_graph) in enumerate(
        SceneCreator.create_scenes(build_gaila_phase_1_curriculum())
    ):
        SceneCreator.graph_for_each(obj_graph, lambda g: None)

        point = SceneCreator.random_position()

        # automated test shouldn't go through every single scene
        if i > 5:
            break
    return point
