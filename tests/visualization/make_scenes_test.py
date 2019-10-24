from adam.visualization.make_scenes import SceneCreator
from adam.curriculum.phase1_curriculum import GAILA_PHASE_1_CURRICULUM
from typing import Tuple


def test_scenes_creation() -> Tuple[float, float, float]:
    for i, (_, obj_graph) in enumerate(
        SceneCreator.create_scenes(GAILA_PHASE_1_CURRICULUM)
    ):
        SceneCreator.graph_for_each(obj_graph, lambda g: None)

        point = SceneCreator.random_position()
        # automated test shouldn't go through every single scene
        if i > 5:
            break
    return point
