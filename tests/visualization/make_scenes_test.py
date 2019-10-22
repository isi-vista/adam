from adam.visualization.make_scenes import (
    SceneCreator,
)
from adam.curriculum.phase1_curriculum import GAILA_PHASE_1_CURRICULUM
from adam.visualization.utils import Shape
from adam.perception.developmental_primitive_perception import RgbColorPerception


def test_scenes_creation() -> None:
    for i, scene in enumerate(SceneCreator.create_scenes(GAILA_PHASE_1_CURRICULUM)):
        for obj, properties in scene.items():
            # only interested in rendering geons
            if (
                obj.geon is None
                or SceneCreator.cross_section_to_geo(obj.geon.cross_section)
                == Shape.IRREGULAR
            ):
                continue
            color = None
            for prop in properties:
                if isinstance(prop, RgbColorPerception):
                    color = prop

        # automated test shouldn't go through every single scene
        if i > 5:
            break