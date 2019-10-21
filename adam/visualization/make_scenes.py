from typing import (
    AbstractSet,
    Any,
    Callable,
    Iterable,
    List,
    Tuple,
    TypeVar,
    Union,
    Optional,
)

import random
from collections import defaultdict
from attr import attrib, attrs
from attr.validators import instance_of

from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

from immutablecollections import ImmutableListMultiDict, immutablelistmultidict

from adam.language.dependency import LinearizedDependencyTree

from adam.curriculum.phase1_curriculum import GAILA_PHASE_1_CURRICULUM
from adam.experiment import InstanceGroup
from adam.geon import CrossSection

from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
    RgbColorPerception,
)

from adam.visualization.panda3d_interface import SituationVisualizer
from adam.visualization.utils import Shape

USAGE_MESSAGE = """ """


def main() -> None:
    random.seed(2015)

    # go through curriculum scenes (fed in from where?) and output geometry types
    print("scene generation test")
    sc = SceneCreator()
    viz = SituationVisualizer()
    for i, scene in enumerate(sc.create_scenes(GAILA_PHASE_1_CURRICULUM)):
        print(f"SCENE {i}")
        for obj, properties in scene.items():
            if obj.geon is None:  # only interested in rendering geons
                continue
            # print(obj.geon.cross_section_size)
            # print(obj.axes)
            obj_added = False  # TODO: clean up this bit of control-flow mess
            for prop in properties:
                if isinstance(prop, RgbColorPerception):
                    viz.add_model(
                        sc.cross_section_to_geo(obj.geon.cross_section),
                        SceneCreator.random_position(),
                        prop,
                    )
                    obj_added = True
            if not obj_added:
                viz.add_model(
                    sc.cross_section_to_geo(obj.geon.cross_section),
                    SceneCreator.random_position(),
                )
        viz.run_for_seconds(2.25)
        input("Press ENTER to continue")
        viz.clear_scene()
        viz.run_for_seconds(0.25)


@attrs(frozen=True, slots=True)
class SceneCreator:
    def create_scenes(
        self,
        instance_groups: Iterable[
            InstanceGroup[
                HighLevelSemanticsSituation,
                LinearizedDependencyTree,
                DevelopmentalPrimitivePerceptionFrame,
            ]
        ],
    ):
        for i, instance_group in enumerate(
            instance_groups
        ):  # each InstanceGroup a page related to a curriculum topic
            for (
                situation,
                dependency_tree,
                perception,
            ) in instance_group.instances():  # each instance a scene
                # scene_objects = []
                property_map = defaultdict(list)
                # we only care about the perception at the moment

                for frame in perception.frames:  # DevelopmentalPrimitivePerceptionFrame
                    # print(frame.perceived_objects)
                    # for obj_percept in frame.perceived_objects:
                    #     if obj_percept.geon is None:
                    #         continue
                    #     scene_objects.append(
                    #         self.cross_section_to_geo(obj_percept.geon.cross_section)
                    #     )
                    for property in frame.property_assertions:
                        if hasattr(property, "color"):
                            # print((property.perceived_object.geon, property.color))
                            property_map[property.perceived_object].append(property.color)
                        else:
                            property_map[property.perceived_object].append(
                                property.binary_property
                            )
                print(property_map)

                yield property_map

                # yield scene_objects

    def cross_section_to_geo(self, cs: CrossSection) -> Shape:
        if cs.has_rotational_symmetry and cs.has_reflective_symmetry and cs.curved:
            return Shape("CIRCULAR")
        elif cs.has_rotational_symmetry and cs.has_reflective_symmetry and not cs.curved:
            return Shape("SQUARE")
        elif not cs.has_rotational_symmetry and cs.has_reflective_symmetry and cs.curved:
            return Shape("OVALISH")
        elif (
            not cs.has_rotational_symmetry
            and cs.has_reflective_symmetry
            and not cs.curved
        ):
            return Shape("RECTANGULAR")
        elif (
            not cs.has_rotational_symmetry
            and not cs.has_reflective_symmetry
            and not cs.curved
        ):
            return Shape("IRREGULAR")
        else:
            raise ValueError("Unknown Geon composition")

    @staticmethod
    def random_position() -> Tuple[float, float, float]:
        """Placeholder implementation for turning the relative position
        of a crossSection into a 3D coordinate. (z is up)"""
        x: float = random.uniform(-7.0, 7.0)
        y: float = random.uniform(-5.0, 5.0)
        z: float = random.uniform(0.0, 4.0)
        return (x, y, z)


if __name__ == "__main__":
    # parameters_only_entry_point(main, usage_message=USAGE_MESSAGE)
    main()
