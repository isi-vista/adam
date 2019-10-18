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

from attr import attrib, attrs
from attr.validators import instance_of

from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

from adam.language.dependency import LinearizedDependencyTree

from adam.curriculum.phase1_curriculum import GAILA_PHASE_1_CURRICULUM
from adam.experiment import InstanceGroup
from adam.geon import CrossSection

from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)

from adam.visualization.panda3d_interface import SituationVisualizer
from adam.visualization.utils import Shape

USAGE_MESSAGE = """ """


def main() -> None:
    # go through curriculum scenes (fed in from where?) and output geometry types
    print("scene generation test")
    sc = SceneCreator()
    viz = SituationVisualizer()
    for i, scene in enumerate(sc.create_scenes(GAILA_PHASE_1_CURRICULUM)):
        print(f"SCENE {i}")
        for obj in scene:
            print(obj)
            viz.add_model(obj)
        viz.run_for_seconds(5.0)
        viz.clear_scene()
        viz.run_for_seconds(0.25)
        if i > 10:
            break



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
                scene_objects = []
                # we only care about the perception at the moment
                for frame in perception.frames:  # DevelopmentalPrimitivePerceptionFrame
                    for obj_percept in frame.perceived_objects:
                        if obj_percept.geon is None:
                            continue
                        scene_objects.append(
                            self._cross_section_to_geo(obj_percept.geon.cross_section)
                        )
                yield scene_objects

    def _cross_section_to_geo(self, cs: CrossSection) -> Shape:
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


if __name__ == "__main__":
    # parameters_only_entry_point(main, usage_message=USAGE_MESSAGE)
    main()
