"""This module is responsible for feeding Scenes from a Curriculum into
   a rendering system to be displayed. It manages *when* the renderer is
   operating and when other code (gathering and processing scene information)
   is executing in a serial manner.
   """
from typing import Iterable, List, Tuple, Union, DefaultDict, Optional

import random
from collections import defaultdict

from adam.curriculum.phase1_curriculum import build_gaila_phase_1_curriculum
import attr
from attr import attrs
from immutablecollections import ImmutableSet, immutableset

from adam.language.dependency import LinearizedDependencyTree

from adam.experiment import InstanceGroup
from adam.geon import CrossSection

from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
    RgbColorPerception,
    HasColor,
    HasBinaryProperty,
    ObjectPerception,
    Relation
)
from adam.ontology import OntologyNode

from adam.visualization.panda3d_interface import SituationVisualizer
from adam.visualization.utils import Shape

@attrs(slots=True)
class SceneNode:
    elt: ObjectPerception = attr.ib()
    children: List["SceneNode"] = attr.ib(factory=list)

def main() -> None:
    random.seed(2015)

    # go through curriculum scenes (fed in from where?) and output geometry types
    print("scene generation test")
    viz = SituationVisualizer()
    for i, scene in enumerate(
        SceneCreator.create_scenes(build_gaila_phase_1_curriculum())
    ):
        print(f"SCENE {i}")
        for obj, properties in scene.items():
            print(f"Object: {obj}")
            # only interested in rendering geons
            # TODO***: allow for Irregular geons to be rendered
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
            viz.add_model(
                SceneCreator.cross_section_to_geo(obj.geon.cross_section),
                SceneCreator.random_position(),
                color,
            )
        viz.run_for_seconds(2.25)
        input("Press ENTER to continue")
        viz.clear_scene()
        viz.run_for_seconds(0.25)


@attrs(frozen=True, slots=True)
class SceneCreator:
    @staticmethod
    def create_scenes(
        instance_groups: Iterable[
            InstanceGroup[
                HighLevelSemanticsSituation,
                LinearizedDependencyTree,
                DevelopmentalPrimitivePerceptionFrame,
            ]
        ],
    ):
        for (
            instance_group
        ) in instance_groups:  # each InstanceGroup a page related to a curriculum topic
            for (
                _,  # situation
                _,  # dependency_tree
                perception,
            ) in instance_group.instances():  # each instance a scene
                # scene_objects = []
                property_map: DefaultDict[
                    ObjectPerception, List[Optional[Union[RgbColorPerception, OntologyNode]]]
                ] = defaultdict(list)
                # we only care about the perception at the moment



                for frame in perception.frames:  # DevelopmentalPrimitivePerceptionFrame
                    # actions will have multiple frames - these will have to be rendered differently
                    for prop in frame.property_assertions:
                        if isinstance(prop, HasColor):
                            # append RgbColorPerception
                            property_map[prop.perceived_object].append(prop.color)
                        elif isinstance(prop, HasBinaryProperty):
                            # append OntologyNode
                            property_map[prop.perceived_object].append(
                                prop.binary_property
                            )

                    print("\n\n")
                    print(frame.relations[0])
                    print(frame.relations[0].relation_type)
                    print(frame.relations[0].first_slot)
                    print(type(frame.relations[0].second_slot))
                    print("\n\n")
                    SceneCreator.nest_objects(frame.relations)
                    print("\n\n")

                    # in the event that an object has no properties, we add it anyway
                    # in case it has a geon that can be rendered
                    for obj in frame.perceived_objects:
                        if obj not in property_map:
                            property_map[obj].append(None)

                yield property_map

    @staticmethod
    def cross_section_to_geo(cs: CrossSection) -> Shape:
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
    def nest_objects(relations: ImmutableSet[Relation["ObjectPerception"]]) -> None:
        """Given a set of Relations, return some kind of mapping of
           the objects that the relations pertain to according to the partOf() relations that they have.

           This could be an ImmutableSet(immutabledict {root_obj : [leaf_objects]}

           could be regular data structures at first and converted to immutable ones after

        """
        # would we need a placeholder root object to be removed before returning?
        d = defaultdict(list)
        for relation in relations:
            if relation.relation_type.handle == "partOf": # should be a better way to check
                d[relation.second_slot].append(relation.first_slot)

        # so now we have everything nested a single level
        print(d)

        most_to_least = sorted((k for k in d), key=lambda k: len(d[k]), reverse=True)
        print(most_to_least)

        scene_graph = []
        for key in most_to_least:
            print(f"assigning key {key}")
            print(f"current graph: {scene_graph}")
            search_node = None
            search_candidates = [node for node in scene_graph]
            print(f"search candidates: {search_candidates}")
            while len(search_candidates) > 0:
                new_prospects = []
                for candidate in search_candidates:
                    print(f"candidate: {candidate}")
                    for child in candidate.children:
                        print(f"child: {child}")
                        if child.elt == key:
                            search_node = child
                            break
                        else:
                            new_prospects.append(child)
                search_candidates = new_prospects

            if search_node is None:
                search_node = SceneNode(key)
                scene_graph.append(search_node)
            print(f"found node is {search_node}")
            # find node with key
            for nested in d[key]:
                search_node.children.append(SceneNode(nested))

        return scene_graph


    @staticmethod
    def random_position() -> Tuple[float, float, float]:
        """Placeholder implementation for turning the relative position
        of a crossSection into a 3D coordinate. (z is up)"""
        x: float = random.uniform(-7.0, 7.0)
        y: float = random.uniform(-5.0, 5.0)
        z: float = random.uniform(0.0, 4.0)
        return x, y, z


if __name__ == "__main__":
    main()
