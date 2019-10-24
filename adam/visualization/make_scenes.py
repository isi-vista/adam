"""This module is responsible for feeding Scenes from a Curriculum into
   a rendering system to be displayed. It manages *when* the renderer is
   operating and when other code (gathering and processing scene information)
   is executing in a serial manner.
   """
from typing import Iterable, List, Tuple, Union, DefaultDict, Optional, Callable
from functools import partial

import random
from collections import defaultdict
import attr
from attr import attrs
from immutablecollections import ImmutableSet

from adam.language.dependency import LinearizedDependencyTree

from adam.curriculum.phase1_curriculum import GAILA_PHASE_1_CURRICULUM
from adam.experiment import InstanceGroup
from adam.geon import CrossSection

from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
    RgbColorPerception,
    HasColor,
    HasBinaryProperty,
    ObjectPerception,
    Relation,
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

    # go through curriculum scenes and output geometry types
    print("scene generation test")
    viz = SituationVisualizer()
    for i, (property_map, obj_graph) in enumerate(
        SceneCreator.create_scenes(GAILA_PHASE_1_CURRICULUM)
    ):
        print(f"SCENE {i}")
        # bind visualizer and properties to render function:
        bound_render_obj = partial(
            render_obj, viz, property_map
        )  # bind renderer to function
        # render each object in graph
        SceneCreator.graph_for_each(obj_graph, bound_render_obj)
        viz.run_for_seconds(2.25)
        input("Press ENTER to continue")
        viz.clear_scene()
        viz.run_for_seconds(0.25)


def render_obj(
    renderer: SituationVisualizer,
    properties: DefaultDict[
        ObjectPerception, List[Optional[Union[RgbColorPerception, OntologyNode]]]
    ],
    obj: ObjectPerception,
) -> None:
    if obj.geon is None:
        return
    shape = SceneCreator.cross_section_to_geo(obj.geon.cross_section)
    # TODO***: allow for Irregular geons to be rendered
    if shape == Shape.IRREGULAR:
        return
    color = None
    for prop in properties[obj]:
        if isinstance(prop, RgbColorPerception):
            color = prop
    renderer.add_model(shape, SceneCreator.random_position(), color)


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
                    ObjectPerception,
                    List[Optional[Union[RgbColorPerception, OntologyNode]]],
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

                    nested_objects = SceneCreator._nest_objects(
                        frame.perceived_objects, frame.relations
                    )

                    # in the event that an object has no properties, we add it anyway
                    # in case it has a geon that can be rendered
                    for obj in frame.perceived_objects:
                        if obj not in property_map:
                            property_map[obj].append(None)

                yield property_map, nested_objects

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
    def _nest_objects(
        perceived_objects: ImmutableSet[ObjectPerception],
        relations: ImmutableSet[Relation["ObjectPerception"]],
    ) -> List[SceneNode]:
        """
        Given a set of objects and corresponding relations, return a pseudo-tree structure
        that has all objects with a partOf relationship between one another nested
        accordingly, with all singular objects residing at the top level.
        (If it was really a tree, there would only be one root element instead of a list).
        """
        d: DefaultDict[ObjectPerception, List["ObjectPerception"]] = defaultdict(list)
        for relation in relations:
            if relation.relation_type.handle == "partOf" and isinstance(
                relation.second_slot, ObjectPerception
            ):  # should be a better way to check
                d[relation.second_slot].append(relation.first_slot)

        # add all additional objects not covered with partOf relations
        for obj in perceived_objects:
            if obj not in d:
                # just create default empty list by accessing dict at key
                d[obj]  # pylint: disable=pointless-statement

        # so now we have everything nested a single level

        # probably not strictly necessary, but the thing with the most
        # references in partOf is probably higher up in the tree
        most_to_least = sorted((k for k in d), key=lambda k: len(d[k]), reverse=True)
        # scene graph is a nested structure where multiple items can be at the top level
        scene_graph: List[SceneNode] = []
        for key in most_to_least:
            search_node = None
            search_candidates = [node for node in scene_graph]
            while search_candidates:
                new_prospects = []
                for candidate in search_candidates:
                    for child in candidate.children:
                        if child.elt == key:
                            search_node = child
                            break
                        else:
                            new_prospects.append(child)
                search_candidates = new_prospects

            if search_node is None:
                search_node = SceneNode(key)
                scene_graph.append(search_node)
            # find node with key
            for nested in d[key]:
                search_node.children.append(SceneNode(nested))

        return scene_graph

    @staticmethod
    def graph_for_each(
        graph: List[SceneNode], fn: Callable[["ObjectPerception"], None]
    ) -> None:
        """Apply some function to each node of the scene graph"""
        nodes = graph
        while nodes:
            recurse: List[SceneNode] = []
            for node in nodes:
                if not node.children:
                    fn(node.elt)
                else:
                    recurse += node.children
            nodes = recurse

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
