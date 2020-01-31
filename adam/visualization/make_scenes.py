"""This module is responsible for feeding Scenes from a Curriculum into
   a rendering system to be displayed. It manages *when* the renderer is
   operating and when other code (gathering and processing scene information)
   is executing in a serial manner.
   """
from typing import (
    Iterable,
    List,
    Tuple,
    Union,
    DefaultDict,
    Optional,
    Callable,
    Any,
    Generator,
    Mapping,
)
from functools import partial

import random
from collections import defaultdict
import numpy as np

import logging

# currently useful for positioning multiple objects:
from adam.curriculum.phase1_curriculum import (
    _make_object_beside_object_curriculum as make_curriculum,
)


import attr
from attr import attrs
from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point
from immutablecollections import ImmutableSet, immutableset

# consider refactoring away this dependency
from panda3d.core import NodePath  # pylint: disable=no-name-in-module

from adam.language.dependency import LinearizedDependencyTree

from adam.experiment import InstanceGroup

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
from adam.relation import IN_REGION

from adam.visualization.panda3d_interface import SituationVisualizer
from adam.visualization.utils import (
    Shape,
    OBJECT_NAMES_TO_EXCLUDE,
    cross_section_to_geon,
    OBJECT_SCALE_MULTIPLIER_MAP,
)

from adam.visualization.positioning import run_model, PositionsMap
from adam.ontology.phase1_spatial_relations import Region

USAGE_MESSAGE = """make_scenes.py param_file
                \twhere param_file has the following parameters:
                \t\titerations: int: total number of iterations to run positioning model over
                \t\tsteps_before_vis: int: number of iterations of positioning model before scene is re-rendered
                \t\tseed: int: random seed for picking initial object positions
                """


@attrs(slots=True)
class SceneNode:
    """
    Node type used for creating graph structure from a Perception of a scene.
    This kind of hierarchical grouping of objects within the scene is helpful for adjusting
    the positions of the objects within the rendering engine.
    """

    name: str = attr.ib()
    perceived_obj: ObjectPerception = attr.ib()
    children: List["SceneNode"] = attr.ib(factory=list)
    parent: "SceneNode" = attr.ib(default=None)


def main(params: Parameters) -> None:
    num_iterations = params.positive_integer("iterations")
    steps_before_vis = params.positive_integer("steps_before_vis")

    random.seed(params.integer("seed"))
    np.random.seed(params.integer("seed"))

    # go through curriculum scenes and output geometry types
    print("scene generation test")
    viz = SituationVisualizer()
    model_scales = viz.get_model_scales()
    for object_type, multiplier in OBJECT_SCALE_MULTIPLIER_MAP.items():
        if object_type in model_scales:
            v3 = model_scales[object_type]
            new_v3 = (v3[0] * multiplier, v3[1] * multiplier, v3[2] * multiplier)
            model_scales[object_type] = new_v3
        else:
            model_scales[object_type] = (multiplier, multiplier, multiplier)

    for model_name, scale in model_scales.items():
        logging.info("SCALE: %s -> %s", model_name, scale.__str__())

    for i, scene_elements in enumerate(SceneCreator.create_scenes([make_curriculum()])):

        print(f"SCENE {i}")
        viz.set_title(" ".join(token for token in scene_elements.tokens))
        # for debugging purposes:
        SceneCreator.graph_for_each(scene_elements.object_graph, print_obj_names)

        # bind visualizer and properties to top level rendering function:
        bound_render_obj = partial(
            render_obj, viz, scene_elements.property_map
        )  # bind visualizer and properties to nested obj rendering function
        # render each object in graph

        SceneCreator.graph_for_each_top_level(
            scene_elements.object_graph, bound_render_obj
        )
        # for debugging purposes to view the results before positioning:
        viz.run_for_seconds(1)
        command = input(
            "Press ENTER to run the positioning system or enter name to save a screenshot\n"
            "Or type 's' for (step or for skip) to skip this scene > "
        )
        if command == "s":
            viz.clear_scene()
            viz.run_for_seconds(0.25)
            continue
        if command:
            viz.screenshot(command)

        # now that every object has been instantiated into the scene,
        # they need to be re-positioned.

        for repositioned_map in _solve_top_level_positions(
            immutableset(
                [
                    node.perceived_obj
                    for node in scene_elements.object_graph
                    if node.name not in OBJECT_NAMES_TO_EXCLUDE
                ]
            ),
            scene_elements.in_region_map,
            model_scales,
            iterations=num_iterations,
            yield_steps=steps_before_vis,
        ):
            print(f"repositioned values: {repositioned_map}")
            viz.set_positions(repositioned_map)

            # the visualizer seems to need about a second to render an update
            viz.run_for_seconds(1)
            # viz.print_scene_graph()
        viz.run_for_seconds(1)

        screenshot_name = input(
            "Press ENTER to continue to the next scene, or the name of a file to save a screenshot to: "
        )
        if screenshot_name:
            viz.screenshot(screenshot_name)
        viz.clear_scene()
        viz.run_for_seconds(0.25)


def render_obj(
    renderer: SituationVisualizer,
    properties: DefaultDict[
        ObjectPerception, List[Optional[Union[RgbColorPerception, OntologyNode]]]
    ],
    obj: ObjectPerception,
) -> NodePath:
    """
    Used to render a top-level object (has no parent)
    Even if the object has no geon, it can be 'rendered' as a dummy node to structure other objects
    Args:
        renderer: rendering engine to render this object with
        properties: set of properties (colors, etc) associated with obj
        obj: the object to be rendered
        omit_irregular: flag for ignoring irregular geons

    Returns: a Panda3d NodePath: the path within the rendering engine's scene graph to the object/node
             rendered by calling this function.

    """
    return render_obj_nested(renderer, properties, obj, None)


def render_obj_nested(
    renderer: SituationVisualizer,
    properties: DefaultDict[
        ObjectPerception, List[Optional[Union[RgbColorPerception, OntologyNode]]]
    ],
    obj: ObjectPerception,
    parent: Optional[NodePath],
) -> NodePath:
    """
    Used to render a nested object (has a parent)
    Even if the object has no geon, it can be 'rendered' as a dummy node to structure other objects
    Args:
        renderer: rendering engine to render this object with
        properties: set of properties (colors, etc) associated with obj
        obj: the object to be rendered
        parent: the parent of the object to be rendered

    Returns: a Panda3d NodePath: the path within the rendering engine's scene graph to the object/node
             rendered by calling this function.

    """
    # if model is represented as a single piece of geometry, get its scale multiplier
    scale_multiplier = 1.0
    object_type = obj.debug_handle.split("_")[0]
    if object_type in OBJECT_SCALE_MULTIPLIER_MAP:
        scale_multiplier = OBJECT_SCALE_MULTIPLIER_MAP[object_type]

    if obj.geon is None:
        if parent is None:
            pos = SceneCreator.random_root_position()
        else:
            pos = SceneCreator.random_leaf_position()
        return renderer.add_dummy_node(
            obj.debug_handle, pos, parent, scale_multiplier=scale_multiplier
        )
    shape = cross_section_to_geon(obj.geon.cross_section)
    # TODO***: allow for Irregular geons to be rendered
    if shape == Shape.IRREGULAR:
        logging.warning("Irregular shape for %s could not be rendered", obj.debug_handle)
    color = None
    for prop in properties[obj]:
        if isinstance(prop, RgbColorPerception):
            color = prop
    if parent is None:
        pos = SceneCreator.random_root_position()
    else:
        pos = SceneCreator.random_leaf_position()
    if shape == Shape.IRREGULAR:
        return renderer.add_dummy_node(name=obj.debug_handle, position=pos, parent=parent)
    return renderer.add_model(
        shape,
        name=obj.debug_handle,
        position=pos,
        color=color,
        parent=parent,
        scale_multiplier=scale_multiplier,
    )


def print_obj_names(obj: ObjectPerception) -> None:
    """
    Debug function to print the name of an ObjectPerception (called while walking a scene graph)
    """
    if obj.geon is not None:
        print(obj.debug_handle + " (has geon)")
    else:
        print(obj.debug_handle)


@attrs(frozen=True, slots=True)
class SceneElements:
    """ Convenience wrapper for the various sub-objects returned by SceneCreator """

    # Objects -> their properties
    property_map: DefaultDict[
        ObjectPerception, List[Optional[Union[RgbColorPerception, OntologyNode]]]
    ] = attr.ib()
    # objects -> in_region relations
    in_region_map: DefaultDict[
        ObjectPerception, List[Region[ObjectPerception]]
    ] = attr.ib()
    # scene nodes arranged in a tree structure
    object_graph: List[SceneNode] = attr.ib()
    # utterance related to the scene
    tokens: Tuple[str, ...] = attr.ib()


@attrs(frozen=True, slots=True)
class SceneCreator:
    """
    Static class for creating a graph structure out of ObjectPerceptions.
    Nests sub-objects (e.g. Person[arm_0[arm_segment_0, arm_segment_1, hand_0]], arm_1[...], ...]
    """

    @staticmethod
    def create_scenes(
        instance_groups: Iterable[
            InstanceGroup[
                HighLevelSemanticsSituation,
                LinearizedDependencyTree,
                DevelopmentalPrimitivePerceptionFrame,
            ]
        ],
    ) -> Generator[SceneElements, None, None]:
        for (
            instance_group
        ) in instance_groups:  # each InstanceGroup a page related to a curriculum topic
            for (
                _,  # situation
                dependency_tree,  # dependency_tree
                perception,
            ) in instance_group.instances():  # each instance is a scene
                # scene_objects = []
                property_map: DefaultDict[
                    ObjectPerception,
                    List[Optional[Union[RgbColorPerception, OntologyNode]]],
                ] = defaultdict(list)
                # we only care about the perception at the moment

                for frame in perception.frames:  # DevelopmentalPrimitivePerceptionFrame

                    in_region_map: DefaultDict[
                        ObjectPerception, List[Region[ObjectPerception]]
                    ] = defaultdict(list)

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

                    # copy over the relations that each ObjectPerception has
                    # primarily interested in in-region relations
                    for relation in frame.relations:
                        if relation.relation_type == IN_REGION and isinstance(
                            relation.second_slot, Region
                        ):
                            in_region_map[relation.first_slot].append(
                                relation.second_slot
                            )

                    nested_objects = SceneCreator._nest_objects(
                        frame.perceived_objects, frame.relations
                    )

                    # in the event that an object has no properties, we add it anyway
                    # in case it has a geon that can be rendered
                    for obj in frame.perceived_objects:
                        if obj not in property_map:
                            property_map[obj].append(None)

                yield SceneElements(
                    property_map,
                    in_region_map,
                    nested_objects,
                    dependency_tree.as_token_sequence(),
                )

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
        # create a structure that can be nested arbitrarily deeply:
        for key in most_to_least:
            search_node = None
            search_candidates = [node for node in scene_graph]
            while search_candidates:
                new_prospects = []
                for candidate in search_candidates:
                    for child in candidate.children:
                        if child.perceived_obj == key:
                            search_node = child
                            break
                        else:
                            new_prospects.append(child)
                search_candidates = new_prospects

            if search_node is None:
                search_node = SceneNode(key.debug_handle, key)
                scene_graph.append(search_node)
            # find node with key
            for nested in d[key]:
                # check if this object's children are already in the scene graph,
                # if so, nest them under this object instead of the top level
                existing_node = None
                for node in scene_graph:
                    if node.name == nested.debug_handle:
                        existing_node = node
                if existing_node is None:
                    search_node.children.append(
                        SceneNode(nested.debug_handle, nested, parent=search_node)
                    )
                else:
                    search_node.children.append(existing_node)
                    scene_graph.remove(existing_node)

        return scene_graph

    @staticmethod
    def graph_for_each(
        graph: List[SceneNode], fn: Callable[["ObjectPerception"], None]
    ) -> None:
        """Apply some function to each leaf node of the scene graph"""
        nodes = [node for node in graph]
        while nodes:
            recurse: List[SceneNode] = []
            for node in nodes:
                if not node.children:
                    fn(node.perceived_obj)
                else:
                    recurse += node.children
            nodes = recurse

    @staticmethod
    def graph_for_each_top_level(
        graph: List[SceneNode], top_fn: Callable[[ObjectPerception], Any]
    ) -> None:
        """Apply some function only to root elements of graph. """
        for top_level in graph:
            # special cases not rendered here:
            if top_level.perceived_obj.debug_handle in OBJECT_NAMES_TO_EXCLUDE:
                continue
            top_fn(top_level.perceived_obj)

    @staticmethod
    def random_root_position() -> Tuple[float, float, float]:
        """Placeholder implementation for turning the relative position
        of a crossSection into a 3D coordinate. (z is up)"""
        x: float = random.uniform(-10, 10)
        y: float = random.uniform(-7.0, 4.0)
        z: float = random.uniform(1.0, 5.0)
        return x, y, z

    @staticmethod
    def random_leaf_position() -> Tuple[float, float, float]:
        """Placeholder starting position for leaf objects (whose position value
        is relative to their parent)."""
        x: float = random.uniform(-1.0, 1.0)
        y: float = random.uniform(-1.0, 1.0)
        z: float = random.uniform(-1.0, 1.0)
        return x, y, z


# TODO: scale of top-level bounding boxes is weird because it needs to encompass all sub-objects
def _solve_top_level_positions(
    top_level_objects: ImmutableSet[ObjectPerception],
    in_region_map: DefaultDict[ObjectPerception, List[Region[ObjectPerception]]],
    model_scales: Mapping[str, Tuple[float, float, float]],
    iterations: int = 200,
    yield_steps: Optional[int] = None,
) -> Iterable[PositionsMap]:
    """
        Solves for positions of top-level objects.
    Args:
        top_level_objects: set of top level objects (ObjectPerception)s
        in_region_map: map of all ObjectPerception -> Region (in-region relations) for top level and sub-objects
        iterations: number of iterations to run model for
        yield_steps: number of iterations that must pass before a new set of positions is yielded/returned

    Returns: List of (3,) tensors, describing the updated positions of the top level objects, corresponding
             in terms of indices with parent_positions
    """

    return run_model(
        top_level_objects,
        in_region_map,
        model_scales,
        num_iterations=iterations,
        yield_steps=yield_steps,
    )


if __name__ == "__main__":
    parameters_only_entry_point(main, usage_message=USAGE_MESSAGE)
