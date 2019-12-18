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
    Dict,
)
from functools import partial

import random
from collections import defaultdict

# currently useful for positioning multiple objects:
from adam.curriculum.phase1_curriculum import _make_multiple_objects_curriculum
import attr
from attr import attrs
from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point
from immutablecollections import ImmutableSet

# consider refactoring away this dependency
from panda3d.core import NodePath  # pylint: disable=no-name-in-module

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
    Relation,
)
from adam.ontology import OntologyNode

from adam.visualization.panda3d_interface import SituationVisualizer
from adam.visualization.utils import Shape

from adam.visualization.positioning import run_model, AdamObject, PositionsMap

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

    # go through curriculum scenes and output geometry types
    print("scene generation test")
    viz = SituationVisualizer()
    for i, (property_map, obj_graph) in enumerate(
        SceneCreator.create_scenes([_make_multiple_objects_curriculum()])
    ):
        # debug: skip the first few scenes with people in them
        if i < 3:
            continue
        print(f"SCENE {i}")
        # for debugging purposes:
        SceneCreator.graph_for_each(obj_graph, print_obj_names)

        # bind visualizer and properties to top level rendering function:
        bound_render_obj = partial(
            render_obj, viz, property_map
        )  # bind visualizer and properties to nested obj rendering function
        bound_render_nested_obj = partial(render_obj_nested, viz, property_map)
        # render each object in graph
        SceneCreator.graph_for_each_top_level(
            obj_graph, bound_render_obj, bound_render_nested_obj
        )

        # for debugging purposes to view the results before positioning:
        viz.run_for_seconds(0.5)
        input("Press ENTER to run the positioning system")

        # now that every object has been instantiated into the scene,
        # they need to be re-positioned.

        for repositioned_map in _solve_top_level_positions(
            viz.top_level_positions(),
            iterations=num_iterations,
            yield_steps=steps_before_vis,
        ):
            print(f"repositioned values: {repositioned_map}")
            viz.set_positions(repositioned_map)

            # the visualizer seems to need about a second to render an update
            viz.run_for_seconds(1)
            # viz.print_scene_graph()

        input("Press ENTER to continue to the next scene")
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
    if obj.geon is None:
        if parent is None:
            pos = SceneCreator.random_root_position()
        else:
            pos = SceneCreator.random_leaf_position()
        return renderer.add_dummy_node(obj.debug_handle, pos, parent)
    shape = SceneCreator.cross_section_to_geo(obj.geon.cross_section)
    # TODO***: allow for Irregular geons to be rendered
    if shape == Shape.IRREGULAR:
        raise RuntimeError(
            "Irregular shapes (i.e. liquids) are not currently supported by the rendering system"
        )
    color = None
    for prop in properties[obj]:
        if isinstance(prop, RgbColorPerception):
            color = prop
    if parent is None:
        pos = SceneCreator.random_root_position()
    else:
        pos = SceneCreator.random_leaf_position()
    return renderer.add_model(
        shape, name=obj.debug_handle, position=pos, color=color, parent=parent
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
    ):
        for (
            instance_group
        ) in instance_groups:  # each InstanceGroup a page related to a curriculum topic
            for (
                _,  # situation
                _,  # dependency_tree
                perception,
            ) in instance_group.instances():  # each instance is a scene
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
        """
        Converts a cross section into a geon type, based on the properties of the cross section
        Args:
            cs: CrossSection to be mapped to a Geon type

        Returns: Shape: a convenience enum mapping to a file name for the to-be-rendered geometry

        """
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
                search_node.children.append(
                    SceneNode(nested.debug_handle, nested, parent=search_node)
                )

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
        graph: List[SceneNode],
        top_fn: Callable[[ObjectPerception], Any],
        recurse_fn: Callable[[ObjectPerception, Any], Any],
    ) -> None:
        """Apply some function only to root elements of graph.
           Use return value from top level function as argument in
           recursively applied function. """
        for top_level in graph:
            # special cases not rendered here:
            if (
                top_level.perceived_obj.debug_handle == "the ground"
                or top_level.perceived_obj.debug_handle == "learner"
            ):
                continue
            top_return = top_fn(top_level.perceived_obj)
            nodes = [(node, top_return) for node in top_level.children]
            while nodes:
                recurse: List[Tuple[SceneNode, Any]] = []

                for node, ret in nodes:
                    if not node.children and ret is not None:
                        recurse_fn(node.perceived_obj, ret)
                    else:
                        new_return = recurse_fn(node.perceived_obj, ret)
                        recurse += [(child, new_return) for child in node.children]
                nodes = recurse

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
    parent_positions: Dict[str, Tuple[float, float, float]],
    iterations: int = 200,
    yield_steps: Optional[int] = None,
) -> Iterable[PositionsMap]:
    """
        Solves for positions of top-level objects.
    Args:
        parent_positions: list of top level objects to be positioned by model
        iterations: number of iterations to run model for
        yield_steps: number of iterations that must pass before a new set of positions is yielded/returned

    Returns: List of (3,) tensors, describing the updated positions of the top level objects, corresponding
             in terms of indices with parent_positions
    """
    objs = [
        AdamObject(name=name, initial_position=parent_position)
        for name, parent_position in parent_positions.items()
    ]

    return run_model(objs, num_iterations=iterations, yield_steps=yield_steps)


if __name__ == "__main__":
    parameters_only_entry_point(main, usage_message=USAGE_MESSAGE)
