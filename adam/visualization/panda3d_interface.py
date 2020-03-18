# pragma: no cover
"""Main interface for Panda3D rendering.
   Executing this module meant for testing purposes. (viewing an object in isolation)
   Defines various default settings (ground plane, lighting, camera
   position. Provides interfaces for adding objects to the scene
   as well as for clearing the scene entirely.

   Ideally, this module should have as little contact as possible
   with the details of Curricula, and instead be focused on
   the capability to display objects with various properties
   supplied from elsewhere.
   """

from typing import Tuple, Optional, Dict, List
import sys
import os

import time
from direct.showbase.ShowBase import ShowBase  # pylint: disable=no-name-in-module

from panda3d.core import DirectionalLight  # pylint: disable=no-name-in-module
from panda3d.core import AmbientLight  # pylint: disable=no-name-in-module
from panda3d.core import PointLight  # pylint: disable=no-name-in-module
from panda3d.core import Material  # pylint: disable=no-name-in-module
from panda3d.core import NodePath  # pylint: disable=no-name-in-module
from panda3d.core import TextNode  # pylint: disable=no-name-in-module
from panda3d.core import AntialiasAttrib  # pylint: disable=no-name-in-module
from panda3d.core import LPoint3f  # pylint: disable=no-name-in-module
from panda3d.core import Filename  # pylint: disable=no-name-in-module

from direct.gui.OnscreenText import OnscreenText  # pylint: disable=no-name-in-module
from adam.visualization.positioning import PositionsMap
from adam.visualization.utils import (
    Shape,
    OBJECT_NAMES_TO_EXCLUDE,
    GEON_SHAPES,
    MODEL_NAMES,
)

from adam.perception.developmental_primitive_perception import RgbColorPerception

from torch import Tensor


class SituationVisualizer(ShowBase):
    model_to_file = {
        Shape.SQUARE: "cube.egg",
        Shape.CIRCULAR: "smooth_sphere.egg",
        Shape.OVALISH: "ovalish.egg",
        Shape.RECTANGULAR: "rectangular.egg",
        Shape.IRREGULAR: "puddle.egg",
    }

    specific_model_to_file = {
        "ball": "basketball.egg",
        "hat": "cowboyhat.egg",
        # "box": "cardboard_box.egg",
        "cup": "mug.egg",
        "table": "table.egg",
        "door": "door.egg",
        "book": "book.egg",
        "bird": "bird.egg",
        "car": "car.egg",
        "cookie": "cookie.egg",
        "juice": "juice.egg",
        "milk": "milk.egg",
        "water": "water.egg",
        "chair-chairback_0": "chair-chairback.egg",
        "chair-chairseat_0": "chair-chairseat.egg",
        "chair-(furniture) leg_0": "chair-leg_left_front.egg",
        "chair-(furniture) leg_1": "chair-leg_left_back.egg",
        "chair-(furniture) leg_2": "chair-leg_right_back.egg",
        "chair-(furniture) leg_3": "chair-leg_right_front.egg",
        "table-(furniture) leg_0": "table-leg_left_front.egg",
        "table-(furniture) leg_1": "table-leg_left_back.egg",
        "table-(furniture) leg_2": "table-leg_right_back.egg",
        "table-(furniture) leg_3": "table-leg_right_front.egg",
        "table-tabletop_0": "table-tabletop.egg",
        "dog-dog-head_0": "dog-dog-head.egg",
        "dog-torso_0": "dog-torso.egg",
        "dog-tail_0": "dog-tail.egg",
        "dog-foot_0": "dog-foot_left_front.egg",
        "dog-leg-segment_0": "dog-leg-segment_0_left_front.egg",
        "dog-leg-segment_1": "dog-leg-segment_1_left_front.egg",
        "dog-foot_1": "dog-foot_right_front.egg",
        "dog-leg-segment_2": "dog-leg-segment_0_right_front.egg",
        "dog-leg-segment_3": "dog-leg-segment_1_right_front.egg",
        "dog-foot_2": "dog-foot_left_back.egg",
        "dog-leg-segment_4": "dog-leg-segment_0_left_back.egg",
        "dog-leg-segment_5": "dog-leg-segment_1_left_back.egg",
        "dog-foot_3": "dog-foot_left_back.egg",
        "dog-leg-segment_6": "dog-leg-segment_0_right_back.egg",
        "dog-leg-segment_7": "dog-leg-segment_1_right_back.egg",
        "car-tire_0": "car-tire_left_front.egg",
        "car-tire_1": "car-tire_left_back.egg",
        "car-tire_2": "car-tire_right_back.egg",
        "car-tire_3": "car-tire_right_front.egg",
        "car-body_0": "car-body.egg",
        "house-roof_0": "house-roof.egg",
        "house-wall_0": "house-wall.egg",
        "truck-body_0": "truck-body.egg",
        "truck-tire_0": "truck-tire_left_front",
        "truck-tire_1": "truck-tire_left_back",
        "truck-tire_2": "truck-tire_right_back",
        "truck-tire_3": "truck-tire_right_front",
        # TEMPORARY MAPPING OF BACK 8 WHEELS TO SAME MODEL AS FRONT WHEELS
        "truck-tire_4": "truck-tire_left_front",
        "truck-tire_5": "truck-tire_left_back",
        "truck-tire_6": "truck-tire_right_back",
        "truck-tire_7": "truck-tire_right_front",
        "truck-flatbed_0": "truck-body.egg",
        "bird-bird-head_0": "bird-head.egg",
        "bird-torso_0": "bird-torso.egg",
        "bird-foot_0": "bird-foot_left.egg",
        "bird-leg-segment_0": "bird-leg_above_knee_left.egg",
        "bird-leg-segment_1": "bird-leg_below_knee_left.egg",
        "bird-foot_1": "bird-foot_right.egg",
        "bird-leg-segment_2": "bird-leg_above_knee_right.egg",
        "bird-leg-segment_3": "bird-leg_below_knee_right.egg",
        "bird-wing_0": "bird-wing_left.egg",
        "bird-wing_1": "bird-wing_right.egg",
        "bird-tail_0": "bird-tail.egg",
        "baby-head_0": "person-head.egg",
        "baby-torso_0": "person-torso.egg",
        "baby-armsegment_0": "person-armsegment_0_right.egg",
        "baby-armsegment_1": "person-armsegment_1_right.egg",
        "baby-armsegment_2": "person-armsegment_2_left.egg",
        "baby-armsegment_3": "person-armsegment_3_left.egg",
        "baby-leg-segment_0": "person-leg-segment_0_right.egg",
        "baby-leg-segment_1": "person-leg-segment_1_right.egg",
        "baby-leg-segment_2": "person-leg-segment_2_left.egg",
        "baby-leg-segment_3": "person-leg-segment_3_left.egg",
        "baby-hand_0": "person-hand_right.egg",
        "baby-hand_1": "person-hand_left.egg",
        "baby-foot_0": "person-foot_right.egg",
        "baby-foot_1": "person-foot_left.egg",
        "dad-head_0": "person-head.egg",
        "dad-torso_0": "person-torso.egg",
        "dad-armsegment_0": "person-armsegment_0_right.egg",
        "dad-armsegment_1": "person-armsegment_1_right.egg",
        "dad-armsegment_2": "person-armsegment_2_left.egg",
        "dad-armsegment_3": "person-armsegment_3_left.egg",
        "dad-leg-segment_0": "person-leg-segment_0_right.egg",
        "dad-leg-segment_1": "person-leg-segment_1_right.egg",
        "dad-leg-segment_2": "person-leg-segment_2_left.egg",
        "dad-leg-segment_3": "person-leg-segment_3_left.egg",
        "dad-hand_0": "person-hand_right.egg",
        "dad-hand_1": "person-hand_left.egg",
        "dad-foot_0": "person-foot_right.egg",
        "dad-foot_1": "person-foot_left.egg",
        "mom-head_0": "person-head.egg",
        "mom-torso_0": "person-torso.egg",
        "mom-armsegment_0": "person-armsegment_0_right.egg",
        "mom-armsegment_1": "person-armsegment_1_right.egg",
        "mom-armsegment_2": "person-armsegment_2_left.egg",
        "mom-armsegment_3": "person-armsegment_3_left.egg",
        "mom-leg-segment_0": "person-leg-segment_0_right.egg",
        "mom-leg-segment_1": "person-leg-segment_1_right.egg",
        "mom-leg-segment_2": "person-leg-segment_2_left.egg",
        "mom-leg-segment_3": "person-leg-segment_3_left.egg",
        "mom-hand_0": "person-hand_right.egg",
        "mom-hand_1": "person-hand_left.egg",
        "mom-foot_0": "person-foot_right.egg",
        "mom-foot_1": "person-foot_left.egg",
    }

    models_used_for_scale_reference = {
        "chair": "chair.egg",
        "ball": "basketball.egg",
        "hat": "cowboyhat.egg",
        # "box": "cardboard_box.egg",
        "cup": "mug.egg",
        "table": "table.egg",
        "door": "door.egg",
        "book": "book.egg",
        "bird": "bird.egg",
        "car": "car.egg",
        "cookie": "cookie.egg",
        "dog": "dog.egg",
        "person": "person.egg",
        "dad": "person.egg",
        "mom": "person.egg",
        "baby": "person.egg",
        "juice": "juice.egg",
        "milk": "milk.egg",
        "water": "water.egg",
        "house": "house.egg",
        "truck": "truck.egg",
    }

    def __init__(self) -> None:
        super().__init__(self)
        # instantiate a light (or more) so that materials are visible

        dlight = DirectionalLight("mainLight")
        dlight_node = self.render.attachNewNode(dlight)
        self.render.setLight(dlight_node)

        plight = PointLight("pointLight")
        plight_node = self.render.attachNewNode(plight)
        plight_node.setPos(10, 20, 1)
        self.render.setLight(plight_node)

        alight = AmbientLight("alight")
        alight.setColor((0.2, 0.2, 0.2, 1))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)

        # self.render is the top node of the default scene graph

        self.ground_plane = self._load_model("ground.egg")
        self.ground_plane.reparentTo(self.render)
        self.ground_plane.setPos(0, 0, 0)
        m = Material()
        m.setDiffuse((249, 249, 249, 255))
        # the "1" argument to setMaterial is crucial to have it override
        # an existing material
        self.ground_plane.setMaterial(m, 1)

        # container of nodes to be dynamically added / removed
        self.geo_nodes: Dict[str, NodePath] = {}

        # list of debug bounding boxes
        self.debug_bounding_boxes: List[NodePath] = []

        # set default camera position/orientation:
        # default mouse controls have to be disabled to set a position manually
        self.disableMouse()

        # check for camera's existence due to possibility of invoking
        # without window in test suite:
        if self.camera is not None:
            self.camera.setPos(0, -45, 9)
            self.camera.setHpr(0, -10, 0)

        self.title_text = OnscreenText(
            text="placeholder",
            pos=(-1.25, -0.8),
            scale=0.07,
            mayChange=True,
            align=TextNode.ALeft,
        )

        # toggle on antialiasing
        self.render.setAntialias(AntialiasAttrib.M_multisample)

    def set_title(self, new_title: str):
        self.title_text.setText(new_title)
        self.run_for_seconds(0.5)

    def add_model(
        self,
        model_type: Shape,
        *,
        name: str,
        lookup_name: str,
        color: RgbColorPerception = None,
        parent: Optional[NodePath] = None,
        position: Optional[Tuple[float, float, float]] = None,
        scale_multiplier: Optional[float] = 1.0,
    ) -> NodePath:
        """
        Adds a piece of primitive geometry to the scene.
        Args:
            model_type: The shape used to represent the model
            *name*: unique name given to this object
            *position*: The position (x, y, z), (z is up) to place the new model
            *color*: RBG color for this model
            *parent*: Reference to a previously placed model (the type returned from this function). If supplied,
                    the new model will be *nested* under this parent model, making its position, orientation, scale
                    relative to the parent model.

        Returns: NodePath: a Panda3D type specifying the exact path to the object in the renderer's scene graph

        """

        if color is None:
            color = RgbColorPerception(122, 122, 122)
        # attempt to find a model file for a particular type of object
        if lookup_name in SituationVisualizer.specific_model_to_file:
            new_model = self._load_model(
                SituationVisualizer.specific_model_to_file[lookup_name]
            )
            new_model.name = name
            print(f"adding: {name}")
        # back off: attempt to find a model for the object's geon
        else:
            try:
                print(f"adding geon: {model_type}")
                new_model = self._load_model(
                    SituationVisualizer.model_to_file[model_type]
                )
                new_model.name = name
            except KeyError:
                print(f"No geometry found for {model_type}")
                raise

        if name.startswith("hat"):
            new_model.set_two_sided(True)

        if position:
            new_model.setPos(position[0], position[1], position[2])

        scale = new_model.getScale()
        new_model.setSx(scale.x * scale_multiplier)
        new_model.setSy(scale.y * scale_multiplier)
        new_model.setSz(scale.z * scale_multiplier)
        # top level:
        if parent is None:
            if name in self.geo_nodes:
                raise RuntimeError(
                    f"Error using name {name}: Model names need to be unique"
                )
            self.geo_nodes[new_model.name] = new_model
            new_model.reparentTo(self.render)
        # nested
        else:
            new_model.reparentTo(parent)
        new_model.setColor((color.red / 255, color.green / 255, color.blue / 255, 1.0))

        return new_model

    def add_dummy_node(
        self,
        name: str,
        lookup_name: str,
        parent: Optional[NodePath] = None,
        position: Optional[Tuple[float, float, float]] = None,
        scale_multiplier: Optional[float] = 1.0,
    ) -> NodePath:
        # TODO: name 'dummy_node' isn't totally accurate now
        print(f"lookup name for dummy node: {lookup_name}")
        if lookup_name in SituationVisualizer.specific_model_to_file:
            print(f"\nADDING SPECIFIC MODEL")
            new_node = self._load_model(
                SituationVisualizer.specific_model_to_file[lookup_name]
            )
            new_node.name = name

        else:
            new_node = NodePath(name)
        scale = new_node.getScale()
        new_node.setSx(scale.x * scale_multiplier)
        new_node.setSy(scale.y * scale_multiplier)
        new_node.setSz(scale.z * scale_multiplier)
        if position:
            new_node.setPos(position[0], position[1], position[2])
        if parent is None:
            new_node.reparentTo(self.render)
            if name in self.geo_nodes:
                raise RuntimeError(
                    f"Error using name {name}: Model names need to be unique"
                )
            self.geo_nodes[new_node.name] = new_node
        else:
            new_node.reparentTo(parent)
        return new_node

    def add_debug_bounding_box(self, name: str, position: Tensor, scale: Tensor):
        new_node = self._load_model("debug_cube.egg")
        new_node.name = name
        new_node.setPos(position.data[0], position.data[1], position.data[2])
        new_node.setScale(scale.data[0][0], scale.data[1][1], scale.data[2][2])
        # m = Material()
        # m.setDiffuse((200, 0, 0, 0))
        # new_node.setColor((200, 0, 0, 10))
        # new_node.setMaterial(m, 1)
        new_node.reparentTo(self.render)
        self.debug_bounding_boxes.append(new_node)

    def clear_scene(self) -> None:
        """Clears out all added objects (other than ground plane, camera, lights)"""
        for node in self.geo_nodes.values():
            node.remove_node()
        self.geo_nodes = {}
        self.clear_debug_nodes()
        self.print_scene_graph()

    def clear_debug_nodes(self) -> None:
        for node in self.debug_bounding_boxes:
            node.remove_node()
        self.debug_bounding_boxes = []

    def top_level_positions(self) -> Dict[str, Tuple[float, float, float]]:
        """Returns a Map of name -> position of all nodes of geometry objects
           (so not cameras and lights). """
        return {name: node.getPos() for name, node in self.geo_nodes.items()}

    def set_positions(self, new_positions: PositionsMap):
        """Modify the position of all top level geometry nodes in the scene."""
        for name, position in new_positions.name_to_position.items():
            if name in OBJECT_NAMES_TO_EXCLUDE:
                continue
            self.geo_nodes[name].setPos(
                position.data[0], position.data[1], position.data[2]
            )

    def multiply_scale(self, geo_node_name: str, scale_multiplier: float):
        node = self.geo_nodes[geo_node_name]
        scale = node.get_scale()
        node.setSx(scale.x * scale_multiplier)
        node.setSy(scale.y * scale_multiplier)
        node.setSz(scale.z * scale_multiplier)

    def run_for_seconds(self, seconds: float) -> None:
        """Executes main rendering loop for given seconds. This needs to be a
           healthy fraction of a second to see changes reflected in the scene."""
        start = int(time.time())
        while time.time() - start < seconds:
            self.taskMgr.step()

    def print_scene_graph(self) -> None:
        print(self.render.ls())

    def get_model_scales(self) -> Dict[str, Tuple[float, float, float]]:
        scale_map: Dict[str, Tuple[float, float, float]] = {}
        for shape in GEON_SHAPES:
            model = self._load_model(SituationVisualizer.model_to_file[shape])
            bounds = model.getTightBounds()
            scale_map[shape.name] = bounds_to_scale(bounds[0], bounds[1])
        for name in MODEL_NAMES:
            model = self._load_model(
                SituationVisualizer.models_used_for_scale_reference[name]
            )
            bounds = model.getTightBounds()
            scale_map[name] = bounds_to_scale(bounds[0], bounds[1])

        return scale_map

    def _load_model(self, name: str):
        working_dir = Filename.fromOsSpecific(os.path.abspath((sys.path[0])))
        return self.loader.loadModel(working_dir + "/adam/visualization/models/" + name)


def bounds_to_scale(
    min_corner: LPoint3f, max_corner: LPoint3f
) -> Tuple[float, float, float]:
    return (
        (max_corner.x - min_corner.x) / 2,
        (max_corner.y - min_corner.y) / 2,
        (max_corner.z - min_corner.z) / 2,
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    VISUALIZER = SituationVisualizer()
    print(f"Current name to file bindings:\n{VISUALIZER.specific_model_to_file}")
    PARSER = ArgumentParser()
    PARSER.add_argument(
        "model_names",
        type=str,
        nargs="*",
        help="model name (lowercase) to view in isolation",
    )
    PARSER.add_argument("--x", type=float, help="x position", default=0.0)
    PARSER.add_argument("--y", type=float, help="y position", default=0.0)
    PARSER.add_argument("--z", type=float, help="z position", default=0.0)
    ARGS = PARSER.parse_args()

    for MODEL_NAME in ARGS.model_names:
        NODE = VISUALIZER.add_model(
            Shape.IRREGULAR,
            name=MODEL_NAME,
            color=RgbColorPerception(100, 100, 100),
            lookup_name=MODEL_NAME,
        )
        node_pos = NODE.get_pos()
        NODE.setPos(node_pos.x + ARGS.x, node_pos.y + ARGS.y, node_pos.z + ARGS.z)
    VISUALIZER.set_title(ARGS.model_names[0])
    VISUALIZER.run()
