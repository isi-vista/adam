"""Main interface for Panda3D rendering.
   Executing this module meant for testing purposes.
   Defines various default settings (ground plane, lighting, camera
   position. Provides interfaces for adding objects to the scene
   as well as for clearing the scene entirely.

   Ideally, this module should have as little contact as possible
   with the details of Curricula, and instead be focused on
   the capability to display objects with various properties
   supplied from elsewhere.
   """

from typing import Tuple, Optional, Dict
import sys
import os

import time
from direct.showbase.ShowBase import ShowBase  # pylint: disable=no-name-in-module
from direct.task import Task  # pylint: disable=no-name-in-module

from panda3d.core import DirectionalLight  # pylint: disable=no-name-in-module
from panda3d.core import PointLight  # pylint: disable=no-name-in-module
from panda3d.core import Material  # pylint: disable=no-name-in-module
from panda3d.core import NodePath  # pylint: disable=no-name-in-module
from panda3d.core import TextNode  # pylint: disable=no-name-in-module
from panda3d.core import AntialiasAttrib  # pylint: disable=no-name-in-module

from direct.gui.OnscreenText import OnscreenText  # pylint: disable=no-name-in-module
from adam.visualization.positioning import PositionsMap
from adam.visualization.utils import Shape, OBJECT_NAMES_TO_EXCLUDE

from adam.perception.developmental_primitive_perception import RgbColorPerception


class SituationVisualizer(ShowBase):

    model_to_file = {
        Shape.SQUARE: "cube.egg",
        Shape.CIRCULAR: "sphere.egg",
        Shape.OVALISH: "ovalish.egg",
        Shape.RECTANGULAR: "rectangular.egg",
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

        # self.render is the top node of the default scene graph

        self.ground_plane = self._load_model("ground.egg")
        self.ground_plane.reparentTo(self.render)
        self.ground_plane.setPos(0, 0, 0)
        m = Material()
        m.setDiffuse((255, 255, 255, 255))
        # the "1" argument to setMaterial is crucial to have it override
        # an existing material
        self.ground_plane.setMaterial(m, 1)

        # container of nodes to be dynamically added / removed
        self.geo_nodes: Dict[str, NodePath] = {}

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
        self.render.setAntialias(AntialiasAttrib.MAuto)

    def set_title(self, new_title: str):
        self.title_text.setText(new_title)
        self.run_for_seconds(0.5)

    def add_model(
        self,
        model_type: Shape,
        *,
        name: str,
        position: Tuple[float, float, float],
        color: RgbColorPerception = None,
        parent: Optional[NodePath] = None,
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
        print(f"adding: {model_type}")
        if color is None:
            color = RgbColorPerception(50, 50, 50)
        try:
            new_model = self._load_model(SituationVisualizer.model_to_file[model_type])
            new_model.name = name
        except KeyError:
            print(f"No geometry found for {model_type}")
            raise
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
        new_model.setPos(position[0], position[1], position[2])
        new_model.setColor((color.red / 255, color.green / 255, color.blue / 255, 1.0))
        return new_model

    def add_dummy_node(
        self,
        name: str,
        position: Tuple[float, float, float],
        parent: Optional[NodePath] = None,
    ) -> NodePath:
        print(f"\nAdding Dummy node: {name}")
        new_node = NodePath(name)
        if parent is None:
            new_node.reparentTo(self.render)
            if name in self.geo_nodes:
                raise RuntimeError(
                    f"Error using name {name}: Model names need to be unique"
                )
            self.geo_nodes[new_node.name] = new_node
        else:
            new_node.reparentTo(parent)
        new_node.setPos(*position)
        return new_node

    def clear_scene(self) -> None:
        """Clears out all added objects (other than ground plane, camera, lights)"""
        for node in self.geo_nodes.values():
            node.remove_node()
        self.geo_nodes = {}

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

    def run_for_seconds(self, seconds: float) -> None:
        """Executes main rendering loop for given seconds. This needs to be a
           healthy fraction of a second to see changes reflected in the scene."""
        start = int(time.time())
        while time.time() - start < seconds:
            self.taskMgr.step()

    def print_scene_graph(self) -> None:
        print(self.render.ls())

    def _load_model(self, name: str):
        working_dir = os.path.abspath((sys.path[0]))
        return self.loader.loadModel(working_dir + "/adam/visualization/models/" + name)
