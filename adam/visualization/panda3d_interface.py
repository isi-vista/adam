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
from math import pi, sin, cos

from typing import Tuple, List, Optional
import sys
import os

import time
import torch
from direct.showbase.ShowBase import ShowBase  # pylint: disable=no-name-in-module
from direct.task import Task  # pylint: disable=no-name-in-module

from panda3d.core import DirectionalLight  # pylint: disable=no-name-in-module
from panda3d.core import PointLight  # pylint: disable=no-name-in-module
from panda3d.core import Material  # pylint: disable=no-name-in-module
from panda3d.core import NodePath  # pylint: disable=no-name-in-module
from panda3d.core import TextNode  # pylint: disable=no-name-in-module

from direct.gui.OnscreenText import OnscreenText  # pylint: disable=no-name-in-module

from adam.visualization.utils import Shape

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
        plight_node.setPos(10, 20, 0)
        self.render.setLight(plight_node)

        # self.render is the top node of the default scene graph

        self.ground_plane = self._load_model("ground.egg")
        self.ground_plane.reparentTo(self.render)
        self.ground_plane.setPos(0, 0, -2)
        m = Material()
        m.setDiffuse((255, 255, 255, 255))
        # the "1" argument to setMaterial is crucial to have it override
        # an existing material
        self.ground_plane.setMaterial(m, 1)

        # container of nodes to be dynamically added / removed
        self.geo_nodes: List[NodePath] = []

        # set default camera position/orientation:
        # default mouse controls have to be disabled to set a position manually
        self.disableMouse()

        # check for camera's existence due to possibility of invoking
        # without window in test suite:
        if self.camera is not None:
            self.camera.setPos(0, -45, 9)
            self.camera.setHpr(0, -10, 0)

        # set GUI text re: camera position and periodic task
        self.camera_pos_text = OnscreenText(
            text="position:",
            pos=(-1.25, -0.7),
            scale=0.07,
            mayChange=True,
            align=TextNode.ALeft,
        )
        self.camera_hpr_text = OnscreenText(
            text="orientation:",
            pos=(-1.25, -0.8),
            scale=0.07,
            mayChange=True,
            align=TextNode.ALeft,
        )
        self.taskMgr.doMethodLater(0.25, self._camera_location_task, "CameraLocationTask")

    def add_model(
        self,
        model_type: Shape,
        pos: Tuple[float, float, float],
        col: RgbColorPerception = None,
        parent: Optional[NodePath] = None,
    ) -> NodePath:
        """Adds a piece of primitive geometry to the scene.
        Will need to be expanded to account for orientation, color, position, etc"""
        print(f"adding: {model_type}")
        if col is None:
            col = RgbColorPerception(50, 50, 50)
        try:
            new_model = self._load_model(SituationVisualizer.model_to_file[model_type])
        except KeyError:
            print(f"No geometry found for {model_type}")
            raise
        # top level:
        if parent is None:
            self.geo_nodes.append(new_model)
            new_model.reparentTo(self.render)
        # nested
        else:
            new_model.reparentTo(parent)
        new_model.setPos(pos[0], pos[1], pos[2])
        new_model.setColor((col.red / 255, col.green / 255, col.blue / 255, 1.0))
        return new_model

    def add_dummy_node(
        self,
        name: str,
        pos: Tuple[float, float, float],
        parent: Optional[NodePath] = None,
    ) -> NodePath:
        print(f"\nAdding Dummy node: {name}")
        new_node = NodePath(name)
        if parent is None:
            new_node.reparentTo(self.render)
            self.geo_nodes.append(new_node)
        else:
            new_node.reparentTo(parent)
        new_node.setPos(*pos)
        return new_node

    def clear_scene(self) -> None:
        """Clears out all added objects (other than ground plane, camera, lights)"""
        for node in self.geo_nodes:
            node.remove_node()
        self.geo_nodes = []

    def top_level_positions(self) -> List[Tuple[float, float, float]]:
        """Returns a list of all positions of top level nodes of geometry objects
           (so not cameras and lights). """
        # TODO: need to ensure this is deterministic
        return [node.getPos() for node in self.geo_nodes]

    def set_positions(self, new_positions: List[torch.Tensor]):
        """Modify the position of all top level geometry nodes in the scene.
           The scene should not be modified in any way between retrieving node
           positions and giving them new values, or this will produce incorrect
           results. """
        assert len(new_positions) == len(self.geo_nodes)
        for node, pos in zip(self.geo_nodes, new_positions):
            node.setPos(*pos)

    def test_scene_init(self) -> None:
        """Initialize a test scene with sample geometry, including a camera rotate task"""
        cylinder = self._load_model("cylinder.egg")
        self.geo_nodes.append(cylinder)
        # Reparent the model to render.
        cylinder.reparentTo(self.render)

        cube = self._load_model("cube.egg")
        self.geo_nodes.append(cube)
        cube.reparentTo(self.render)
        cube.setPos(0, 0, 5)
        cube.setColor((1.0, 0.0, 0.0, 1.0))

        cube2 = self._load_model("cube.egg")
        self.geo_nodes.append(cube2)
        cube2.reparentTo(self.render)
        cube2.setPos(5, 0, 0)
        cube2.setScale(1.25, 1.25, 1.25)
        cube2.setColor((0, 1, 0, 0.5))

        # Add the spinCameraTask procedure to the task manager.
        self.taskMgr.add(self._spin_camera_task, "SpinCameraTask", priority=-100)

    def run_for_seconds(self, seconds: float) -> None:
        """Executes main rendering loop for given seconds. This needs to be a
           healthy fraction of a second to see changes reflected in the scene."""
        start = int(time.time())
        while time.time() - start < seconds:
            self.taskMgr.step()

    def print_scene_graph(self) -> None:
        print(self.render.ls())

    # Define a procedure to move the camera.
    def _spin_camera_task(self, task):
        angle_degrees = task.time * 6.0
        angle_radians = angle_degrees * (pi / 180.0)
        self.camera.setPos(25 * sin(angle_radians), -25.0 * cos(angle_radians), 4)
        self.camera.setHpr(angle_degrees, 0, 0)
        return Task.cont

    def _camera_location_task(self, task):  # pylint: disable=unused-argument
        pos = self.camera.getPos()
        hpr = self.camera.getHpr()
        self.camera_pos_text.setText(f"position: {pos}")
        self.camera_hpr_text.setText(f"orientation: {hpr}")
        return Task.again  # perform task with delay

    def _load_model(self, name: str):
        working_dir = os.path.abspath((sys.path[0]))
        return self.loader.loadModel(working_dir + "/adam/visualization/models/" + name)


# for testing purposes
