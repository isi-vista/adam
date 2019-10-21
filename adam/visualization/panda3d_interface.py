from math import pi, sin, cos

from typing import Tuple
import sys, os

import time

from direct.showbase.ShowBase import ShowBase
from direct.task import Task

from panda3d.core import Filename
from panda3d.core import DirectionalLight, PointLight, Material

from direct.gui.OnscreenText import OnscreenText
from pandac.PandaModules import TextNode

# checking to see that the import functions:
import adam

from adam.visualization.utils import Shape

from adam.math_3d import Point
from adam.perception.developmental_primitive_perception import RgbColorPerception as Color


class SituationVisualizer(ShowBase):
    def __init__(self) -> None:
        ShowBase.__init__(self)

        # instantiate a light so that materials are visible

        dlight = DirectionalLight("mainLight")
        dlight_node = self.render.attachNewNode(dlight)
        self.render.setLight(dlight_node)

        plight = PointLight("pointLight")
        plight_nodePtr = self.render.attachNewNode(plight)
        plight_nodePtr.setPos(10, 20, 0)
        self.render.setLight(plight_nodePtr)

        # self.render is the top node of the default scene graph

        self.ground_plane = self._load_model("ground.egg")
        self.ground_plane.reparentTo(self.render)
        # self.ground_plane.setHpr(0, 180, 0) # heading, pitch, roll
        self.ground_plane.setPos(0, 0, -1)
        m: Material = Material()
        m.setDiffuse((255, 255, 255, 255))
        self.ground_plane.setMaterial(m, 1)

        # nodes to be dynamically added / removed
        self.geo_nodes = []

        # set default camera position/orientation:
        # default mouse controls have to be disabled to set a position manually
        self.disableMouse()
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
        self.taskMgr.doMethodLater(0.25, self._cameraLocationTask, "CameraLocationTask")

    def add_model(
        self,
        model_type: Shape,
        pos: Tuple[float, float, float],
        col: Color = Color(50, 50, 50),
    ) -> None:
        """Adds a piece of primitive geometry to the scene.
        Will need to be expanded to account for orientation, color, position, etc"""
        if model_type == Shape.SQUARE:
            new_model = self._load_model("cube.egg")
        elif model_type == Shape.RECTANGULAR:
            new_model = self._load_model("rectangular.egg")
        elif model_type == Shape.OVALISH:
            new_model = self._load_model("ovalish.egg")
        elif model_type == Shape.CIRCULAR:
            new_model = self._load_model("sphere.egg")
        elif model_type == Shape.IRREGULAR:
            # TODO: fill this out
            return
        self.geo_nodes.append(new_model)
        new_model.reparentTo(self.render)
        new_model.setPos(pos[0], pos[1], pos[2])
        new_model.setColor((col.red / 255, col.green / 255, col.blue / 255, 1.0))

    def clear_scene(self) -> None:
        """Clears out all added objects (other than ground plane, camera, lights"""

        # need to understand how the scene graph works in order to remove all nodes that
        # have been added to the scene w/ geometry
        for node in self.geo_nodes:
            node.remove_node()
        self.geo_nodes = []

    def test_scene_init(self) -> None:
        """Initialize a test scene with sample geometry, including a camera rotate task"""
        self.cylinder = self._load_model("cylinder.egg")
        # Reparent the model to render.
        self.cylinder.reparentTo(self.render)

        self.cube = self._load_model("cube.egg")
        self.cube.reparentTo(self.render)
        self.cube.setPos(0, 0, 5)
        # the "1" argument to setMaterial is crucial to have it override
        # an existing material
        self.cube.setColor((1.0, 0.0, 0.0, 1.0))

        self.cube2 = self._load_model("cube.egg")
        self.cube2.reparentTo(self.render)
        self.cube2.setPos(5, 0, 0)
        self.cube2.setScale(1.25, 1.25, 1.25)
        self.cube2.setColor((0, 1, 0, 0.5))

        # Add the spinCameraTask procedure to the task manager.
        self.taskMgr.add(self._spinCameraTask, "SpinCameraTask", priority=-100)

    def run_for_seconds(self, seconds: float) -> None:
        """Executes main loop for given seconds"""
        start = int(time.time())
        while time.time() - start < seconds:
            self.taskMgr.step()

    # Define a procedure to move the camera.
    def _spinCameraTask(self, task):
        angleDegrees = task.time * 6.0
        angleRadians = angleDegrees * (pi / 180.0)
        self.camera.setPos(25 * sin(angleRadians), -25.0 * cos(angleRadians), 4)
        self.camera.setHpr(angleDegrees, 0, 0)
        return Task.cont

    def _cameraLocationTask(self, task):
        pos = self.camera.getPos()
        hpr = self.camera.getHpr()
        self.camera_pos_text.setText(f"position: {pos}")
        self.camera_hpr_text.setText(f"orientation: {hpr}")
        return Task.again  # perform task with delay

    def _load_model(self, name: str):
        working_dir = os.path.abspath((sys.path[0]))
        return self.loader.loadModel(working_dir + "/adam/visualization/models/" + name)



# for testing purposes
if __name__ == "__main__":

    app = SituationVisualizer()
    app.test_scene_init()

    app.add_model(Shape.SQUARE, (1, 2, 2))

    app.run_for_seconds(3)

    app.run_for_seconds(6)

    app.clear_scene()

    app.run_for_seconds(3)

    app.add_model(Shape.RECTANGULAR, (-2, 2, 2))

    app.add_model(Shape.CIRCULAR, (0, 0, 8))

    app.add_model(Shape.OVALISH, (4, 5, 2))

    app.run_for_seconds(3)

    # input("Press ENTER to continue")
    app.taskMgr.run()
