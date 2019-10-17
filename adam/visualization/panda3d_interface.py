from math import pi, sin, cos

import sys, os

from direct.showbase.ShowBase import ShowBase
from direct.task import Task

from panda3d.core import Filename
from panda3d.core import DirectionalLight
from panda3d.core import PointLight

# checking to see that the import functions:
import adam

from adam.visualization.simple_material import SimpleMaterial


class MyApp(ShowBase):
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

        # Load the environment model.
        # self.scene = self.loader.loadModel("models/environment")

        self.cylinder = self.load_model("cylinder.egg")
        # Reparent the model to render.
        self.cylinder.reparentTo(self.render)

        self.cube = self.load_model("cube.egg")
        self.cube.reparentTo(self.render)
        self.cube.setPos(0, 0, 5)
        # the "1" argument to setMaterial is crucial to have it override
        # an existing material
        self.cube.setMaterial(SimpleMaterial(255, 0, 0, name="red").mat, 1)

        self.cube2 = self.load_model("cube.egg")
        self.cube2.reparentTo(self.render)
        self.cube2.setPos(5, 0, 0)
        self.cube2.setScale(1.25, 1.25, 1.25)
        self.cube2.setMaterial(SimpleMaterial(0, 255, 0, 100, name="green").mat, 1)

        # Apply scale and position transforms on the model.
        # self.scene.setScale(0.25, 0.25, 0.25)
        # self.scene.setPos(-8, 42, 0)

        # Add the spinCameraTask procedure to the task manager.
        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")

    # Define a procedure to move the camera.
    def spinCameraTask(self, task):
        angleDegrees = task.time * 24.0
        angleRadians = angleDegrees * (pi / 180.0)
        self.camera.setPos(20 * sin(angleRadians), -20.0 * cos(angleRadians), 3)
        self.camera.setHpr(angleDegrees, 0, 0)
        return Task.cont

    def load_model(self, name: str):
        working_dir = os.path.abspath((sys.path[0]))
        return self.loader.loadModel(working_dir + "/adam/visualization/models/" + name)


if __name__ == "__main__":

    app = MyApp()
    app.run()
