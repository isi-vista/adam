"""
An entry point which allows the user to render object images to either a file or
3D window. It is handy for testing object descriptions.
"""
from math import sin, cos
from pathlib import Path

from attr import attrs, attrib
from attr.validators import instance_of
from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point
import panda3d.core
from direct.showbase.ShowBase import ShowBase


#def main(params: Parameters) -> None:
    # if params.optional_boolean_with_default("windowless", False):
    #     # start Panda3D in windowless format,
    #     # see https://discourse.panda3d.org/t/solved-render-only-one-frame-and-other-questions/12899/8
    #     panda3d.core.loadPrcFileData("",
    #                     """
    #                        load-display p3tinydisplay # to force CPU only rendering (to make it
    #                        available as an option if everything else fail, use aux-display
    #                        p3tinydisplay)
    #                        window-type offscreen # Spawn an offscreen buffer (use window-type
    #                        none if you don't need any rendering)
    #                        audio-library-name null # Prevent ALSA errors
    #                        show-frame-rate-meter 0
    #                        sync-video 0
    #                     """)
    #     base = ShowBase()
    #     base.graphics_engine.renderFrame()
    #     base.screenshot(namePrefix='screenshot', defaultFilename=1, source=None, imageComment="")
    # else:
from adam.perception.marr import Marr3dObject


@attrs(frozen=True)
class CylinderDrawer:
    model_path: Path = attrib(validator=instance_of(Path))
    base: ShowBase = attrib(validator=instance_of(ShowBase))
    bounding_cylinder_alpha = attrib(validator=instance_of(float))

    def draw_marr3d_object(self, obj: Marr3dObject) -> None:
        self.draw_cylinder(obj.bounding_cylinder, alpha=0.1)


    def draw_cylinder(self,
                      model_dir: Path,
                      start_x: float, start_y: float, start_z: float,
                      orientation_deg1: float, orientation_deg2:float, orientation_deg3: float,
                      diameter: float, length: float):
        # repeated load_models are cheap because they are cached by Panda3D
        cylinder_model = self.base.loader.loadModel(str(model_dir / "cylinder"))
        # todo: these are arbitrary, fix them!
        cylinder_model.setPos(start_x, start_y, start_z)
        cylinder_model.setScale(diameter, diameter, length)
        cylinder_model.reparentTo(self.base.render)
        return cylinder_model

class App(ShowBase):



    def __init__(self):
        ShowBase.__init__(self)

        base.disableMouse()
        base.useDrive()

        self.scene = self.loader.loadModel("models/environment")
        # Reparent the model to render.
        #self.scene.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        self.scene.setScale(0.25, 0.25, 0.25)
        self.scene.setPos(-8, 42, 0)

        # Load the environment model.
        model_dir = (Path(__file__).parent / "../../models").absolute()
        cylinder_1 = self.draw_cylinder(model_dir,
                                        -5, -5, -5,
                                        0, 0, 0,
                                        2, 10)
        cylinder_1.setColor(0.8, 0.2, 0.2, 1.0)

        cylinder_2 = self.draw_cylinder(model_dir,
                                        0, 0, 0,
                                        0, 0, 0,
                                        5, 5)
        cylinder_2.setColor(0.2, 0.8, 0.2, 1.0)

        #self.cyl = self.loader.loadModel(str(model_dir / "cylinder"))
        #self.cyl = self.loader.loadModel("models/teapot")
        #self.cyl.setScale(0.25, 0.25, 0.25)
        #self.cyl.setColor(0.8, 0.2, 0.2, 1.0)
        #self.cyl.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        self.camera.setPos(20 , -20.0 , 3)
        self.camera.setHpr(0, 0, 0)


if __name__ == "__main__":
#    parameters_only_entry_point(main)
    base = App()
    base.run()
