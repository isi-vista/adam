from pandac.PandaModules import ConfigVariableString  # pylint: disable=no-name-in-module

from adam.visualization.panda3d_interface import SituationVisualizer
from adam.visualization.utils import Shape

ConfigVariableString("window-type", "none").setValue("none")

# This should be the only test to actually instantiate panda3d
def test_basic_3d_scene() -> None:
    app = SituationVisualizer()
    app.test_scene_init()

    app.add_model(Shape.SQUARE, (1, 2, 2))

    # app.run_for_seconds(0.25)

    # app.run_for_seconds(0.25)

    app.clear_scene()

    # app.run_for_seconds(0.25)

    app.add_model(Shape.RECTANGULAR, (-2, 2, 2))

    app.add_model(Shape.CIRCULAR, (0, 0, 8))

    app.add_model(Shape.OVALISH, (4, 5, 2))

    # app.run_for_seconds(0.25)
