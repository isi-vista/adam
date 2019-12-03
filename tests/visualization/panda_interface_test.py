from pandac.PandaModules import ConfigVariableString  # pylint: disable=no-name-in-module

from adam.visualization.panda3d_interface import SituationVisualizer
from adam.visualization.utils import Shape

ConfigVariableString("window-type", "none").setValue("none")

# This should be the only test to actually instantiate panda3d
def test_basic_3d_scene() -> None:
    app = SituationVisualizer()
    app.test_scene_init()

    app.add_model(Shape.SQUARE, (1, 2, 2))
    app.add_model(Shape.RECTANGULAR, (2, 2, 2))
    try:
        app.add_model(Shape.IRREGULAR, (4, 4, 4))
    except KeyError:
        pass
    oval = app.add_model(Shape.OVALISH, (5, 5, 5))
    app.add_model(Shape.CIRCULAR, (7, 7, 7), col=None, parent=oval)

    app.print_scene_graph()

    dummy_node = app.add_dummy_node("dummy", (0, 0, 0))
    other_dummy_node = app.add_dummy_node("second_dummy", dummy_node)

    print(other_dummy_node)

    app.test_scene_init()

    app.clear_scene()

    app.add_model(Shape.RECTANGULAR, (-2, 2, 2))

    app.add_model(Shape.CIRCULAR, (0, 0, 8))

    app.add_model(Shape.OVALISH, (4, 5, 2))
