from adam.math_3d import Point
from adam.perception import SingleObjectDummyVisualPerception


def test_dummy_visual_perception():
    located_truck = SingleObjectDummyVisualPerception(
        "truck", location=Point(1.0, 2.0, 3.0)
    )
    assert located_truck.tag == "truck"
    assert located_truck.location == Point(1.0, 2.0, 3.0)
