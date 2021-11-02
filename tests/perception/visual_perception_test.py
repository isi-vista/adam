from adam.perception import PerceptualRepresentation
from adam.perception.visual_perception import VisualPerceptionFrame
from tests.perception import ONE_OBJECT_TEST_SCENE_JSON


def test_one_object_scene():
    PerceptualRepresentation.single_frame(
        VisualPerceptionFrame.from_json_str(ONE_OBJECT_TEST_SCENE_JSON)
    )
