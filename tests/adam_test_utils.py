from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception import ObjectPerception


def perception_with_handle(
    frame: DevelopmentalPrimitivePerceptionFrame, handle: str
) -> ObjectPerception:
    for object_perception in frame.perceived_objects:
        if object_perception.debug_handle == handle:
            return object_perception
    raise RuntimeError(
        f"Could not find object perception with handle {handle} " f"in {frame}"
    )
