from adam.axis import GeonAxis
from adam.learner.object_recognizer import SHARED_WORLD_ITEMS
from adam.perception import ObjectPerception, GROUND_PERCEPTION, LEARNER_PERCEPTION
from pickle import Pickler, Unpickler


PERSISTENT_AXIS_TAG = "PersistentAxis"
PERSISTENT_OBJECT_PERCEPTION_TAG = "PersistentObjectPerception"


class AdamPickler(Pickler):
    """
    A pickler customized for ADAM's needs.

    This pickler implements persistence logic for things that there should only be one of, like the
    ground, or the "gravitational up/down" axis.
    """

    @staticmethod
    def persistent_id(object_):
        persistent_id_ = None

        if isinstance(object_, GeonAxis):
            if object_ in SHARED_WORLD_ITEMS:
                persistent_id_ = PERSISTENT_AXIS_TAG, object_.debug_name

        elif isinstance(object_, ObjectPerception):
            if object_ == GROUND_PERCEPTION or object_ == LEARNER_PERCEPTION:
                persistent_id_ = PERSISTENT_OBJECT_PERCEPTION_TAG, object_.debug_handle

        return persistent_id_


class AdamUnpickler(Unpickler):
    """
    An unpickler customized for ADAM's needs.

    This pickler implements the loading of things that there should only be one of, like the ground,
    or the "gravitational up/down" axis.
    """

    @staticmethod
    def persistent_load(persistent_id):
        if not isinstance(persistent_id, tuple) or len(persistent_id) < 1:
            raise RuntimeError(
                "Got bad persistent ID {pid}; persistent ID must be a tuple of at least one item"
            )

        tag = persistent_id[0]
        if tag == PERSISTENT_AXIS_TAG:
            name = persistent_id[1]
            for axis in SHARED_WORLD_ITEMS:
                # Check that the item is an axis before just to make sure
                if isinstance(axis, GeonAxis) and axis.debug_name == name:
                    return axis
            raise RuntimeError(
                f"Persistent axis found with name {name} but no such shared world item found!"
            )

        elif tag == PERSISTENT_OBJECT_PERCEPTION_TAG:
            name = persistent_id[1]
            if name == GROUND_PERCEPTION.debug_handle:
                return GROUND_PERCEPTION
            elif name == LEARNER_PERCEPTION.debug_handle:
                return LEARNER_PERCEPTION
            else:
                raise RuntimeError(
                    f"Persistent object perception found with name {name} but no such object perception known!"
                )
        else:
            raise RuntimeError(f"Got unrecognized persistent ID {persistent_id}!")
