from typing import TypeVar
from pickle import HIGHEST_PROTOCOL
from io import BytesIO, SEEK_SET

import pytest

from adam.pickle import AdamPickler, AdamUnpickler
from adam.learner.object_recognizer import SHARED_WORLD_ITEMS
from adam.perception import GROUND_PERCEPTION
from adam.perception import LEARNER_PERCEPTION


T = TypeVar("T")


def _pickle_and_unpickle_object(object: T) -> T:
    stream = BytesIO()
    pickler = AdamPickler(file=stream, protocol=HIGHEST_PROTOCOL)
    pickler.dump(object)

    stream.seek(0, SEEK_SET)
    unpickler = AdamUnpickler(file=stream)
    return unpickler.load()


def test_pickle_preserves_shared_world_item_identity():
    for item in SHARED_WORLD_ITEMS:
        new_item = _pickle_and_unpickle_object(item)
        assert new_item is item


def test_pickle_preserves_ground_perception_identity():
    new_ground_perception = _pickle_and_unpickle_object(GROUND_PERCEPTION)
    assert new_ground_perception is GROUND_PERCEPTION


@pytest.mark.xfail(
    reason="Learner perception is not currently preserved "
    "since not preserving it doesn't seem to cause any learning problems."
)
def test_pickle_preserves_learner_perception_identity():
    new_ground_perception = _pickle_and_unpickle_object(LEARNER_PERCEPTION)
    assert new_ground_perception is LEARNER_PERCEPTION
