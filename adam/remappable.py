from typing import TypeVar, Generic, Mapping, List

from typing_extensions import Protocol

_ObjectFromT = TypeVar("_ObjectFromT")
_ObjectToT = TypeVar("_ObjectToT")


class CanRemapObjects(Protocol, Generic[_ObjectFromT]):
    def copy_remapping_objects(
        self, object_map: Mapping[_ObjectFromT, _ObjectToT]
    ) -> "CanRemapObjects[_ObjectToT]":
        pass

    def accumulate_referenced_objects(
        self, object_accumulator: List[_ObjectFromT]
    ) -> None:
        pass
