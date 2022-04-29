from typing import TypeVar, Mapping, List

from typing_extensions import Protocol, runtime

_ObjectFromT = TypeVar("_ObjectFromT")
_ObjectToT = TypeVar("_ObjectToT")


@runtime
class CanRemapObjects(Protocol[_ObjectFromT]):
    def copy_remapping_objects(
        self, object_map: Mapping[_ObjectFromT, _ObjectToT]
    ) -> "CanRemapObjects[_ObjectToT]":
        pass

    def accumulate_referenced_objects(
        self, object_accumulator: List[_ObjectFromT]
    ) -> None:
        pass
