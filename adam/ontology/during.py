from itertools import chain
from typing import Generic, Mapping, TypeVar, List

from attr import attrib, attrs
from immutablecollections import (
    ImmutableSet,
    ImmutableSetMultiDict,
    immutableset,
    immutablesetmultidict,
)
from immutablecollections.converter_utils import (
    _to_immutableset,
    _to_immutablesetmultidict,
)

from adam.ontology.phase1_spatial_relations import SpatialPath
from adam.relation import Relation

_ObjectT = TypeVar("_ObjectT")
_NewObjectT = TypeVar("_NewObjectT")


@attrs(frozen=True)
class DuringAction(Generic[_ObjectT]):
    objects_to_paths: ImmutableSetMultiDict[_ObjectT, SpatialPath[_ObjectT]] = attrib(
        converter=_to_immutablesetmultidict, default=immutablesetmultidict(), kw_only=True
    )
    at_some_point: ImmutableSet[Relation[_ObjectT]] = attrib(
        converter=_to_immutableset, default=immutableset(), kw_only=True
    )
    continuously: ImmutableSet[Relation[_ObjectT]] = attrib(
        converter=_to_immutableset, default=immutableset(), kw_only=True
    )

    def copy_remapping_objects(
        self, object_mapping: Mapping[_ObjectT, _NewObjectT]
    ) -> "DuringAction[_NewObjectT]":
        return DuringAction(
            objects_to_paths=(
                (object_mapping[object_], path.copy_remapping_objects(object_mapping))
                for (object_, path) in self.objects_to_paths.items()
            ),
            at_some_point=(
                relation.copy_remapping_objects(object_mapping)
                for relation in self.at_some_point
            ),
            continuously=(
                relation.copy_remapping_objects(object_mapping)
                for relation in self.continuously
            ),
        )

    def accumulate_referenced_objects(self, object_accumulator: List[_ObjectT]) -> None:
        r"""
        Adds all objects referenced by this `DuringAction` to *object_accumulator*.
        """
        for (_, path) in self.objects_to_paths.items():
            path.accumulate_referenced_objects(object_accumulator)
        for relation in self.at_some_point:
            relation.accumulate_referenced_objects(object_accumulator)
        for relation in self.continuously:
            relation.accumulate_referenced_objects(object_accumulator)

    def union(self, other_during: "DuringAction[_ObjectT]") -> "DuringAction[_ObjectT]":
        return DuringAction(
            objects_to_paths=immutablesetmultidict(
                chain(
                    self.objects_to_paths.items(), other_during.objects_to_paths.items()
                )
            ),
            at_some_point=chain(self.at_some_point, other_during.at_some_point),
            continuously=chain(self.continuously, other_during.continuously),
        )
