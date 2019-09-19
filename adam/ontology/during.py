from typing import Generic, Mapping, TypeVar

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
    paths: ImmutableSetMultiDict[_ObjectT, SpatialPath[_ObjectT]] = attrib(
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
            paths=(
                (object_, path.copy_remapping_objects(object_mapping))
                for (object_, path) in self.paths.items()
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
