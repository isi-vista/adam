from typing import Generic

from attr import attrs, attrib
from immutablecollections import (
    ImmutableSetMultiDict,
    immutablesetmultidict,
    ImmutableSet,
    immutableset,
)
from immutablecollections.converter_utils import (
    _to_immutablesetmultidict,
    _to_immutableset,
)

from adam.ontology.phase1_spatial_relations import SpatialPath
from adam.relation import ObjectT, Relation


@attrs(frozen=True)
class DuringAction(Generic[ObjectT]):
    paths: ImmutableSetMultiDict[ObjectT, SpatialPath[ObjectT]] = attrib(
        converter=_to_immutablesetmultidict, default=immutablesetmultidict(), kw_only=True
    )
    at_some_point: ImmutableSet[Relation[ObjectT]] = attrib(
        converter=_to_immutableset, default=immutableset(), kw_only=True
    )
    continuously: ImmutableSet[Relation[ObjectT]] = attrib(
        converter=_to_immutableset, default=immutableset(), kw_only=True
    )
