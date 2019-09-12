from typing import Mapping, Union

from attr import attrs, attrib
from immutablecollections import ImmutableSet, immutableset
from immutablecollections.converter_utils import _to_immutableset

from adam.ontology import OntologyNode
from adam.situation import SituationObject, SituationRelation


@attrs(frozen=True, slots=True, auto_attribs=True)
class ActionDescriptionFrame:
    # the keys here should be semantic roles
    roles_to_entities: Mapping[OntologyNode, SituationObject]


@attrs(frozen=True, slots=True)
class ActionDescription:
    frames: ImmutableSet[ActionDescriptionFrame]
    # Preconditions
    preconditions: ImmutableSet[SituationRelation] = attrib(
        converter=_to_immutableset, default=immutableset(), kw_only=True
    )
    # Postconditions
    postconditions: ImmutableSet[SituationRelation] = attrib(
        converter=_to_immutableset, default=immutableset(), kw_only=True
    )