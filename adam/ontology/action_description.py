from typing import Mapping, Union

from attr import attrs, attrib
from immutablecollections import (
    ImmutableSet,
    immutableset,
    ImmutableSetMultiDict,
    immutablesetmultidict,
    ImmutableDict,
    immutabledict,
)
from immutablecollections.converter_utils import _to_immutableset, _to_immutabledict

from adam.ontology import OntologyNode
from adam.situation import SituationObject, SituationRelation


@attrs(frozen=True, slots=True, auto_attribs=True)
class ActionDescriptionFrame:
    # the keys here should be semantic roles
    roles_to_entities: ImmutableDict[OntologyNode, SituationObject] = attrib(
        converter=_to_immutabledict, default=immutabledict()
    )
    entities_to_roles: ImmutableSetMultiDict[SituationObject, OntologyNode] = attrib(
        init=False
    )

    @entities_to_roles.default
    def _init_entities_to_roles(
        self
    ) -> ImmutableSetMultiDict[SituationObject, OntologyNode]:
        return immutablesetmultidict(
            (entity, role) for role, entity in self.roles_to_entities.items()
        )


@attrs(frozen=True, slots=True)
class ActionDescription:
    frames: ImmutableSet[ActionDescriptionFrame] = attrib(
        converter=_to_immutableset, default=immutableset(), kw_only=True
    )
    # Preconditions
    preconditions: ImmutableSet[SituationRelation] = attrib(
        converter=_to_immutableset, default=immutableset(), kw_only=True
    )
    # Postconditions
    postconditions: ImmutableSet[SituationRelation] = attrib(
        converter=_to_immutableset, default=immutableset(), kw_only=True
    )
