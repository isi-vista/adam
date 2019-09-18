from attr import attrib, attrs
from immutablecollections import (
    ImmutableDict,
    ImmutableSet,
    ImmutableSetMultiDict,
    immutabledict,
    immutableset,
    immutablesetmultidict,
)
from immutablecollections.converter_utils import _to_immutabledict, _to_immutableset

from adam.ontology import OntologyNode
from adam.relation import Relation
from adam.situation import SituationObject


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
    # Frames: a set of action description frames each of which carries information about the mappings
    # between general semantic roles and to entities specific to the action
    # e.g. AGENT -> _PUT_AGENT (PUT_AGENT would carry action-specific info, and 'mom ' would be an instance of it.
    frames: ImmutableSet[ActionDescriptionFrame] = attrib(
        converter=_to_immutableset, default=immutableset(), kw_only=True
    )
    # conditions which hold both before and after the action
    enduring_conditions: ImmutableSet[Relation[SituationObject]] = attrib(
        converter=_to_immutableset, default=immutableset(), kw_only=True
    )
    # Preconditions
    preconditions: ImmutableSet[Relation[SituationObject]] = attrib(
        converter=_to_immutableset, default=immutableset(), kw_only=True
    )
    # Postconditions
    postconditions: ImmutableSet[Relation[SituationObject]] = attrib(
        converter=_to_immutableset, default=immutableset(), kw_only=True
    )
