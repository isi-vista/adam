from itertools import chain
from typing import List, Optional

from attr import attrib, attrs
from attr.validators import instance_of, optional
from immutablecollections import (
    ImmutableDict,
    ImmutableSet,
    ImmutableSetMultiDict,
    immutabledict,
    immutableset,
    immutablesetmultidict,
)
from immutablecollections.converter_utils import (
    _to_immutabledict,
    _to_immutablesetmultidict,
)

from adam.ontology import OntologyNode
from adam.ontology.during import DuringAction
from adam.relation import Relation, flatten_relations
from adam.situation import SituationObject

# really this should become its own class and not piggy-back
# on SituationObject
ActionDescriptionVariable = SituationObject


@attrs(frozen=True, slots=True, auto_attribs=True)
class ActionDescriptionFrame:
    # the keys here should be semantic roles
    roles_to_variables: ImmutableDict[OntologyNode, ActionDescriptionVariable] = attrib(
        converter=_to_immutabledict, default=immutabledict()
    )
    variables_to_roles: ImmutableSetMultiDict[
        ActionDescriptionVariable, OntologyNode
    ] = attrib(init=False)
    semantic_roles: ImmutableSet[OntologyNode] = attrib(init=False)

    @variables_to_roles.default
    def _init_entities_to_roles(
        self
    ) -> ImmutableSetMultiDict[SituationObject, OntologyNode]:
        return immutablesetmultidict(
            (entity, role) for role, entity in self.roles_to_variables.items()
        )

    @semantic_roles.default
    def _init_semantic_roles(self) -> ImmutableSet[OntologyNode]:
        return immutableset(self.roles_to_variables.keys())


@attrs(frozen=True, slots=True)
class ActionDescription:
    frame: ActionDescriptionFrame = attrib(
        validator=instance_of(ActionDescriptionFrame), kw_only=True
    )
    # nested generic in optional seems to be confusing mypy
    during: Optional[DuringAction[ActionDescriptionVariable]] = attrib(  # type: ignore
        validator=optional(instance_of(DuringAction)), default=None, kw_only=True
    )
    # conditions which hold both before and after the action
    enduring_conditions: ImmutableSet[Relation[ActionDescriptionVariable]] = attrib(
        converter=flatten_relations, default=immutableset(), kw_only=True
    )
    # Preconditions
    preconditions: ImmutableSet[Relation[ActionDescriptionVariable]] = attrib(
        converter=flatten_relations, default=immutableset(), kw_only=True
    )
    # Postconditions
    postconditions: ImmutableSet[Relation[ActionDescriptionVariable]] = attrib(
        converter=flatten_relations, default=immutableset(), kw_only=True
    )
    # Asserted properties of objects in action
    asserted_properties: ImmutableSetMultiDict[
        ActionDescriptionVariable, OntologyNode
    ] = attrib(
        converter=_to_immutablesetmultidict, default=immutablesetmultidict(), kw_only=True
    )
    auxiliary_variables: ImmutableSet[ActionDescriptionVariable] = attrib(init=False)
    """
    These are variables which do not occupy semantic roles 
    but are are still referred to by conditions, paths, etc.
    An example would be the container for liquid for a "drink" action.
    """

    def __attrs_post_init__(self) -> None:
        for relation in chain(
            self.enduring_conditions, self.preconditions, self.postconditions
        ):
            if not isinstance(relation, Relation):
                raise RuntimeError(
                    f"All conditions on an action description ought to be Relations "
                    f"but got {relation}"
                )

    @auxiliary_variables.default
    def _init_auxiliary_variables(self):
        auxiliary_variables: List[ActionDescriptionVariable] = []
        if self.during:
            self.during.accumulate_referenced_objects(auxiliary_variables)
        for relation in chain(
            self.enduring_conditions, self.preconditions, self.postconditions
        ):
            relation.accumulate_referenced_objects(auxiliary_variables)
        return immutableset(
            variable
            for variable in auxiliary_variables
            if variable not in self.frame.variables_to_roles
        )
