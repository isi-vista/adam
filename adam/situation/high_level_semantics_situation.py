from attr import attrib, attrs
from attr.validators import instance_of
from immutablecollections import ImmutableSet, immutableset
from immutablecollections.converter_utils import _to_immutableset
from more_itertools import flatten
from vistautils.preconditions import check_arg

from adam.ontology.ontology import Ontology
from adam.ontology.phase1_spatial_relations import Region
from adam.relation import Relation, flatten_relations
from adam.situation import Situation, SituationAction, SituationObject


@attrs(frozen=True, slots=True, repr=False)
class HighLevelSemanticsSituation(Situation):
    """
    A human-friendly representation of `Situation`.
    """

    ontology: Ontology = attrib(validator=instance_of(Ontology))
    """
    What `Ontology` items from the objects, relations, and actions 
    in this `Situation` will come from.
    """
    objects: ImmutableSet[SituationObject] = attrib(converter=_to_immutableset)
    """
    All the objects present in a `Situation`.
    """
    persisting_relations: ImmutableSet[Relation[SituationObject]] = attrib(
        converter=flatten_relations, default=immutableset()
    )
    """
    The relations which hold in this `Situation`,
    both before and after any actions which occur.
    
    It is not necessary to state every relationship which holds in a situation.
    Rather this should contain the salient relationships
    which should be expressed in the linguistic description.
    
    Do not specify those relations here which are *implied* by any actions which occur.
    Those are handled automatically. 
    """
    before_action_relations: ImmutableSet[Relation[SituationObject]] = attrib(
        converter=flatten_relations, default=immutableset()
    )
    """
    The relations which hold in this `Situation`,
    before, but not necessarily after, any actions which occur.
    
    It is not necessary to state every relationship which holds in a situation.
    Rather this should contain the salient relationships
    which should be expressed in the linguistic description.
    
    Do not specify those relations here which are *implied* by any actions which occur.
    Those are handled automatically. 
    """
    after_action_relations: ImmutableSet[Relation[SituationObject]] = attrib(
        converter=flatten_relations, default=immutableset()
    )
    """
    The relations which hold in this `Situation`,
    after, but not necessarily before, any actions which occur.

    It is not necessary to state every relationship which holds in a situation.
    Rather this should contain the salient relationships
    which should be expressed in the linguistic description.

    Do not specify those relations here which are *implied* by any actions which occur.
    Those are handled automatically. 
    """
    actions: ImmutableSet[SituationAction] = attrib(
        converter=_to_immutableset, default=immutableset()
    )
    """
    The actions occurring in this `Situation`
    """

    def __attrs_post_init__(self) -> None:
        check_arg(self.objects, "A situation must contain at least one object")
        for relation in self.persisting_relations:
            if not isinstance(relation.first_slot, SituationObject) or not isinstance(
                relation.second_slot, (SituationObject, Region)
            ):
                raise RuntimeError(
                    f"Relation fillers for situations must be situation objects "
                    f"but got {relation}"
                )
            if (
                isinstance(relation.second_slot, Region)
                and relation.second_slot.reference_object not in self.objects
            ):
                raise RuntimeError(
                    f"Any object referred to by a region must be included in the "
                    f"set of situation objects but region {relation.second_slot}"
                    f" with situation objects {self.objects}"
                )
        for action in self.actions:
            for action_role_filler in flatten(
                action.argument_roles_to_fillers.value_groups()
            ):
                if (
                    isinstance(action_role_filler, SituationObject)
                    and action_role_filler not in self.objects
                ):
                    raise RuntimeError(
                        "Any object filling a semantic role must be included in the "
                        "set of situation objects."
                    )
                elif (
                    isinstance(action_role_filler, Region)
                    and action_role_filler.reference_object not in self.objects
                ):
                    raise RuntimeError(
                        "Any object referred to by a region must be included in the "
                        "set of situation objects."
                    )
        if not self.actions and (
            self.before_action_relations or self.after_action_relations
        ):
            raise RuntimeError(
                "Cannot specify relations to hold before or after actions "
                "if there are no actions"
            )

    def __repr__(self) -> str:
        # TODO: the way we currently repr situations doesn't handle multiple nodes
        # of the same ontology type well.  We'd like to use subscripts (_0, _1)
        # to distinguish them, which requires pulling all the repr logic up to this
        # level and not delegating to the reprs of the sub-objects.
        # https://github.com/isi-vista/adam/issues/62
        lines = ["{"]
        lines.extend(f"\t{obj!r}" for obj in self.objects)
        lines.extend(f"\t{relation!r}" for relation in self.persisting_relations)
        lines.extend(f"\t{action!r}" for action in self.actions)
        lines.append("}")
        return "\n".join(lines)
