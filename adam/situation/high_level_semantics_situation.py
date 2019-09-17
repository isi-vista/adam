from attr import attrs, attrib
from attr.validators import instance_of
from immutablecollections import ImmutableSet, immutableset
from immutablecollections.converter_utils import _to_immutableset
from more_itertools import flatten
from vistautils.preconditions import check_arg

from adam.ontology import Region
from adam.ontology.ontology import Ontology
from adam.situation import Situation, SituationObject, SituationRelation, SituationAction


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
    relations: ImmutableSet[SituationRelation] = attrib(
        converter=_to_immutableset, default=immutableset()
    )
    """
    The relations which hold in this `Situation`.
    
    It is not necessary to state every relationship which holds through.
    Rather this should contain the salient relationships
    which should be expressed in the linguistic description.
    """
    actions: ImmutableSet[SituationAction] = attrib(
        converter=_to_immutableset, default=immutableset()
    )
    """
    The actions occurring in this `Situation`
    """

    def __attrs_post_init__(self) -> None:
        check_arg(self.objects, "A situation must contain at least one object")
        for relation in self.relations:
            if (
                isinstance(relation.second_slot, Region)
                and relation.second_slot.reference_object not in self.objects
            ):
                raise RuntimeError(
                    "Any object referred to by a region must be included in the "
                    "set of situation objects."
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

    def __repr__(self) -> str:
        # TODO: the way we currently repr situations doesn't handle multiple nodes
        # of the same ontology type well.  We'd like to use subscripts (_0, _1)
        # to distinguish them, which requires pulling all the repr logic up to this
        # level and not delegating to the reprs of the sub-objects.
        # https://github.com/isi-vista/adam/issues/62
        lines = ["{"]
        lines.extend(f"\t{obj!r}" for obj in self.objects)
        lines.extend(f"\t{relation!r}" for relation in self.relations)
        lines.extend(f"\t{action!r}" for action in self.actions)
        lines.append("}")
        return "\n".join(lines)
