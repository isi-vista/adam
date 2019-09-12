from attr import attrs, attrib
from attr.validators import instance_of
from immutablecollections import ImmutableSet, immutableset
from immutablecollections.converter_utils import _to_immutableset

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
    objects: ImmutableSet[SituationObject] = attrib(
        converter=_to_immutableset, default=immutableset()
    )
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
