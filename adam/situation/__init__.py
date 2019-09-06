"""
Structures for describing situations in the world at an abstracted, human-friendly level.
"""
from abc import ABC
from typing import Mapping, Optional

from attr import attrs, attrib
from attr.validators import instance_of, optional
from immutablecollections import (
    immutableset,
    ImmutableSet,
    immutabledict,
    immutablesetmultidict,
    ImmutableSetMultiDict,
)

# noinspection PyProtectedMember
from immutablecollections.converter_utils import (
    _to_immutableset,
    _to_immutabledict,
    _to_immutablesetmultidict,
)

from adam.math_3d import Point
from adam.ontology import OntologyProperty, OntologyNode, Ontology


class Situation(ABC):
    r"""
    A situation is a high-level representation of a configuration of objects, possibly including
    changes in the states of objects across time.

    An example situation might represent
    a person holding a toy truck and then putting it on a table.

    A curriculum is a sequence of `Situation`\ s.

    Situations are a high-level description intended to make it easy for human beings to specify
    curricula.

    Situations will be transformed into pairs of `PerceptualRepresentation`\ s and
    `LinguisticDescription`\ s for input to a `LanguageLearner`
    by `PerceptualRepresentationGenerator`\ s and `LanguageGenerator`\ s, respectively.
    """


@attrs(frozen=True)
class BagOfFeaturesSituationRepresentation(Situation):
    r"""
    Represents a `Situation` as an unstructured set of features.

    For testing purposes only.
    """
    features: ImmutableSet[str] = attrib(converter=immutableset, default=immutableset())
    """
    The set of string features which describes this situation.
    """


class SituationNode:
    """
    A general interface encompassing `SituationObject`, `SituationRelation`, and `SituationAction`.

    Currently its only function is to help with type checking.
    """


@attrs(frozen=True, slots=True, hash=None, cmp=False, repr=False)
class SituationObject(SituationNode):
    """
    An object present in some situation.

    Every object must refer to an `OntologyNode` linking it to a type in an ontology.

    Unlike most of our classes, `SituationObject` has *id*-based hashing and equality.  This is
    because two objects with identical properties are nonetheless distinct.
    """

    ontology_node: Optional[OntologyNode] = attrib(
        validator=optional(instance_of(OntologyNode)), default=None
    )
    """
    The `OntologyNode` specifying the type of thing this object is.
    """
    properties: ImmutableSet[OntologyProperty] = attrib(
        converter=_to_immutableset, default=immutableset()
    )
    r"""
    The `OntologyProperty`\ s this object has.
    """

    def __attrs_post_init__(self) -> None:
        # disabled warning below is due to a PyCharm bug
        # noinspection PyTypeChecker
        for property_ in self.properties:
            if not isinstance(property_, OntologyProperty):
                raise ValueError(
                    f"Situation object property {property_} is not an "
                    f"OntologyProperty"
                )

    def __repr__(self) -> str:
        if self.properties:
            additional_properties = ", ".join(repr(prop) for prop in self.properties)
            additional_properties_string = f"[{additional_properties}]"
        else:
            additional_properties_string = ""
        if self.ontology_node:
            handle_string = self.ontology_node.handle
        else:
            handle_string = "???"

        return f"{handle_string}{additional_properties_string}"


@attrs(frozen=True, slots=True)
class LocatedObjectSituation(Situation):
    r"""
    A representation of a `Situation` as a set of objects located at particular `Point`\ s.
    """

    objects_to_locations: Mapping[SituationObject, Point] = attrib(
        converter=_to_immutabledict, default=immutabledict()
    )
    r"""
    A mapping of `SituationObject`\ s to `Point`\ s giving their locations.
    """


@attrs(frozen=True, slots=True, repr=False)
class SituationRelation(SituationNode):
    """
    A relationship which holds between two objects in a `Situation`.
    """

    relation_type: OntologyNode = attrib(validator=instance_of(OntologyNode))
    first_slot: SituationObject = attrib(validator=instance_of(SituationObject))
    second_slot: SituationObject = attrib(validator=instance_of(SituationObject))

    def __repr__(self) -> str:
        return f"{self.relation_type}({self.first_slot}, {self.second_slot})"


@attrs(frozen=True, slots=True, repr=False)
class SituationAction(SituationNode):
    """
    An action occurring in a `Situation`.
    """

    action_type: OntologyNode = attrib(validator=instance_of(OntologyNode))
    argument_roles_to_fillers: ImmutableSetMultiDict[
        OntologyNode, SituationNode
    ] = attrib(converter=_to_immutablesetmultidict, default=immutablesetmultidict())
    r"""
    A mapping of semantic roles (given as `OntologyNode`\ s) to their fillers.

    There may be multiple fillers for the same semantic role 
    (e.g. conjoined arguments).
    """

    def __repr__(self) -> str:
        return f"{self.action_type}({self.argument_roles_to_fillers})"


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
