from typing import Optional

from attr import attrib, attrs
from attr.validators import instance_of, optional
from immutablecollections import ImmutableSet, immutableset
from immutablecollections.converter_utils import _to_immutableset

from adam.geon import Geon
from adam.ontology import OntologyNode
from adam.relation import Relation


@attrs(frozen=True, slots=True, repr=False)
class ObjectStructuralSchema:
    r"""
    A hierarchical representation of the internal structure of some type of object.

    An `ObjectStructuralSchema` represents the general pattern of the structure of an object,
    rather than the structure of any particular object
    (e.g. people in general, rather than a particular person).

    For example a person's body is made up of a head, torso, left arm, right arm, left leg, and
    right leg. These sub-objects have various relations to one another
    (e.g. the head is above and supported by the torso).

    Declaring an `ObjectStructuralSchema` can be verbose;
     see `Relation`\ s for additional tips on how to make this more compact.
    """

    ontology_node: OntologyNode = attrib(validator=instance_of(OntologyNode))
    """
    The `OntologyNode` this `ObjectStructuralSchema` represents the structure of.
    """
    sub_objects: ImmutableSet["SubObject"] = attrib(
        converter=_to_immutableset, default=immutableset()
    )
    r"""
    The component parts which make up an object of the type *parent_object*.
    
    These `SubObject`\ s themselves wrap `ObjectStructuralSchema`\ s 
    and can therefore themselves have complex internal structure.
    """
    sub_object_relations: ImmutableSet[Relation["SubObject"]] = attrib(
        converter=_to_immutableset, default=immutableset()
    )
    r"""
    A set of `Relation`\ s which define how the `SubObject`\ s relate to one another. 
    """
    geon: Optional[Geon] = attrib(
        validator=optional(instance_of(Geon)), default=None, kw_only=True
    )


# need cmp=False to keep otherwise identical sub-components distinct
# (e.g. left arm, right arm)
@attrs(frozen=True, slots=True, repr=False, cmp=False)
class SubObject:
    r"""
    A sub-component of a generic type of object.

    This is for use only in constructing `ObjectStructuralSchema`\ ta.
    """

    schema: ObjectStructuralSchema = attrib()
    """
    The `ObjectStructuralSchema` describing the internal structure of this sub-component.
    
    For example, an ARM is a sub-component of a PERSON, but ARM itself has a complex structure
    (e.g. it includes a hand)
    """
