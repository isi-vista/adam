from abc import ABC
from typing import Union

from attr import attrib, attrs
from attr.validators import in_, instance_of
from immutablecollections import ImmutableSet, immutableset
from immutablecollections.converter_utils import _to_immutableset
from vistautils.preconditions import check_arg
from vistautils.range import Range

from adam.ontology import OntologyNode, Region, IN_REGION
from adam.perception import PerceptualRepresentationFrame


@attrs(slots=True, frozen=True, repr=False)
class DevelopmentalPrimitivePerceptionFrame(PerceptualRepresentationFrame):
    r"""
    A static snapshot of a `Situation` based on developmentally-motivated perceptual primitives.

    This represents a situation as

    - a set of `ObjectPerception`\ s, with one corresponding to each
      object in the scene (e.g. a ball, Mom, Dad, etc.)
    - a set of `PropertyPerception`\ s which associate a `ObjectPerception` with perceived
      properties
      of various sort (e.g. color, sentience, etc.)
    - a set of `RelationPerception`\ s which describe the learner's perception of how two
      `ObjectPerception`\ s are related.

    This is the default perceptual representation for at least the first phase of the ADAM project.
    """
    perceived_objects: ImmutableSet["ObjectPerception"] = attrib(
        converter=_to_immutableset, default=immutableset()
    )
    r"""
    a set of `ObjectPerception`\ s, with one corresponding to each
      object in the scene (e.g. a ball, Mom, Dad, etc.)
    """
    property_assertions: ImmutableSet["PropertyPerception"] = attrib(
        converter=_to_immutableset, default=immutableset()
    )
    r"""
    a set of `PropertyPerception`\ s which associate a `ObjectPerception` with perceived properties
      of various sort (e.g. color, sentience, etc.)
    """
    relations: ImmutableSet["RelationPerception"] = attrib(
        converter=_to_immutableset, default=immutableset()
    )
    r"""
    a set of `RelationPerception`\ s which describe the learner's perception of how two 
      `ObjectPerception`\ s are related.
      
    Symmetric relations should be included as two separate relations, one in each direction.            
    """


@attrs(slots=True, frozen=True, repr=False)
class ObjectPerception:
    r"""
    The learner's perception of a particular object.

    This object pretty much just represents the object's existence; its attributes are handled via
    `PropertyPerception`\ s.
    """
    debug_handle: str = attrib(validator=instance_of(str))
    """
    A human-readable string associated with this object.
    
    It is for debugging use only and should not be accessed by any algorithms.
    """

    def __repr__(self) -> str:
        return self.debug_handle


@attrs(slots=True, frozen=True, repr=False)
class PropertyPerception(ABC):
    """
    A learner's perception that the *perceived_object* possesses a certain property.

    The particular property is specified in a sub-class dependent way.
    """

    perceived_object = attrib(validator=instance_of(ObjectPerception))


@attrs(slots=True, frozen=True, repr=False)
class RelationPerception:
    """
    A learner's perecption that two objects, *arg1* and *arg2* have a relation of type
    *relation_type* with one another.

    *arg2* will be a `Region` instead of an `ObjectPerception` if and only if
    the *relation_type* is `IN_REGION`.
    """

    relation_type: OntologyNode = attrib(validator=instance_of(OntologyNode))
    arg1: ObjectPerception = attrib(validator=instance_of(ObjectPerception))
    # for type ignore see
    # https://github.com/isi-vista/adam/issues/144
    arg2: Union[ObjectPerception, Region[ObjectPerception]] = attrib(
        validator=instance_of(
            (ObjectPerception, Region[ObjectPerception])  # type: ignore
        )
    )

    def __attrs_post_init__(self) -> None:
        check_arg(
            not isinstance(self.arg2, ObjectPerception) or self.relation_type == IN_REGION
        )

    def __repr__(self) -> str:
        return f"{self.relation_type}({self.arg1}, {self.arg2})"


@attrs(slots=True, frozen=True, repr=False)
class HasBinaryProperty(PropertyPerception):
    """
    A learner's perception that *perceived_object* possesses the given *flag_property*.
    """

    binary_property = attrib(validator=instance_of(OntologyNode))

    def __repr__(self) -> str:
        return f"hasProperty({self.perceived_object}, {self.binary_property})"


@attrs(slots=True, frozen=True, repr=False)
class RgbColorPerception:
    """
    A perceived color.
    """

    red: int = attrib(validator=in_(Range.closed(0, 255)))
    green: int = attrib(validator=in_(Range.closed(0, 255)))
    blue: int = attrib(validator=in_(Range.closed(0, 255)))

    def __repr__(self) -> str:
        """
        We represent colors by hex strings because these are easy to visualize using web tools.
        Returns:

        """
        return f"#{self.red:02x}{self.green:02x}{self.blue:02x}"


@attrs(slots=True, frozen=True, repr=False)
class HasColor(PropertyPerception):
    """
    A learner's perception that *perceived_object* has the `RgbColorPerception` *color*.
    """

    color = attrib(validator=instance_of(RgbColorPerception))

    def __repr__(self) -> str:
        return f"hasColor({self.perceived_object}, {self.color})"
