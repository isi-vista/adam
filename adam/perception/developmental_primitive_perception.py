from abc import ABC

from attr import attrib, attrs
from attr.validators import in_, instance_of
from immutablecollections import ImmutableSet, immutableset
from immutablecollections.converter_utils import _to_immutableset
from vistautils.range import Range

from adam.ontology import Ontology, OntologyNode
from adam.ontology.phase1_ontology import RECOGNIZED_PARTICULAR
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
    """

    relation_type: OntologyNode = attrib(validator=instance_of(OntologyNode))
    arg1: ObjectPerception = attrib(validator=instance_of(ObjectPerception))
    arg2: ObjectPerception = attrib(validator=instance_of(ObjectPerception))

    def __repr__(self) -> str:
        return f"{self.relation_type}({self.arg1}, {self.arg2})"


@attrs(slots=True, frozen=True, repr=False)
class HasBinaryProperty(PropertyPerception):
    """
    A learner's perception that *perceived_object* possesses the given *flag_property*.
    """

    binary_property = attrib(validator=instance_of(OntologyNode))

    def __repr__(self) -> str:
        return f"hasProperty({self.perceived_object}, {self.binary_property}"


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


@attrs(slots=True, frozen=True, repr=False)
class IsRecognizedParticular(PropertyPerception):
    """
    A learner's perception that the *perceived_object* is some particular instance that it knows,
    given by an `OntologyNode` which must have the `RECOGNIZED_PARTICULAR` property.

    The canonical examples here are "Mom" and "Dad".

    An `Ontology` must be provided to verify that the node is a `RECOGNIZED_PARTICULAR`.
    """

    particular_ontology_node: OntologyNode = attrib(validator=instance_of(OntologyNode))
    ontology: Ontology = attrib(validator=instance_of(Ontology), kw_only=True)

    def __attrs_post_init__(self) -> None:
        if not self.ontology.has_all_properties(
            self.particular_ontology_node, [RECOGNIZED_PARTICULAR]
        ):
            raise RuntimeError(
                "The learner can only perceive the ontology node of an object "
                "if it is a recognized particular (e.g. Mom, Dad)"
            )

    def __repr__(self) -> str:
        return f"recognizedAs({self.perceived_object}, {self.particular_ontology_node})"
