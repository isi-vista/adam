from abc import ABC

from adam.axes import AxesInfo
from adam.ontology import OntologyNode
from adam.perception import ObjectPerception, PerceptualRepresentationFrame
from adam.relation import Relation, flatten_relations
from attr import attrib, attrs
from attr.validators import in_, instance_of
from immutablecollections import ImmutableSet, immutableset
from immutablecollections.converter_utils import _to_immutableset
from vistautils.preconditions import check_arg
from vistautils.range import Range


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
    - a set of `Relation`\ s which describe the learner's perception of how two
      `ObjectPerception`\ s are related.

    This is the default perceptual representation for at least the first phase of the ADAM project.
    """
    perceived_objects: ImmutableSet["ObjectPerception"] = attrib(
        converter=_to_immutableset
    )
    r"""
    a set of `ObjectPerception`\ s, with one corresponding to each
      object in the scene (e.g. a ball, Mom, Dad, etc.)
    """
    axis_info: AxesInfo["ObjectPerception"] = attrib(
        # instance default okay because immutable
        validator=instance_of(AxesInfo),  # type: ignore
        kw_only=True,
        default=AxesInfo(),
    )
    property_assertions: ImmutableSet["PropertyPerception"] = attrib(
        converter=_to_immutableset, default=immutableset(), kw_only=True
    )
    r"""
    a set of `PropertyPerception`\ s which associate a `ObjectPerception` with perceived properties
      of various sort (e.g. color, sentience, etc.)
    """
    relations: ImmutableSet[Relation["ObjectPerception"]] = attrib(  # type: ignore
        converter=flatten_relations, default=immutableset(), kw_only=True
    )
    r"""
    a set of `Relation`\ s which describe the learner's perception of how two 
      `ObjectPerception`\ s are related.
      
    Symmetric relations should be included as two separate relations, one in each direction.            
    """

    def __attrs_post_init__(self) -> None:
        for relation in self.relations:
            check_arg(
                not relation.negated,
                "Negated relations cannot appear in perceptual "
                "representations but got %s",
                (relation,),
            )


@attrs(slots=True, frozen=True, repr=False)
class PropertyPerception(ABC):
    """
    A learner's perception that the *perceived_object* possesses a certain property.

    The particular property is specified in a sub-class dependent way.
    """

    perceived_object = attrib(validator=instance_of(ObjectPerception))


@attrs(slots=True, frozen=True, repr=False)
class HasBinaryProperty(PropertyPerception):
    """
    A learner's perception that *perceived_object* possesses the given *flag_property*.
    """

    binary_property = attrib(validator=instance_of(OntologyNode))

    def __repr__(self) -> str:
        return f"hasProperty({self.perceived_object}, {self.binary_property})"


@attrs(slots=True, frozen=True, repr=False, eq=True)
class RgbColorPerception:
    """
    A perceived color.
    """

    red: int = attrib(validator=in_(Range.closed(0, 255)))
    green: int = attrib(validator=in_(Range.closed(0, 255)))
    blue: int = attrib(validator=in_(Range.closed(0, 255)))

    def inverse(self) -> "RgbColorPerception":
        return RgbColorPerception(255 - self.red, 255 - self.green, 255 - self.blue)

    @property
    def hex(self) -> str:
        return f"#{self.red:02x}{self.green:02x}{self.blue:02x}"

    def __repr__(self) -> str:
        """
        We represent colors by hex strings because these are easy to visualize using web tools.
        Returns:

        """
        return self.hex


@attrs(slots=True, frozen=True, repr=False)
class HasColor(PropertyPerception):
    """
    A learner's perception that *perceived_object* has the `RgbColorPerception` *color*.
    """

    color = attrib(validator=instance_of(RgbColorPerception))

    def __repr__(self) -> str:
        return f"hasColor({self.perceived_object}, {self.color})"
