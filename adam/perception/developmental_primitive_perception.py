from abc import ABC
from typing import Sequence, Union

from attr import attrib, attrs
from attr.validators import in_, instance_of
from immutablecollections import ImmutableSet, immutableset
from immutablecollections.converter_utils import _to_immutableset
from vistautils.preconditions import check_arg
from vistautils.range import Range

from adam.axes import AxesInfo
from adam.ontology import OntologyNode
from adam.perception import ObjectPerception, PerceptualRepresentationFrame
from adam.relation import Relation, flatten_relations


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


_VALID_COLOR_RANGE = Range.closed(0, 255)


@attrs(slots=True, frozen=True, repr=False, eq=True)
class RgbColorPerception:
    """
    A perceived color.
    """

    red: int = attrib(validator=in_(_VALID_COLOR_RANGE))
    green: int = attrib(validator=in_(_VALID_COLOR_RANGE))
    blue: int = attrib(validator=in_(_VALID_COLOR_RANGE))

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


@attrs(slots=True, frozen=True, repr=False, eq=True)
class RgbColorSpaceBox:
    """
    A box-like region in RGB color space.

    This can be used as a simple way to represent ranges of colors.
    """

    red_range: attrib(validator=instance_of(Range))
    green_range: attrib(validator=instance_of(Range))
    blue_range: attrib(validator=instance_of(Range))

    def __attrs_post_init__(self) -> None:
        check_arg(_VALID_COLOR_RANGE.encloses(self.red_range))
        check_arg(_VALID_COLOR_RANGE.encloses(self.green_range))
        check_arg(_VALID_COLOR_RANGE.encloses(self.blue_range))

    @staticmethod
    def box_around(
        color: RgbColorPerception, *, taxicab_radius: float
    ) -> "RgbColorSpaceBox":
        return RgbColorSpaceBox(
            Range.closed(
                max(0, color.red - taxicab_radius), min(255, color.red + taxicab_radius)
            ),
            Range.closed(
                max(0, color.green - taxicab_radius),
                min(255, color.green + taxicab_radius),
            ),
            Range.closed(
                max(0, color.blue - taxicab_radius), min(255, color.blue + taxicab_radius)
            ),
        )

    def encloses(self, other: "RgbColorSpaceBox") -> bool:
        return (
            self.red_range.encloses(other.red_range)
            and self.green_range.encloses(other.green_range)
            and self.blue_range.encloses(other.blue_range)
        )

    def minimal_enclosing_box(
        self, colors: Sequence[Union[RgbColorPerception, "RgbColorSpaceBox"]]
    ) -> "RgbColorSpaceBox":
        reds = []
        greens = []
        blues = []

        for color in colors:
            if isinstance(color, RgbColorPerception):
                reds.append(color.red)
                greens.append(color.green)
                blues.append(color.blue)
            else:
                reds.append(color.red_range.upper_bound, color.red_range.lower_bound)
                greens.append(
                    color.green_range.upper_bound, color.green_range.lower_bound
                )
                blues.append(color.blue_range.upper_bound, color.blue_range.lower_bound)

        return RgbColorSpaceBox(
            red_range=Range.closed(min(reds), max(reds)),
            green_range=Range.closed(min(greens), max(greens)),
            blue_range=Range.closed(min(blues), max(blues)),
        )

    def fraction_of_color_space_covered(self) -> float:
        """
        The fraction of the RGB color space this cube covers.

        Note that this can give unintuitive results - all shades of pure red
        ( any value for red, 0 for green and blue)
        covered 0% of the color space!
        """
        return (
            (self.red_range.upper_bound - self.red_range.lower_bound)
            * (self.green_range.upper_bound - self.green_range.lower_bound)
            * (self.blue_range.upper_bound - self.blue_range.lower_bound)
        ) / (255 * 255 * 255)

    def __contains__(self, rgb: RgbColorPerception) -> bool:
        return (
            rgb.red in self.red_range
            and rgb.green in self.green_range
            and rgb.blue in self.blue_range
        )

    def __repr__(self) -> str:
        return f"r{self.red_range}g{self.green_range}b{self.blue_range}"


@attrs(slots=True, frozen=True, repr=False)
class HasColor(PropertyPerception):
    """
    A learner's perception that *perceived_object* has the `RgbColorPerception` *color*.
    """

    color = attrib(validator=instance_of(RgbColorPerception))

    def __repr__(self) -> str:
        return f"hasColor({self.perceived_object}, {self.color})"
