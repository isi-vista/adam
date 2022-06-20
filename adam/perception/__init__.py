r"""
This module provides classes related to the perceptual primitive representation
used to describe `Situation`\ s from the point-of-view of `TopLevelLanguageLearner`\ s.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Generic, Optional, Tuple, TypeVar, Mapping, Any

from attr import attrib, attrs
from attr.validators import instance_of, optional
from immutablecollections import ImmutableSet, immutableset
from immutablecollections.converter_utils import _to_immutableset
from vistautils.preconditions import check_arg

from adam.geon import Geon, MaybeHasGeon
from adam.axes import Axes, HasAxes, WORLD_AXES, LEARNER_AXES
from adam.math_3d import Point
from adam.ontology.during import DuringAction
from adam.ontology.phase1_spatial_relations import Region
from adam.random_utils import SequenceChooser
from adam.situation import LocatedObjectSituation, Situation

_SituationT = TypeVar("_SituationT", bound=Situation)


class MatchMode(Enum):
    OBJECT = "object"
    NON_OBJECT = "non_object"


@attrs(slots=True, frozen=True, repr=False, cache_hash=True)
class ObjectPerception(HasAxes, MaybeHasGeon):
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
    geon: Optional[Geon] = attrib(validator=optional(instance_of(Geon)), default=None)
    axes: Axes = attrib(validator=instance_of(Axes))

    def __attrs_post_init__(self) -> None:
        if self.geon:
            check_arg(self.geon.axes == self.axes)

    @axes.default
    def _init_axes(self) -> Axes:
        if self.geon:
            return self.geon.axes
        else:
            raise RuntimeError(
                "If a geon is not associated with an object perception, "
                "its axes must be specified explicitly"
            )

    def dot_label(self) -> str:
        # object perceptions have no content, so they are blank nodes
        return self.debug_handle

    def __repr__(self) -> str:
        return self.debug_handle


RegionPerception = Region[ObjectPerception]  # pylint:disable=invalid-name

GROUND_PERCEPTION = ObjectPerception("the ground", geon=None, axes=WORLD_AXES)
LEARNER_PERCEPTION = ObjectPerception("learner", geon=None, axes=LEARNER_AXES)


class PerceptualRepresentationFrame(ABC):
    r"""
    Represents a `TopLevelLanguageLearner`\ 's perception of some `Situation`\ at a single moment.

    One or more of these forms a `PerceptualRepresentation`.
    """


PerceptionT = TypeVar("PerceptionT", bound="PerceptualRepresentationFrame")
# second type variable is for use in static methods
_PerceptionT2 = TypeVar("_PerceptionT2", bound="PerceptualRepresentationFrame")


@attrs(frozen=True)
class PerceptualRepresentation(Generic[PerceptionT]):
    r"""
    A `TopLevelLanguageLearner`'s perception of a `Situation`
    as a sequence of perceptual representations of individual moments (*frames*).
    """

    frames: Tuple[PerceptionT, ...] = attrib(converter=tuple)
    """
    The frames making up the description of a situation.

    Usually for a static situation, this will be a single frame,
    but there could be two or three for complex actions.
    """
    # mypy is confused by the instance_of with a generic class
    during: Optional[DuringAction[ObjectPerception]] = attrib(  # type: ignore
        validator=optional(instance_of(DuringAction)), default=None, kw_only=True
    )
    simulated_actions_features: Optional[Mapping[str, Any]] = attrib(
        default=None,
        kw_only=True,
    )

    @staticmethod
    def single_frame(
        perception_frame: _PerceptionT2,
    ) -> "PerceptualRepresentation[_PerceptionT2]":
        """
        Convenience method for generating a `PerceptualRepresentation` which is a single frame.

        Args:
            perception_frame: a `PerceptualRepresentationFrame`

        Returns:
            A `PerceptualRepresentation` wrapping the provided frame.

        """
        return PerceptualRepresentation((perception_frame,))  # type: ignore

    def is_dynamic(self) -> bool:
        """
        Does this situation represent a changing state-of-affairs?
        """
        return len(self.frames) > 1


class PerceptualRepresentationGenerator(Generic[_SituationT, PerceptionT], ABC):
    r"""
    A strategy for generating `PerceptualRepresentation`\ s of `Situation` s.

    This is used when constructing curricula procedurally
    so that humans do not need to build perceptual representations by hand.
    """

    @abstractmethod
    def generate_perception(
        self, situation: _SituationT, chooser: SequenceChooser
    ) -> PerceptualRepresentation[PerceptionT]:
        """
        Generate a `PerceptualRepresentation` of a `Situation`.

        Args:
            situation: The `Situation` to represent.
            chooser: An optional `SequenceChooser` to be used for any required random choices. If
                     none is provided, an unspecified but deterministic source of "random" choice
                     is used.

        Returns:
            A `PerceptualRepresentation` of the `Situation`.
        """


@attrs(frozen=True, slots=True)
class BagOfFeaturesPerceptualRepresentationFrame(PerceptualRepresentationFrame):
    r"""
    Represents a `TopLevelLanguageLearner`'s perception of a `Situation` at a single moment
    as an unstructured set of features.

    For testing purposes only.
    """
    features: ImmutableSet[str] = attrib(converter=immutableset)
    """
    A set of string features describing the `Situation` .
    """


@attrs(frozen=True, slots=True)
class DummyVisualPerceptionFrame(PerceptualRepresentationFrame):
    r"""
    A visual representation made up of several objects represented by
    `DummyVisualPerceptionFrame.SingleObjectPerception`\ s.

    This is only for testing purposes.
    """
    object_perceptions: ImmutableSet[
        "DummyVisualPerceptionFrame.SingleObjectPerception"
    ] = attrib(converter=_to_immutableset, default=immutableset())
    r"""
    A set of perceptions of objects as `DummyVisualPerceptionFrame.SingleObjectPerception`\ s.
    """

    @attrs(frozen=True, slots=True)
    class SingleObjectPerception:
        """
        A visual representation of a `Situation` at a single moment
        as a string describing an object together with a location,
        with no other structure or properties.

        It simply says for e.g. a truck "this looks like a truck." and here is its (`Point`) location.

        This is only for testing purposes.
        """

        tag: str = attrib(validator=instance_of(str))
        """
        A simple string desription of an object
        """
        location: Point = attrib(validator=instance_of(Point))
        """
        The `Point` where the object is located.
        """


class DummyVisualPerceptionGenerator(
    PerceptualRepresentationGenerator[LocatedObjectSituation, DummyVisualPerceptionFrame]
):
    """
    Computes simple `PerceptualRepresentation` for a `LocatedObjectSituation`
    using the name of the type of each object plus its location to
    generate a `DummyVisualPerceptionFrame` .
    """

    def generate_perception(  # pylint:disable=unused-argument
        self, situation: LocatedObjectSituation, chooser: SequenceChooser
    ) -> PerceptualRepresentation[DummyVisualPerceptionFrame]:
        return PerceptualRepresentation.single_frame(
            DummyVisualPerceptionFrame(
                DummyVisualPerceptionFrame.SingleObjectPerception(
                    tag=obj.ontology_node.handle if obj.ontology_node else "unknown",
                    location=point,
                )
                for (obj, point) in situation.objects_to_locations.items()
            )
        )
