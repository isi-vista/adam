r"""
This module provides classes related to the perceptual primitive representation used to describe
`Situation`\ s from the point-of-view of `LanguageLearner`\ s.
"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Tuple

from attr import attrs, attrib
from attr.validators import instance_of
from immutablecollections import ImmutableSet, immutableset

from adam.language.language_generator import SituationT
from adam.math_3d import Point
from adam.random_utils import SequenceChooser, fixed_random_factory


class PerceptualRepresentationFrame(ABC):
    r"""
    Represents a `LanguageLearner`\ 's perception of some `Situation`\ at a single moment.

    One or more of these, paired with a `LinguisticDescription`\ , forms an observation that a
    `LanguageLearner` learns from.
    """


PerceptionT = TypeVar("_PerceptionT", bound=PerceptualRepresentationFrame)


@attrs(frozen=True)
class PerceptualRepresentation(Generic[PerceptionT]):
    """
    A learner's perception of a situation as a sequence of perceptual representations of
    individual moments.

    Usually for a static situation, this will be a single frame, but it could be two or
    three for complex actions.
    """

    frames: Tuple[PerceptionT, ...] = attrib(converter=tuple)


@attrs(frozen=True)
class BagOfFeaturesPerceptualRepresentationFrame(PerceptualRepresentationFrame):
    r"""
    Represents a learner's perception of a `Situation` as an unstructured set of features.

    For testing purposes only.
    """
    features: ImmutableSet[str] = attrib(converter=immutableset)


@attrs(frozen=True)
class DummyVisualPerception:
    """
    A visual representation with a location and a tag, no structure or properties.

    It simply says for e.g. a truck "this looks like a truck." and here is its (point) location.

    This is only for testing purposes.
    """

    tag: str = attrib(validator=instance_of(str))
    location: Point = attrib(validator=instance_of(Point))


class PerceptualRepresentationGenerator(Generic[SituationT, PerceptionT], ABC):
    r"""
    A way of generating `PerceptualRepresentation`\ s of `Situation` s.
    """
    @abstractmethod
    def generate_perception(self, situation: SituationT, chooser: SequenceChooser =
    fixed_random_factory()) -> PerceptualRepresentation[PerceptionT]:
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

