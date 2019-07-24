r"""
This module provides classes related to the perceptual primitive representation used to describe
`Situation`\ s from the point-of-view of `LanguageLearner`\ s.
"""
from abc import ABC
from typing import TypeVar, Generic, Tuple

from attr import attrs, attrib
from attr.validators import instance_of
from immutablecollections import ImmutableSet, immutableset

from adam.math_3d import Point


class PerceptualRepresentationFrame(ABC):
    r"""
    Represents a `LanguageLearner`\ 's perception of some `Situation`\ at a single moment.

    One or more of these, paired with a `LinguisticDescription`\ , forms an observation that a
    `LanguageLearner` learns from.
    """


_PerceptionT = TypeVar("_PerceptionT", bound=PerceptualRepresentationFrame)


@attrs(frozen=True)
class PerceptualRepresentation(Generic[_PerceptionT]):
    """
    A learner's perception of a situation as a sequence of perceptual representations of
    individual moments.

    Usually for a static situation, this will be a single frame, but it could be two or
    three for complex actions.
    """

    frames: Tuple[_PerceptionT, ...] = attrib(converter=tuple)


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
