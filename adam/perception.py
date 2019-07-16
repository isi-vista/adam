r"""
This module provides classes related to the perceptual primitive representation used to describe
`Situation`\ s from the point-of-view of `LanguageLearner`\ s.
"""
from abc import ABC
from typing import TypeVar

from attr import attrs, attrib
from immutablecollections import ImmutableSet, immutableset


class PerceptualRepresentation(ABC):
    r"""
    Represents a `LanguageLearner`\ 's perception of some `Situation`\ .

    This, paired with a `LinguisticDescription`\ , forms an observation that a `LanguageLearner`\
    learns from.
    """


_PerceptionT = TypeVar("_PerceptionT", bound=PerceptualRepresentation)


@attrs(frozen=True)
class BagOfFeaturesPerceptualRepresentation(PerceptualRepresentation):
    r"""
    Represents a learner's perception of a `Situation` as an unstructured set of features.

    For testing purposes only.
    """
    features: ImmutableSet[str] = attrib(converter=immutableset)
