"""
This module provides classes related to the perceptual primitive representation used to describe
`Situation`\ s from the point-of-view of `LanguageLearner`\ s.
"""
from abc import ABC


class PerceptualRepresentation(ABC):
    """
    Represents a `LanguageLearner`\ 's perception of some `Situation`\ .

    This, paired with a `LinguisticDescription`\ , forms an observation that a `LanguageLearner`\  learns
    from.
    """
