"""
This module provides classes related to the perceptual primitive representation used to describe
Situations from the point-of-view of Learners.
"""
from abc import ABC


class PerceptualRepresentation(ABC):
    """
    Represents a Learner's perception of some Situation.

    This, paired with a LinguisticDescription, forms an observation that a Learner learns from.
    """
