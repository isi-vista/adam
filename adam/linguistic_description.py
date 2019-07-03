"""
Representations of the linguistic input and outputs of a Learner.
"""
from abc import ABC


class LinguisticDescription(ABC):
    """
    A linguistic description of a Situation.

    This, together with a PerceptualRepresentation, forms an observation that a Learner learns from.

    A trained Learner can then generate new LinguisticDescriptions given only a
    PerceptualRepresentation of a Situation.
    """
