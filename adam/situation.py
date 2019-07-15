"""
Structures for describing situations in the world at an abstacted, human-friendly level.
"""
from abc import ABC


class Situation(ABC):
    """
    A situation is a high-level representation of a configuration of objects, possibly including
    changes in the states of objects across time.

    A Curriculum is a sequence of situations.

    Situations are a high-level description intended to make it easy for human beings to specify
    curricula.  Situations will be transformed into pairs of `PerceptualRepresentation`\ s and
    `LinguisticDescription`\ s for input to a `LanguageLearner`\ .
    """
