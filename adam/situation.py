"""
Structures for describing situations in the world at an abstacted, human-friendly level.
"""
from abc import ABC

from attr import attrs, attrib
from immutablecollections import immutableset, ImmutableSet


class Situation(ABC):
    r"""
    A situation is a high-level representation of a configuration of objects, possibly including
    changes in the states of objects across time.

    A Curriculum is a sequence of situations.

    Situations are a high-level description intended to make it easy for human beings to specify
    curricula.  Situations will be transformed into pairs of `PerceptualRepresentation`\ s and
    `LinguisticDescription`\ s for input to a `LanguageLearner`.
    """


@attrs(frozen=True)
class BagOfFeaturesSituationRepresentation(Situation):
    r"""
    Represents a `Situation` as an unstructured set of features.

    For testing purposes only.
    """
    features: ImmutableSet[str] = attrib(converter=immutableset)
