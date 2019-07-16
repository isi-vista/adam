r"""
Representations of the linguistic input and outputs of a `LanguageLearner`\ .
"""
from abc import ABC
from typing import Tuple

from attr import attrs, attrib
from attr.validators import instance_of


class LinguisticDescription(ABC):
    r"""
    A linguistic description of a `Situation`\ .

    This, together with a `PerceptualRepresentation`\ , forms an observation that a
    `LanguageLearner` learns from.

    A trained `LanguageLearner` can then generate new `LinguisticDescription`\ s given only a
    `PerceptualRepresentation` of a `Situation`.
    """


@attrs(frozen=True)
class TokenSequenceLinguisticDescription(LinguisticDescription):
    """
    A `LinguisticDescription` which consists of a sequence of tokens.
    """

    tokens: Tuple[str, ...] = attrib(validator=instance_of(tuple))
