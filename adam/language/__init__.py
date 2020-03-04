r"""
Representations of the linguistic input and outputs of a `LanguageLearner`\ .
"""
from abc import ABC, abstractmethod
from typing import Tuple, TypeVar

from attr import attrib, attrs
from attr.validators import instance_of

from vistautils.preconditions import check_arg


class LinguisticDescription(ABC):
    r"""
    A linguistic description of a `Situation`\ .

    This, together with a `PerceptualRepresentation`\ , forms an observation that a
    `LanguageLearner` learns from.

    A trained `LanguageLearner` can then generate new `LinguisticDescription`\ s given only a
    `PerceptualRepresentation` of a `Situation`.
    """

    @abstractmethod
    def as_token_sequence(self) -> Tuple[str, ...]:
        """
        Get this description as a tuple of token strings.

        Returns:
            A tuple of token strings describing this `LinguisticDescription`
        """

    def as_token_string(self) -> str:
        return " ".join(self.as_token_sequence())


LinguisticDescriptionT = TypeVar("LinguisticDescriptionT", bound=LinguisticDescription)


@attrs(frozen=True)
class TokenSequenceLinguisticDescription(LinguisticDescription):
    """
    A `LinguisticDescription` which consists of a sequence of tokens.
    """

    tokens: Tuple[str, ...] = attrib(validator=instance_of(tuple))

    def as_token_sequence(self) -> Tuple[str, ...]:
        return self.tokens

    @attrs(frozen=True, slots=True)
    class Span:
        _start_index: int = attrib(validator=instance_of(int))
        _end_index: int = attrib(validator=instance_of(int))

        def __attrs_post_init__(self) -> None:
            check_arg(self._start_index <= self._end_index)
