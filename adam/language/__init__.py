r"""
Representations of the linguistic input and outputs of a `LanguageLearner`\ .
"""
from abc import ABC, abstractmethod
from typing import Tuple, TypeVar, Sequence, Sized

from attr import attrib, attrs
from attr.validators import instance_of, deep_iterable

from immutablecollections.converter_utils import _to_tuple
from vistautils.span import Span


class LinguisticDescription(ABC, Sequence[str]):
    r"""
    A linguistic description of a `Situation`\ .

    This, together with a `PerceptualRepresentation`\ , forms an observation that a
    `LanguageLearner` learns from.

    A trained `LanguageLearner` can then generate new `LinguisticDescription`\ s given only a
    `PerceptualRepresentation` of a `Situation`.

    Any `LinguisticDescription` must minimally be able to be treated as a sequence of token strings.
    """

    def span(self, start_index: int, *, end_index_exclusive: int) -> Span:
        return Span(start_index, end_index_exclusive)

    def as_token_sequence(self) -> Tuple[str, ...]:
        """
        Get this description as a tuple of token strings.

        Returns:
            A tuple of token strings describing this `LinguisticDescription`
        """
        return tuple(self)

    def as_token_string(self) -> str:
        return " ".join(self.as_token_sequence())


LinguisticDescriptionT = TypeVar("LinguisticDescriptionT", bound=LinguisticDescription)


@attrs(frozen=True)
class TokenSequenceLinguisticDescription(LinguisticDescription):
    """
    A `LinguisticDescription` which consists of a sequence of tokens.
    """

    tokens: Tuple[str, ...] = attrib(
        converter=_to_tuple, validator=deep_iterable(instance_of(str))
    )

    def as_token_sequence(self) -> Tuple[str, ...]:
        return self.tokens

    def __getitem__(self, item) -> str:
        return self.tokens[item]

    def __len__(self) -> int:
        return len(self.tokens)
