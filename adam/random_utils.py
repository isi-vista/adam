"""
Utilities for working with random numbers.

This currently contains only an abstraction over `random.choice` which makes it easier to test
things which make random choices.
"""
from abc import ABC
from random import Random
from typing import TypeVar, Sequence

from attr import attrs, attrib
from attr.validators import instance_of

T = TypeVar("T")  # pylint:disable=invalid-name


class SequenceChooser(ABC):
    def choice(self, elements: Sequence[T]) -> T:
        """
        Choose one element from *elements* using some undefined.

        Args:
            elements: The sequence of elements to choose from.  If this sequence is empty, an
            `IndexError` should be raised.

        Returns:
            One of the elements of *elements*; no further requirement is defined.
        """


@attrs(frozen=True, slots=True)
class AlwaysChooseTheFirst(SequenceChooser):
    """
    A `SequenceChooser` which always chooses the first element.
    """

    # noinspection PyMethodMayBeStatic
    def choice(self, elements: Sequence[T]) -> T:
        return elements[0]


@attrs(frozen=True, slots=True)
class RandomChooser(SequenceChooser):
    """
    A `SequenceChooser` which delegates the choice to a contained standard library random number
     generator.
    """

    _random: Random = attrib(validator=instance_of(Random))

    def choice(self, elements: Sequence[T]) -> T:
        return self._random.choice(elements)
