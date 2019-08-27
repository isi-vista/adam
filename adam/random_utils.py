"""
Utilities for working with random numbers.

This currently contains only an abstraction over `random.choice` which makes it easier to test
things which make random choices.
"""
import random
from abc import ABC
from random import Random
from typing import TypeVar, Sequence

from attr import attrs, attrib
from attr.validators import instance_of

T = TypeVar("T")  # pylint:disable=invalid-name


class SequenceChooser(ABC):
    """
    Abstraction over a strategy for selecting items from a sequence.
    """

    def choice(self, elements: Sequence[T]) -> T:
        """
        Choose one element from *elements* using some undefined policy.

        Args:
            elements: The sequence of elements to choose from.  If this sequence is empty, an
            `IndexError` should be raised.

        Returns:
            One of the elements of *elements*; no further requirement is defined.
        """


@attrs(frozen=True, slots=True)
class FixedIndexChooser(SequenceChooser):
    """
    A `SequenceChooser` which always chooses the first element.
    """

    _index_to_choose: int = attrib(validator=instance_of(int))

    # noinspection PyMethodMayBeStatic
    def choice(self, elements: Sequence[T]) -> T:
        return elements[self._index_to_choose]


@attrs(frozen=True, slots=True)
class RandomChooser(SequenceChooser):
    """
    A `SequenceChooser` which delegates the choice to a contained standard library random number
     generator.
    """

    _random: Random = attrib(validator=instance_of(Random))

    def choice(self, elements: Sequence[T]) -> T:
        return self._random.choice(elements)

    @staticmethod
    def for_seed(seed: int = 0) -> "RandomChooser":
        """
        Get a `RandomChooser` from a random number generator initialized with the specified seed.
        """
        ret = random.Random()
        ret.seed(seed)
        return RandomChooser(ret)
