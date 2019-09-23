"""
Utilities for working with random numbers.

This currently contains only an abstraction over `random.choice` which makes it easier to test
things which make random choices.
"""
import random
from abc import ABC
from random import Random
from typing import Sequence, TypeVar

from attr import attrib, attrs
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
    A `SequenceChooser` which always chooses the element at the given index.

    If the fixed index exceeds the length of the supplied (non-empty) sequence,
    then the element at the fixed index modulo the sequence length is returned.
    """

    _index_to_choose: int = attrib(validator=instance_of(int))

    # noinspection PyMethodMayBeStatic
    def choice(self, elements: Sequence[T]) -> T:
        return elements[self._index_to_choose % len(elements)]


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


@attrs(slots=True)
class RotatingIndexChooser(SequenceChooser):
    """
    A `SequenceChooser` which increments the index it chooses after each choice.

    If the current index exceeds the length of the supplied (non-empty) sequence,
    then the element at the current index modulo the sequence length is returned.
    """

    _cur_index: int = attrib(default=0, init=False)

    def choice(self, elements: Sequence[T]) -> T:
        ret = elements[self._cur_index % len(elements)]
        self._cur_index += 1
        return ret
