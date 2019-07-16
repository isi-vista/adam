r"""
Classes to represent a strategy for presenting `LearningExample`\ s to a `LanguageLearner`.
"""
from abc import ABC, abstractmethod
from random import Random
from typing import Sequence, Tuple

from attr import attrs, attrib

from adam.learner import LearningExample


class CurriculumGenerator(ABC):
    @abstractmethod
    def generate_curriculum(self, rng: Random) -> Sequence[LearningExample]:
        r"""
        Produce a sequence of `LearningExample`s for a `Learner`\ .

        Args:
            rng: random number generator to be used for random decisions (if any) made during the
            curriculum generation process.

        Returns:
            A sequence of `LearningExample`\ s to be presented to a `LanguageLearner`.
        """

    @staticmethod
    def create_always_generating(curriculum: Sequence[LearningExample]) -> \
            'CurriculumGenerator':
        r"""
        Get a `CurriculumGenerator` which always generates the specific curriculum.

        Args:
            curriculum: The sequence of `LearningExample`\ s to always generate.

        Returns:
            A `CurriculumGenerator` which always generates the specific curriculum.
        """


@attrs(frozen=True)
class _ExplicitCurriculumGenerator(CurriculumGenerator):
    """
    A curriculum generator which always returns the exact list of `LearningExample`\ s
    provided at its construction.

    This is useful for testing.
    """
    _learning_examples: Tuple[LearningExample, ...] = attrib(converter=tuple)

    def generate_curriculum(self, rng: Random) -> Sequence[LearningExample]:
        return self._learning_examples
