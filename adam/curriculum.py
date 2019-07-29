r"""
Classes to represent a strategy for presenting `LearningExample`\ s to a `LanguageLearner`.
"""
from abc import ABC, abstractmethod
from random import Random
from typing import Generic, Sequence

from adam.learner import LearningExample
from adam.language import LinguisticDescriptionT
from adam.perception import _PerceptionT


class CurriculumGenerator(ABC, Generic[_PerceptionT, LinguisticDescriptionT]):
    @abstractmethod
    def generate_curriculum(
        self, rng: Random
    ) -> Sequence[LearningExample[_PerceptionT, LinguisticDescriptionT]]:
        r"""
        Produce a sequence of `LearningExample`s for a `Learner`\ .

        Args:
            rng: random number generator to be used for random decisions (if any) made during the
            curriculum generation process.

        Returns:
            A sequence of `LearningExample`\ s to be presented to a `LanguageLearner`.
        """

    @staticmethod
    def create_always_generating(
        curriculum: Sequence[LearningExample[_PerceptionT, LinguisticDescriptionT]]
    ) -> "CurriculumGenerator[_PerceptionT, LinguisticDescriptionT]":
        r"""
        Get a `CurriculumGenerator` which always generates the specific curriculum.

        Args:
            curriculum: The sequence of `LearningExample`\ s to always generate.

        Returns:
            A `CurriculumGenerator` which always generates the specific curriculum.
        """
        return _ExplicitCurriculumGenerator(curriculum)


# for some reason attrs and mypy don't play well here, so we do this the old-fashioned way
class _ExplicitCurriculumGenerator(
    CurriculumGenerator[_PerceptionT, LinguisticDescriptionT]
):
    r"""
    A curriculum generator which always returns the exact list of `LearningExample`\ s
    provided at its construction.

    This is useful for testing.
    """

    def __init__(
        self,
        learning_examples: Sequence[
            LearningExample[_PerceptionT, LinguisticDescriptionT]
        ],
    ) -> None:
        self._learning_examples = tuple(learning_examples)

    def generate_curriculum(
        self, rng: Random  # pylint:disable=unused-argument
    ) -> Sequence[LearningExample[_PerceptionT, LinguisticDescriptionT]]:
        return self._learning_examples
