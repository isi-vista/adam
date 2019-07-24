"""
Interfaces for language learning code.
"""

from abc import ABC, abstractmethod
from typing import Generic, Mapping, Dict

from attr import Factory, attrib, attrs
from attr.validators import instance_of

from immutablecollections import immutabledict

from adam.linguistic_description import LinguisticDescription, _LinguisticDescriptionT
from adam.perception import _PerceptionT, PerceptualRepresentation


@attrs(frozen=True)
class LearningExample(Generic[_PerceptionT, _LinguisticDescriptionT]):
    """
    A `PerceptualRepresentation` of a situation and its `LinguisticDescription`
    that a `LanguageLearner` can learn from.
    """

    # attrs can't check the generic types, so we just check the super-types
    perception: PerceptualRepresentation[_PerceptionT] = attrib(  # type:ignore
        validator=instance_of(PerceptualRepresentation)
    )
    linguistic_description: _LinguisticDescriptionT = attrib(  # type:ignore
        validator=instance_of(LinguisticDescription)
    )


class LanguageLearner(Generic[_PerceptionT, _LinguisticDescriptionT], ABC):
    r"""
    Models an infant learning language.

    A `LanguageLearner` learns language by observing a sequence of `LearningExample`\ s.

    A `LanguageLearner` can describe new situations given a `PerceptualRepresentation`\ .
    """

    @abstractmethod
    def observe(
        self, learning_example: LearningExample[_PerceptionT, _LinguisticDescriptionT]
    ) -> None:
        """
        Observe a `LearningExample`, possibly updating internal state.
        """

    @abstractmethod
    def describe(
        self, perception: PerceptualRepresentation[_PerceptionT]
    ) -> Mapping[_LinguisticDescriptionT, float]:
        r"""
        Given a `PerceptualRepresentation` of a situation, produce one or more
        `LinguisticDescription`\ s of it.

        The descriptions are returned as a mapping from linguistic descriptions to their scores.
        The scores are not defined other than "higher is better."

        It is possible that the learner cannot produce a description, in which case an empty
        mapping is returned.
        """


@attrs
class MemorizingLanguageLearner(
    Generic[_PerceptionT, _LinguisticDescriptionT],
    LanguageLearner[_PerceptionT, _LinguisticDescriptionT],
):
    """
    A trivial implementation of `LanguageLearner` which just memorizes situations it has seen before
    and cannot produce descriptions of any other situations.

    If this learner observes the same `PerceptualRepresentation` multiple times, only the final
    description is memorized.

    This implementation is only useful for testing.
    """

    _memorized_situations: Dict[
        PerceptualRepresentation[_PerceptionT], _LinguisticDescriptionT
    ] = attrib(init=False, default=Factory(dict))

    def observe(
        self, learning_example: LearningExample[_PerceptionT, _LinguisticDescriptionT]
    ) -> None:
        self._memorized_situations[
            learning_example.perception
        ] = learning_example.linguistic_description

    def describe(
        self, perception: PerceptualRepresentation[_PerceptionT]
    ) -> Mapping[_LinguisticDescriptionT, float]:
        memorized_description = self._memorized_situations.get(perception)
        if memorized_description:
            return immutabledict(((memorized_description, 1.0),))
        else:
            return immutabledict()
