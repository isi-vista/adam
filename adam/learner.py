from abc import ABC, abstractmethod
from typing import Generic, Mapping, TypeVar

from attr import Factory, attrib, attrs
from attr.validators import instance_of

from immutablecollections import immutabledict

from adam.linguistic_description import LinguisticDescription
from adam.perception import PerceptualRepresentation

_PerceptionT = TypeVar("_PerceptionT", bound=PerceptualRepresentation)
_LinguisticDescriptionT = TypeVar("_LinguisticDescriptionT", bound=LinguisticDescription)


@attrs(frozen=True)
class LearningExample(Generic[_PerceptionT, _LinguisticDescriptionT]):
    perception: _PerceptionT = attrib(validator=instance_of(PerceptualRepresentation))
    linguistic_description: _LinguisticDescriptionT = attrib(
        validator=instance_of(LinguisticDescription)
    )


class LanguageLearner(Generic[_PerceptionT, _LinguisticDescriptionT], ABC):
    """
    Models an infant learning language.

    A Learner learns language by observing a sequence of LearningExamples.
    A Learner can describe new situations giben a PerceptualRepresentation.
    """

    @abstractmethod
    def observe(
        self, learning_example: LearningExample[_PerceptionT, _LinguisticDescriptionT]
    ) -> None:
        """
        Observe a learning example, possibly updating internal state.
        """

    @abstractmethod
    def describe(
        self, perception: _PerceptionT
    ) -> Mapping[_LinguisticDescriptionT, float]:
        """
        Given a perception of a situation, produce one or more linguistic descriptions of it.

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
    A trivial implementation of LanguageLearner which just memorizes situations it has seen before
    and cannot produce descriptions of any other situations.

    If this learner observes the same perceptual representation multiple times, only the final
    description is memorized.

    This implementation is only useful for testing.
    """

    _memorized_situations: Mapping[_PerceptionT, _LinguisticDescriptionT] = attrib(
        init=False, default=Factory(dict)
    )

    def observe(
        self, learning_example: LearningExample[_PerceptionT, _LinguisticDescriptionT]
    ) -> None:
        self._memorized_situations[
            learning_example.perception
        ] = learning_example.linguistic_description

    def describe(
        self, perception: _PerceptionT
    ) -> Mapping[_LinguisticDescriptionT, float]:
        memorized_description = self._memorized_situations.get(perception)
        if memorized_description:
            return immutabledict(((memorized_description, 1.0),))
        else:
            return immutabledict()
