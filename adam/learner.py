"""
Interfaces for language learning code.
"""

from abc import ABC, abstractmethod
from typing import Dict, Generic, Mapping, Set

from attr import Factory, attrib, attrs
from attr.validators import instance_of
from immutablecollections import immutabledict

from adam.language import LinguisticDescription, LinguisticDescriptionT
from adam.perception import PerceptionT, PerceptualRepresentation
from adam.perception.developmental_primitive_perception import PropertyPerception


@attrs(frozen=True)
class LearningExample(Generic[PerceptionT, LinguisticDescriptionT]):
    """
    A `PerceptualRepresentation` of a situation and its `LinguisticDescription`
    that a `LanguageLearner` can learn from.
    """

    # attrs can't check the generic types, so we just check the super-types
    perception: PerceptualRepresentation[PerceptionT] = attrib(  # type:ignore
        validator=instance_of(PerceptualRepresentation)
    )
    """
    The `LanguageLearner`'s perception of the `Situation`
    """
    linguistic_description: LinguisticDescriptionT = attrib(  # type:ignore
        validator=instance_of(LinguisticDescription)
    )
    """
    A human-language description of the `Situation`
    """


class LanguageLearner(ABC, Generic[PerceptionT, LinguisticDescriptionT]):
    r"""
    Models an infant learning language.

    A `LanguageLearner` learns language by observing a sequence of `LearningExample`\ s.

    A `LanguageLearner` can describe new situations given a `PerceptualRepresentation`\ .
    """

    @abstractmethod
    def observe(
            self, learning_example: LearningExample[PerceptionT, LinguisticDescriptionT]
    ) -> None:
        """
        Observe a `LearningExample`, possibly updating internal state.
        """

    @abstractmethod
    def describe(
            self, perception: PerceptualRepresentation[PerceptionT]
    ) -> Mapping[LinguisticDescriptionT, float]:
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
    Generic[PerceptionT, LinguisticDescriptionT],
    LanguageLearner[PerceptionT, LinguisticDescriptionT],
):
    """
    A trivial implementation of `LanguageLearner` which just memorizes situations it has seen before
    and cannot produce descriptions of any other situations.

    If this learner observes the same `PerceptualRepresentation` multiple times, only the final
    description is memorized.

    This implementation is only useful for testing.
    """

    _memorized_situations: Dict[
        PerceptualRepresentation[PerceptionT], LinguisticDescriptionT
    ] = attrib(init=False, default=Factory(dict))

    def observe(
            self, learning_example: LearningExample[PerceptionT, LinguisticDescriptionT]
    ) -> None:
        self._memorized_situations[
            learning_example.perception
        ] = learning_example.linguistic_description

    def describe(
            self, perception: PerceptualRepresentation[PerceptionT]
    ) -> Mapping[LinguisticDescriptionT, float]:
        memorized_description = self._memorized_situations.get(perception)
        if memorized_description:
            return immutabledict(((memorized_description, 1.0),))
        else:
            return immutabledict()


@attrs
class SubsetLanguageLearner(
    Generic[PerceptionT, LinguisticDescriptionT],
    LanguageLearner[PerceptionT, LinguisticDescriptionT],
):
    """
    An implementation of `LanguageLearner` for subset learning based approach for single object detection.
    """

    _descriptions_to_properties: Dict[
        LinguisticDescriptionT, Set[PropertyPerception]
    ] = attrib(init=False, default=Factory(dict))

    def observe(
            self, learning_example: LearningExample[PerceptionT, LinguisticDescriptionT]
    ) -> None:
        frames = learning_example.perception.frames
        description = learning_example.linguistic_description
        if len(frames) != 1:
            raise RuntimeError('Subset learner can only handle single frames for now')

        property_assertions = frames[0].property_assertions
        # If already observed, reduce the properties for that description to the common subset of
        # the new observation and the previous observations
        if description in self._descriptions_to_properties:
            self._descriptions_to_properties[description] = \
                self._descriptions_to_properties[description].intersection(property_assertions)
        else:
            self._descriptions_to_properties[learning_example.linguistic_description] = property_assertions

    def describe(
            self, perception: PerceptualRepresentation[PerceptionT]
    ) -> Mapping[LinguisticDescriptionT, float]:
        if len(perception.frames) != 1:
            raise RuntimeError('Subset learner can only handle single frames for now')

        perception_properties = perception.frames[0].property_assertions
        # get the learned description for which there are the maximum number of matching properties (i.e. most specific)
        max_matching_properties = 0
        learned_description = None
        for description, properties in self._descriptions_to_properties.items():
            if all(prop in perception.frames[0].property_assertions for prop in properties) \
                    and (len(properties) > max_matching_properties):
                learned_description = description
        if learned_description and max_matching_properties > 0:
            return immutabledict(((learned_description, 1.0),))
        else:
            return immutabledict()
