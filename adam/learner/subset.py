from typing import Dict, Generic, Mapping, Set, Tuple

from attr import Factory, attrib, attrs
from immutablecollections import immutabledict

from adam.language import LinguisticDescriptionT, TokenSequenceLinguisticDescription
from adam.learner import LanguageLearner, LearningExample
from adam.perception import PerceptionT, PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    PropertyPerception,
    DevelopmentalPrimitivePerceptionFrame,
)


@attrs
class SubsetLanguageLearner(
    Generic[PerceptionT, LinguisticDescriptionT],
    LanguageLearner[PerceptionT, TokenSequenceLinguisticDescription],
):
    """
    An implementation of `LanguageLearner` for subset learning based approach for single object detection.
    """

    _descriptions_to_properties: Dict[Tuple[str, ...], Set[PropertyPerception]] = attrib(
        init=False, default=Factory(dict)
    )

    def observe(
        self,
        learning_example: LearningExample[
            PerceptionT, TokenSequenceLinguisticDescription
        ],
    ) -> None:
        perception = learning_example.perception
        perception_frames = perception.frames
        if len(perception_frames) != 1:
            raise RuntimeError("Subset learner can only handle single frames for now")

        observed_linguistic_description = (
            learning_example.linguistic_description.as_token_sequence()
        )
        if isinstance(perception_frames[0], DevelopmentalPrimitivePerceptionFrame):
            observed_property_assertions = perception_frames[0].property_assertions
        else:
            raise RuntimeError(f"Cannot process perception of type {type(perception)}")

        if observed_linguistic_description in self._descriptions_to_properties:
            # If already observed, reduce the properties for that description to the common subset of
            # the new observation and the previous observations
            # TODO: We should relax this requirement for learning: issue #361
            self._descriptions_to_properties[
                observed_linguistic_description
            ] = self._descriptions_to_properties[
                observed_linguistic_description
            ].intersection(
                observed_property_assertions
            )
        else:
            # If it's a new description, learn that as a new observation
            self._descriptions_to_properties[observed_linguistic_description] = set(
                observed_property_assertions
            )

    def describe(
        self, perception: PerceptualRepresentation[PerceptionT]
    ) -> Mapping[TokenSequenceLinguisticDescription, float]:
        if len(perception.frames) != 1:
            raise RuntimeError("Subset learner can only handle single frames for now")
        if isinstance(perception.frames[0], DevelopmentalPrimitivePerceptionFrame):
            observed_property_assertions = perception.frames[0].property_assertions
        else:
            raise RuntimeError("Cannot process perception type.")

        # get the learned description for which there are the maximum number of matching properties (i.e. most specific)
        max_matching_properties = 0
        learned_description = None
        for description, properties in self._descriptions_to_properties.items():
            if all(prop in observed_property_assertions for prop in properties) and (
                len(properties) > max_matching_properties
            ):
                learned_description = description
                max_matching_properties = len(properties)
        if learned_description:
            return immutabledict(
                ((TokenSequenceLinguisticDescription(learned_description), 1.0),)
            )
        else:
            return immutabledict()
