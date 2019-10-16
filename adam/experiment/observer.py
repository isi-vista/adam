import logging
from abc import ABC, abstractmethod
from typing import Generic, Mapping, Optional, Tuple

from attr import attrib, attrs
from attr.validators import instance_of

from adam.language import (
    LinguisticDescription,
    LinguisticDescriptionT,
    TokenSequenceLinguisticDescription,
)
from adam.language.language_generator import SituationT
from adam.perception import PerceptionT, PerceptualRepresentation


class DescriptionObserver(Generic[SituationT, LinguisticDescriptionT, PerceptionT], ABC):
    r"""
    Something which can observe the descriptions produced by `LanguageLearner`\ s.

    Typically a `DescriptionObserver` will provide some sort of summary of its observations
    when its *report* method is called.
    """

    @abstractmethod
    def observe(
        self,
        situation: Optional[SituationT],
        true_description: TokenSequenceLinguisticDescription,
        perceptual_representation: PerceptualRepresentation[PerceptionT],
        predicted_descriptions: Mapping[TokenSequenceLinguisticDescription, float],
    ) -> None:
        r"""
        Observe a description provided by a `LanguageLearner`.

        Args:
            situation: The `Situation` being described. This is optional.
            true_description: The "gold-standard" description of the situation.
            perceptual_representation: The `PerceptualRepresentation` of the situation received by
                                       the `LanguageLearner`.
            predicted_descriptions:  The scored `LinguisticDescription`\ s produced by
                                     the `LanguageLearner`.
        """

    @abstractmethod
    def report(self) -> None:
        """
        Take some action based on the observations.

        Typically, this will be to write a report either to the console, to a file, or both.
        """


@attrs(slots=True)
class TopChoiceExactMatchObserver(
    DescriptionObserver[SituationT, LinguisticDescriptionT, PerceptionT]
):
    r"""
    Log how often the top-scoring predicted `LinguisticDescription` for a `Situation` exactly
    matches the expected `LinguisticDescription`.

    If there are multiple predicted `LinguisticDescription`\ s with the same score, which one is
    compared to determine matching is undefined.
    """
    name: str = attrib(validator=instance_of(str))
    _num_observations: int = attrib(init=False, default=0)
    _num_top_choice_matches: int = attrib(init=False, default=0)

    def observe(  # pylint:disable=unused-argument
        self,
        situation: Optional[SituationT],
        true_description: TokenSequenceLinguisticDescription,
        perceptual_representation: PerceptualRepresentation[PerceptionT],
        predicted_descriptions: Mapping[TokenSequenceLinguisticDescription, float],
    ) -> None:
        self._num_observations += 1

        if predicted_descriptions:
            top_choice = max(predicted_descriptions.items(), key=_by_score)

            if top_choice == true_description:
                self._num_top_choice_matches += 1

    def report(self) -> None:
        logging.info(
            f"{self.name}: top prediction matched expected prediction "
            f"{self._num_observations} / {self._num_top_choice_matches} times ("
            f"{100.0*self._num_top_choice_matches/self._num_observations:2.3f} %)"
        )


# used by TopChoiceExactMatchObserver
def _by_score(scored_description: Tuple[LinguisticDescription, float]) -> float:
    return scored_description[1]
