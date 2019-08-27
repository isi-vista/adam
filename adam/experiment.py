"""
Allows managing experimental configurations in code.
"""
import logging
from abc import ABC, abstractmethod
from itertools import chain
from typing import Any, Callable, Generic, Iterable, Mapping, Optional, Sequence, Tuple

from attr import attrib, attrs
from attr.validators import instance_of
from immutablecollections.converter_utils import _to_tuple
from vistautils.preconditions import check_arg

from adam.language import LinguisticDescription, LinguisticDescriptionT
from adam.language.language_generator import LanguageGenerator, SituationT
from adam.learner import LanguageLearner, LearningExample
from adam.perception import (
    PerceptionT,
    PerceptualRepresentation,
    PerceptualRepresentationGenerator,
)
from adam.random_utils import SequenceChooser


class InstanceGroup(ABC, Generic[SituationT, LinguisticDescriptionT, PerceptionT]):
    @abstractmethod
    def name(self) -> str:
        """
        TODO: fill me in
        Returns:

        """

    @abstractmethod
    def instances(
        self
    ) -> Iterable[
        Tuple[
            Optional[SituationT],
            LinguisticDescriptionT,
            PerceptualRepresentation[PerceptionT],
        ]
    ]:
        """
        TODO: fill me in
        Returns:

        """


@attrs(frozen=True, slots=True)
class ExplicitWithoutSituationInstanceGroup(
    InstanceGroup[None, LinguisticDescriptionT, PerceptionT]
):
    _name: str = attrib(validator=instance_of(str))
    # https://github.com/python-attrs/attrs/issues/519
    _instances: Tuple[  # type: ignore
        Tuple[LinguisticDescriptionT, PerceptualRepresentation[PerceptionT]]
    ] = attrib(converter=_to_tuple)

    def name(self) -> str:
        return self._name

    def instances(
        self
    ) -> Iterable[
        Tuple[None, LinguisticDescriptionT, PerceptualRepresentation[PerceptionT]]
    ]:
        for (linguistic_description, perception) in self._instances:
            yield (None, linguistic_description, perception)


@attrs(frozen=True, slots=True)
class ExplicitWithSituationInstanceGroup(
    InstanceGroup[SituationT, LinguisticDescriptionT, PerceptionT]
):
    _name: str = attrib(validator=instance_of(str))
    # https://github.com/python-attrs/attrs/issues/519
    _instances: Tuple[  # type: ignore
        Tuple[SituationT, LinguisticDescriptionT, PerceptualRepresentation[PerceptionT]]
    ] = attrib(converter=_to_tuple)

    def name(self) -> str:
        return self._name

    def instances(
        self
    ) -> Iterable[
        Tuple[
            Optional[SituationT],
            LinguisticDescriptionT,
            PerceptualRepresentation[PerceptionT],
        ]
    ]:
        return self._instances


@attrs(frozen=True, slots=True)
class GeneratedFromExplicitSituationsInstanceGroup(
    InstanceGroup[SituationT, LinguisticDescriptionT, PerceptionT]
):
    _name: str = attrib(validator=instance_of(str))
    _situations: Iterable[SituationT] = attrib(validator=instance_of(Iterable))
    _language_generator: LanguageGenerator[SituationT, LinguisticDescriptionT] = attrib(
        validator=instance_of(LanguageGenerator)
    )
    _perception_generator: PerceptualRepresentationGenerator[
        SituationT, PerceptionT
    ] = attrib(validator=instance_of(PerceptualRepresentationGenerator))
    _chooser: SequenceChooser = attrib(validator=instance_of(SequenceChooser))

    def name(self) -> str:
        return self._name

    def instances(
        self
    ) -> Iterable[
        Tuple[
            Optional[SituationT],
            LinguisticDescriptionT,
            PerceptualRepresentation[PerceptionT],
        ]
    ]:
        for situation in self._situations:
            # suppress PyCharm type inference bug
            # noinspection PyTypeChecker
            for linguistic_description in self._language_generator.generate_language(
                situation, self._chooser
            ):
                yield (
                    situation,
                    linguistic_description,
                    self._perception_generator.generate_perception(
                        situation, self._chooser
                    ),
                )


class DescriptionObserver(Generic[SituationT, LinguisticDescriptionT, PerceptionT], ABC):
    @abstractmethod
    def observe(
        self,
        situation: Optional[SituationT],
        true_description: LinguisticDescriptionT,
        perceptual_representation: PerceptualRepresentation[PerceptionT],
        predicted_descriptions: Mapping[LinguisticDescriptionT, float],
    ) -> None:
        """
        FILL ME IN

        Args:
            situation:
            true_description:
            perceptual_representation:
            predicted_descriptions:

        Returns:
        """

    @abstractmethod
    def report(self) -> None:
        """
        FILL ME IN

        Returns:

        """


# used by TopChoiceExactMatchObserver
def _by_score(scored_description: Tuple[LinguisticDescription, float]) -> float:
    return scored_description[1]


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

    def observe(
        self,
        situation: Optional[SituationT],
        true_description: LinguisticDescriptionT,
        perceptual_representation: PerceptualRepresentation[PerceptionT],
        predicted_descriptions: Mapping[LinguisticDescriptionT, float],
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


@attrs(frozen=True)
class Experiment(Generic[SituationT, LinguisticDescriptionT, PerceptionT]):
    name: str = attrib(validator=instance_of(str))
    training_stages: Sequence[
        InstanceGroup[SituationT, LinguisticDescriptionT, PerceptionT]
    ] = attrib(converter=_to_tuple, kw_only=True)
    learner_factory: Callable[
        [], LanguageLearner[PerceptionT, LinguisticDescriptionT]
    ] = attrib(kw_only=True)
    sequence_chooser: SequenceChooser = attrib(
        validator=instance_of(SequenceChooser), kw_only=True
    )
    pre_example_training_observers: Tuple[  # type: ignore
        DescriptionObserver[SituationT, LinguisticDescriptionT, PerceptionT]
    ] = attrib(converter=_to_tuple, default=tuple(), kw_only=True)
    post_example_training_observers: Tuple[  # type: ignore
        DescriptionObserver[SituationT, LinguisticDescriptionT, PerceptionT]
    ] = attrib(converter=_to_tuple, default=tuple(), kw_only=True)
    warm_up_test_instance_groups: Sequence[
        InstanceGroup[Any, LinguisticDescriptionT, PerceptionT]
    ] = attrib(converter=_to_tuple, default=tuple(), kw_only=True)
    test_instance_groups: Sequence[
        InstanceGroup[Any, LinguisticDescriptionT, PerceptionT]
    ] = attrib(converter=_to_tuple, default=tuple(), kw_only=True)
    test_observers: Sequence[  # type: ignore
        DescriptionObserver[SituationT, LinguisticDescriptionT, PerceptionT]
    ] = attrib(converter=_to_tuple, default=tuple(), kw_only=True)

    def __attrs_post_init__(self) -> None:
        check_arg(callable(self.learner_factory), "Learner factory must be callable")


def execute_experiment(
    experiment: Experiment[SituationT, LinguisticDescriptionT, PerceptionT]
) -> None:
    logging.info("Beginning experiment %s", experiment.name)

    learner = experiment.learner_factory()
    logging.info("Instantiated learner %s", learner)

    for training_stage in experiment.training_stages:
        logging.info("Beginning training stage %s", training_stage.name)
        for (
            situation,
            linguistic_description,
            perceptual_representation,
        ) in training_stage.instances():
            if experiment.pre_example_training_observers:
                learner_descriptions_before_seeing_example = learner.describe(
                    perceptual_representation
                )
                if situation:
                    for pre_example_observer in experiment.pre_example_training_observers:
                        pre_example_observer.observe(
                            situation,
                            linguistic_description,
                            perceptual_representation,
                            learner_descriptions_before_seeing_example,
                        )
                else:
                    raise ValueError("Training instances cannot lack a situation")
            learner.observe(
                LearningExample(perceptual_representation, linguistic_description)
            )

            if experiment.post_example_training_observers:
                learner_descriptions_after_seeing_example = learner.describe(
                    perceptual_representation
                )
                for post_example_observer in experiment.post_example_training_observers:
                    post_example_observer.observe(
                        situation,
                        linguistic_description,
                        perceptual_representation,
                        learner_descriptions_after_seeing_example,
                    )
    logging.info("Training complete")

    for training_observer in chain(
        experiment.pre_example_training_observers,
        experiment.post_example_training_observers,
    ):
        training_observer.report()

    logging.info("Warming up for tests")
    for warm_up_instance_group in experiment.warm_up_test_instance_groups:
        for (
            situation,
            warm_up_test_instance_language,
            warm_up_test_instance_perception,
        ) in warm_up_instance_group.instances():
            learner.observe(
                LearningExample(
                    warm_up_test_instance_perception, warm_up_test_instance_language
                )
            )

    logging.info("Performing tests")
    for test_instance_group in experiment.test_instance_groups:
        for (
            situation,
            test_instance_language,
            test_instance_perception,
        ) in test_instance_group.instances():
            descriptions_from_learner = learner.describe(test_instance_perception)
            for test_observer in experiment.test_observers:
                test_observer.observe(
                    situation,
                    test_instance_language,
                    test_instance_perception,
                    descriptions_from_learner,
                )

    for test_observer in experiment.test_observers:
        test_observer.report()

    logging.info("Experiment %s complete", experiment.name)
