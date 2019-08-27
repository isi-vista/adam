"""
Allows managing experimental configurations in code.
"""
import logging
from itertools import chain

# for some reason, pylint doesn't recognize the types used in quoted type annotations
from typing import Any, Callable, Generic, Sequence, Tuple  # pylint:disable=unused-import

from attr import attrib, attrs
from attr.validators import instance_of

# noinspection PyProtectedMember
from immutablecollections.converter_utils import _to_tuple
from vistautils.preconditions import check_arg

from adam.language import LinguisticDescriptionT
from adam.language.language_generator import SituationT
from adam.learner import LanguageLearner, LearningExample
from adam.perception import PerceptionT
from adam.random_utils import SequenceChooser
from adam.experiment.instance_group import InstanceGroup
from adam.experiment.observer import DescriptionObserver


@attrs(frozen=True)
class Experiment(Generic[SituationT, LinguisticDescriptionT, PerceptionT]):
    r"""
    A particular experimental configuration.

    An experiment specifies what data is fed to the `LanguageLearner`, what `LanguageLearner` is
    used and how it is configured, how the trained `LanguageLearner` is tested, and what
    analysis is done on the results.

    Every experiment has one or more *training_stages*;
    simple experiments will have only one,
    while those which reflect a curriculum
    which increases in complexity over time may have several.
    Each of these is an `InstanceGroup`
    providing triples of a `Situation`
    with the corresponding `LinguisticDescription` and `PerceptualRepresentation`.
    There are many ways an `InstanceGroup` could be specified,
    ranging from simply a collection of these triples (e.g. `ExplicitWithSituationInstanceGroup`)
    to a complex rule-governed process (e.g. `GeneratedFromExplicitSituationsInstanceGroup`).

    *learner_factory* specifies a no-argument function which will return the `LanguageLearner`
    which should be trained.

    *test_instance_groups* are the situations and perceptual representations
    the trained `LanguageLearner` will be asked to describe.
    These are specified by `InstanceGroup`\ s,
    just like the training data.

    *warmup_instance_groups* may optionally be provided
    if at test time the `LanguageLearner` needs to be shown some observations
    before evaluation
    (for example, to introduce some new objects).

    At various stages in the experiment,
    `DescriptionObserver`\ s will be able to examine
    the descriptions of situations produced by the `LanguageLearner`.
    Based on their observations, they can provide reports at the end of the experiment.

    *pre_example_training_observers* are provided
    with the description a `LanguageLearner` would give to a situation during training
    before it is shown the correct description.
    *post_example_training_observers* are the same
    except they receive the learner's description
    after it has updated itself by seeing the correct description.
    *test_observers* observe the descriptions given by the learner on the test instances.
    These are what should be used for computing evaluation metrics.

    Every experiment also has a *name* which is a human-readable description.
    A *sequence_chooser* must also be provided to ensure reproducibility
    of any random decisions made by components of the experiment.

    Note that `Experiment` objects should not be reused because components
    (especially the observers) may maintain state.
    """

    name: str = attrib(validator=instance_of(str))
    training_stages: "Sequence[InstanceGroup[SituationT, LinguisticDescriptionT, PerceptionT]]" = attrib(
        converter=_to_tuple, kw_only=True
    )
    learner_factory: Callable[
        [], LanguageLearner[PerceptionT, LinguisticDescriptionT]
    ] = attrib(kw_only=True)
    sequence_chooser: SequenceChooser = attrib(
        validator=instance_of(SequenceChooser), kw_only=True
    )
    pre_example_training_observers: "Tuple[DescriptionObserver[SituationT, LinguisticDescriptionT, PerceptionT]]" = attrib(  # type: ignore
        converter=_to_tuple, default=tuple(), kw_only=True
    )
    post_example_training_observers: "Tuple[DescriptionObserver[SituationT, LinguisticDescriptionT, PerceptionT]]" = attrib(  # type: ignore
        converter=_to_tuple, default=tuple(), kw_only=True
    )
    warm_up_test_instance_groups: "Sequence[InstanceGroup[Any, LinguisticDescriptionT, PerceptionT]]" = attrib(
        converter=_to_tuple, default=tuple(), kw_only=True
    )
    test_instance_groups: "Sequence[InstanceGroup[Any, LinguisticDescriptionT, PerceptionT]]" = attrib(
        converter=_to_tuple, default=tuple(), kw_only=True
    )
    test_observers: "Sequence[DescriptionObserver[SituationT, LinguisticDescriptionT, PerceptionT]]" = attrib(
        converter=_to_tuple, default=tuple(), kw_only=True
    )

    def __attrs_post_init__(self) -> None:
        check_arg(callable(self.learner_factory), "Learner factory must be callable")


def execute_experiment(
    experiment: Experiment[SituationT, LinguisticDescriptionT, PerceptionT]
) -> None:
    """
    Runs an `Experiment`.
    """
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
                    raise ValueError(
                        "Obsserved training instances cannot lack a situation"
                    )

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
