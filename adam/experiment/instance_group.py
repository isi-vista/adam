from abc import ABC, abstractmethod
from typing import Generic, Iterable, Tuple, Optional

from attr import attrs, attrib
from attr.validators import instance_of
from immutablecollections.converter_utils import _to_tuple

from adam.language import LinguisticDescriptionT
from adam.language.language_generator import SituationT, LanguageGenerator
from adam.perception import (
    PerceptionT,
    PerceptualRepresentation,
    PerceptualRepresentationGenerator,
)
from adam.random_utils import SequenceChooser


class InstanceGroup(ABC, Generic[SituationT, LinguisticDescriptionT, PerceptionT]):
    r"""
    An `InstanceGroup` can provide triples of (optional)
    `Situation`\ s, `LinguisticDescription`\ s, and `PerceptualRepresentation`\ s
    for use in training or testing `LanguageLearner`\ s
    with the `Experiment` class.
    """

    @abstractmethod
    def name(self) -> str:
        """
        A human-readable name for this instance group.
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
        The instances in the order they should be shown to the `LanguageLearner`.
        """


@attrs(frozen=True, slots=True)
class ExplicitWithoutSituationInstanceGroup(
    InstanceGroup[None, LinguisticDescriptionT, PerceptionT]
):
    r"""
    A collection of instances where the user explicitly specifies
    the `LinguisticDescription`\ s and `PerceptualRepresentation`\ s
    but not the `Situation`\ s.
    """
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
    r"""
    A collection of instances where the user explicitly specifies
    the the `Situation`\ s, `LinguisticDescription`\ s, and `PerceptualRepresentation`\ s.
    """
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
    r"""
    Creates a collection of instances
    by taking an explicitly provided sequence of `Situation`\ s
    and deriving the `LinguisticDescription`\ s and `PerceptualRepresentation`\ s
    by applying the *language_generator* and *perception_generator*, respectively.
    """
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
