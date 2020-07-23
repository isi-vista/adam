r"""
Ways to produce human language descriptions of `Situation`\ s by rule.
"""
from abc import ABC, abstractmethod
from typing import Generic

from adam.language import LinguisticDescriptionT, TokenSequenceLinguisticDescription
from adam.language.ontology_dictionary import OntologyLexicon
from adam.random_utils import SequenceChooser
from adam.situation import LocatedObjectSituation, SituationT
from attr import attrib, attrs
from attr.validators import instance_of
from immutablecollections import ImmutableSet, immutableset
from vistautils.iter_utils import only


class LanguageGenerator(Generic[SituationT, LinguisticDescriptionT], ABC):
    r"""
    A way of describing `Situation`\ s using human `LinguisticDescription`\ s.
    """

    @abstractmethod
    def generate_language(
        self, situation: SituationT, chooser: SequenceChooser
    ) -> ImmutableSet[LinguisticDescriptionT]:
        r"""
        Generate a collection of human language descriptions of the given `Situation`.
        Args:
            situation: the `Situation` to describe.
            chooser: the `SequenceChooser` to use if any random decisions are required.

        Returns:
            A `LinguisticDescription` of that situation.
        """


@attrs(frozen=True, slots=True)
class ChooseFirstLanguageGenerator(LanguageGenerator[SituationT, LinguisticDescriptionT]):
    """
    A `LanguageGenerator` used to wrap another `LanguageGenerator` and discard all but its first
    generated option.
    """

    _wrapped_generator: LanguageGenerator[SituationT, LinguisticDescriptionT] = attrib(
        validator=instance_of(LanguageGenerator)
    )

    def generate_language(
        self, situation: SituationT, chooser: SequenceChooser
    ) -> ImmutableSet[LinguisticDescriptionT]:
        wrapped_result = self._wrapped_generator.generate_language(situation, chooser)
        if wrapped_result:
            return immutableset([wrapped_result[0]])
        else:
            return immutableset()


@attrs(frozen=True, slots=True)
class ChooseRandomLanguageGenerator(
    LanguageGenerator[SituationT, LinguisticDescriptionT]
):
    """
    A `LanguageGenerator` used to wrap another `LanguageGenerator` and discard all but one of its
    descriptions, selected at random using the provided `SequenceChooser` .
    """

    _wrapped_generator: LanguageGenerator[SituationT, LinguisticDescriptionT] = attrib(
        validator=instance_of(LanguageGenerator)
    )
    _sequence_chooser: SequenceChooser = attrib(validator=instance_of(SequenceChooser))

    def generate_language(
        self, situation: SituationT, chooser: SequenceChooser
    ) -> ImmutableSet[LinguisticDescriptionT]:
        wrapped_result = self._wrapped_generator.generate_language(situation, chooser)
        if wrapped_result:
            # noinspection PyTypeChecker
            return immutableset([self._sequence_chooser.choice(wrapped_result)])
        else:
            return immutableset()


@attrs(frozen=True, slots=True)
class SingleObjectLanguageGenerator(
    LanguageGenerator[LocatedObjectSituation, TokenSequenceLinguisticDescription]
):
    r"""
    `LanguageGenerator` for describing a single object.

    Describes a `Situation` containing a `SituationObject` with a single word: its name
    according to some `OntologyLexicon`.

    This language generator will throw a `ValueError` if it receives any situation which
    contains either multiple `SituationObject`\ s.
    """
    _ontology_lexicon: OntologyLexicon = attrib(validator=instance_of(OntologyLexicon))

    def generate_language(
        self,
        situation: LocatedObjectSituation,
        chooser: SequenceChooser,  # pylint:disable=unused-argument
    ) -> ImmutableSet[TokenSequenceLinguisticDescription]:
        if len(situation.objects_to_locations) != 1:
            raise ValueError(
                r"A situation must contain exactly one object to be described by "
                r"SingleObjectLanguageGenerator, but got {situation}"
            )

        lone_object = only(situation.objects_to_locations)

        if not lone_object.ontology_node:
            raise ValueError(
                f"Object {lone_object} lacks an ontology node in situation "
                f"{situation}."
            )

        # disabled warning due to PyCharm bug
        # noinspection PyTypeChecker
        return immutableset(
            TokenSequenceLinguisticDescription((word.base_form,))
            for word in self._ontology_lexicon.words_for_node(lone_object.ontology_node)
        )
