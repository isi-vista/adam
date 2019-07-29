r"""
Ways to produce human language descriptions of `Situation`\ s by rule.
"""
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from attr import attrs, attrib
from attr.validators import instance_of
from immutablecollections import ImmutableSet, immutableset
from vistautils.iter_utils import only

from adam.language import LinguisticDescriptionT, TokenSequenceLinguisticDescription
from adam.language.ontology_dictionary import OntologyLexicon
from adam.random_utils import SequenceChooser, fixed_random_factory
from adam.situation import Situation, LocatedObjectSituation

SituationT = TypeVar("SituationT", bound=Situation)


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
class SingleObjectLanguageGenerator(
    LanguageGenerator[LocatedObjectSituation, TokenSequenceLinguisticDescription]
):
    r"""
    `LanguageGenerator` for describing a single object.

    Describes a `Situation` containing a `SituationObject` with a single word: its name
    according to some `OntologyDictionary`.

    This language generator will throw a `ValueError` if it receives any situation which
    contains either multiple `SituationObject`\ s.
    """
    _ontology_lexicon: OntologyLexicon = attrib(validator=instance_of(OntologyLexicon))

    def generate_language(
        self,
        situation: LocatedObjectSituation,
        chooser: SequenceChooser = fixed_random_factory(),  # pylint:disable=unused-argument
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
