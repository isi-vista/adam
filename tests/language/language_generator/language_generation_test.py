from immutablecollections import ImmutableSet, immutableset

from adam.language import TokenSequenceLinguisticDescription
from adam.language.language_generator import (
    ChooseFirstLanguageGenerator,
    ChooseRandomLanguageGenerator,
    LanguageGenerator,
    SingleObjectLanguageGenerator,
)
from adam.language_specific.english.english_phase_1_lexicon import (
    GAILA_PHASE_1_ENGLISH_LEXICON,
)
from adam.math_3d import Point
from adam.ontology.phase1_ontology import BALL
from adam.random_utils import FixedIndexChooser, RandomChooser, SequenceChooser
from adam.situation import LocatedObjectSituation, Situation
from tests.adam_test_utils import situation_object


def test_single_object_generator():
    # warning due to PyCharm bug
    # noinspection PyTypeChecker
    situation = LocatedObjectSituation(
        objects_to_locations=((situation_object(BALL), Point(0, 0, 0)),)
    )

    single_obj_generator = SingleObjectLanguageGenerator(GAILA_PHASE_1_ENGLISH_LEXICON)

    languages_for_situation = single_obj_generator.generate_language(
        situation, RandomChooser.for_seed(0)
    )
    assert len(languages_for_situation) == 1
    assert languages_for_situation[0].as_token_sequence() == ("ball",)


class DummyLanguageGenerator(
    LanguageGenerator[Situation, TokenSequenceLinguisticDescription]
):
    def generate_language(  # pylint:disable=unused-argument
        self, situation: Situation, chooser: SequenceChooser
    ) -> ImmutableSet[TokenSequenceLinguisticDescription]:
        return immutableset(
            (
                TokenSequenceLinguisticDescription(("hello", "world")),
                TokenSequenceLinguisticDescription(("hello", "fred")),
            )
        )


def test_choose_first():
    generator = ChooseFirstLanguageGenerator(DummyLanguageGenerator())
    # pycharm fails to recognize converter
    # noinspection PyTypeChecker
    situation = LocatedObjectSituation([(situation_object(BALL), Point(0, 0, 0))])

    generated_descriptions = generator.generate_language(
        situation, RandomChooser.for_seed(0)
    )
    assert len(generated_descriptions) == 1
    assert generated_descriptions[0].as_token_sequence() == ("hello", "world")


def test_choose_random():
    generator = ChooseRandomLanguageGenerator(
        DummyLanguageGenerator(), FixedIndexChooser(1)
    )
    # pycharm fails to recognize converter
    # noinspection PyTypeChecker
    situation = LocatedObjectSituation([(situation_object(BALL), Point(0, 0, 0))])

    generated_descriptions = generator.generate_language(
        situation, RandomChooser.for_seed(0)
    )
    assert len(generated_descriptions) == 1
    assert generated_descriptions[0].as_token_sequence() == ("hello", "fred")
