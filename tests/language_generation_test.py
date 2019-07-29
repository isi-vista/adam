from adam.language.language_generator import SingleObjectLanguageGenerator
from adam.math_3d import Point
from adam.situation import LocatedObjectSituation, SituationObject
from .testing_lexicon import ENGLISH_TESTING_LEXICON
from .testing_ontology import TRUCK, INANIMATE


def test_single_object_generator():
    # warning due to PyCharm bug
    # noinspection PyTypeChecker
    situation = LocatedObjectSituation(
        objects_to_locations=((SituationObject(TRUCK, [INANIMATE]), Point(0, 0, 0)),)
    )

    single_obj_generator = SingleObjectLanguageGenerator(ENGLISH_TESTING_LEXICON)

    languages_for_situation = single_obj_generator.generate_language(situation)
    assert len(languages_for_situation) == 1
    assert languages_for_situation[0].as_token_sequence() == ("truck",)
