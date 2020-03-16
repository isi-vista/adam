from itertools import chain

from more_itertools import flatten

from adam.curriculum.curriculum_utils import (
    standard_object,
    phase1_instances,
    PHASE1_CHOOSER,
)
from adam.curriculum.phase1_curriculum import _object_with_color_template
from adam.learner import LearningExample
from adam.learner.attributes import SubsetAttributeLearner
from adam.ontology.phase1_ontology import (
    RED,
    LIGHT_BROWN,
    DARK_BROWN,
    WHITE,
    BLACK,
    GREEN,
    BLUE,
    BALL,
    BOOK,
    GAILA_PHASE_1_ONTOLOGY,
)
from adam.situation.templates.phase1_templates import property_variable, sampled
from tests.learner import TEST_OBJECT_RECOGNIZER

SUBSET_ATTRIBUTE_LEARNER_FACTORY = lambda: SubsetAttributeLearner(
    object_recognizer=TEST_OBJECT_RECOGNIZER, ontology=GAILA_PHASE_1_ONTOLOGY
)


def test_subset_red_color_learner():
    red = property_variable("red", RED)
    ball = standard_object("ball", BALL, added_properties=[red])
    book = standard_object("book", BOOK, added_properties=[red])

    color_ball_template = _object_with_color_template(ball)

    templates = [color_ball_template, _object_with_color_template(book)]

    red_train_curriculum = phase1_instances(
        "Red Color Train",
        situations=chain(
            *[
                flatten(
                    [
                        sampled(
                            template,
                            chooser=PHASE1_CHOOSER,
                            ontology=GAILA_PHASE_1_ONTOLOGY,
                            max_to_sample=2,
                        )
                        for template in templates
                    ]
                )
            ]
        ),
    )

    red_test_curriculum = phase1_instances(
        "Red Color Test",
        situations=sampled(
            color_ball_template,
            chooser=PHASE1_CHOOSER,
            ontology=GAILA_PHASE_1_ONTOLOGY,
            max_to_sample=1,
        ),
    )

    learner = SUBSET_ATTRIBUTE_LEARNER_FACTORY()

    for (
        _,
        linguistic_description,
        perceptual_representation,
    ) in red_train_curriculum.instances():
        learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )

    for (
        _,
        test_lingustics_description,
        test_perceptual_representation,
    ) in red_test_curriculum.instances():
        descriptions_from_learner = learner.describe(test_perceptual_representation)
        gold = test_lingustics_description.as_token_sequence()
        assert descriptions_from_learner
        assert [desc.as_token_sequence() for desc in descriptions_from_learner][0] == gold


def test_subset_blue_color_learner():
    blue = property_variable("blue", BLUE)
    ball = standard_object("ball", BALL, added_properties=[blue])
    book = standard_object("book", BOOK, added_properties=[blue])

    color_ball_template = _object_with_color_template(ball)

    templates = [color_ball_template, _object_with_color_template(book)]

    blue_train_curriculum = phase1_instances(
        "Blue Color Train",
        situations=chain(
            *[
                flatten(
                    [
                        sampled(
                            template,
                            chooser=PHASE1_CHOOSER,
                            ontology=GAILA_PHASE_1_ONTOLOGY,
                            max_to_sample=2,
                        )
                        for template in templates
                    ]
                )
            ]
        ),
    )

    blue_test_curriculum = phase1_instances(
        "Blue Color Test",
        situations=sampled(
            color_ball_template,
            chooser=PHASE1_CHOOSER,
            ontology=GAILA_PHASE_1_ONTOLOGY,
            max_to_sample=1,
        ),
    )

    learner = SUBSET_ATTRIBUTE_LEARNER_FACTORY()

    for (
        _,
        linguistic_description,
        perceptual_representation,
    ) in blue_train_curriculum.instances():
        learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )

    for (
        _,
        test_lingustics_description,
        test_perceptual_representation,
    ) in blue_test_curriculum.instances():
        descriptions_from_learner = learner.describe(test_perceptual_representation)
        gold = test_lingustics_description.as_token_sequence()
        assert descriptions_from_learner
        assert [desc.as_token_sequence() for desc in descriptions_from_learner][0] == gold


def test_subset_green_color_learner():
    green = property_variable("green", GREEN)
    ball = standard_object("ball", BALL, added_properties=[green])
    book = standard_object("book", BOOK, added_properties=[green])

    color_ball_template = _object_with_color_template(ball)

    templates = [color_ball_template, _object_with_color_template(book)]

    green_train_curriculum = phase1_instances(
        "Green Color Train",
        situations=chain(
            *[
                flatten(
                    [
                        sampled(
                            template,
                            chooser=PHASE1_CHOOSER,
                            ontology=GAILA_PHASE_1_ONTOLOGY,
                            max_to_sample=2,
                        )
                        for template in templates
                    ]
                )
            ]
        ),
    )

    green_test_curriculum = phase1_instances(
        "Green Color Test",
        situations=sampled(
            color_ball_template,
            chooser=PHASE1_CHOOSER,
            ontology=GAILA_PHASE_1_ONTOLOGY,
            max_to_sample=1,
        ),
    )

    learner = SUBSET_ATTRIBUTE_LEARNER_FACTORY()

    for (
        _,
        linguistic_description,
        perceptual_representation,
    ) in green_train_curriculum.instances():
        learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )

    for (
        _,
        test_lingustics_description,
        test_perceptual_representation,
    ) in green_test_curriculum.instances():
        descriptions_from_learner = learner.describe(test_perceptual_representation)
        gold = test_lingustics_description.as_token_sequence()
        assert descriptions_from_learner
        assert [desc.as_token_sequence() for desc in descriptions_from_learner][0] == gold


def test_subset_black_color_learner():
    black = property_variable("black", BLACK)
    pass


def test_subset_white_color_learner():
    white = property_variable("white", WHITE)
    pass


def test_subset_light_brown_color_learner():
    light_brown = property_variable("light-brown", LIGHT_BROWN)
    pass


def test_subset_dark_brown_color_learner():
    dark_brown = property_variable("dark-brown", DARK_BROWN)
    pass
