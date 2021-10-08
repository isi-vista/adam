from itertools import chain

import pytest
from more_itertools import flatten

from adam.curriculum.curriculum_utils import (
    standard_object,
    phase1_instances,
    PHASE1_CHOOSER_FACTORY,
    PHASE1_TEST_CHOOSER_FACTORY,
)
from adam.curriculum.phase1_curriculum import (
    _object_with_color_template,
    _x_has_y_template,
)
from adam.language.language_utils import phase1_language_generator
from adam.language_specific.english.english_language_generator import IGNORE_HAS_AS_VERB
from adam.learner import LearningExample
from adam.learner.attributes import SubsetAttributeLearner
from adam.learner.integrated_learner import IntegratedTemplateLearner
from adam.learner.language_mode import LanguageMode
from adam.ontology import IS_SPEAKER, IS_ADDRESSEE
from adam.ontology.phase1_ontology import (
    RED,
    WHITE,
    BLACK,
    GREEN,
    BLUE,
    BALL,
    BOOK,
    GAILA_PHASE_1_ONTOLOGY,
    CAR,
    PERSON,
    INANIMATE_OBJECT,
    PERSON_CAN_HAVE,
    MOM,
    DAD,
    BABY,
)
from adam.situation.templates.phase1_templates import property_variable, sampled
from tests.learner import (
    LANGUAGE_MODE_TO_TEMPLATE_LEARNER_OBJECT_RECOGNIZER,
)


def integrated_learner_factory(language_mode: LanguageMode):
    return IntegratedTemplateLearner(
        object_learner=LANGUAGE_MODE_TO_TEMPLATE_LEARNER_OBJECT_RECOGNIZER[language_mode],
        attribute_learner=SubsetAttributeLearner(
            ontology=GAILA_PHASE_1_ONTOLOGY, beam_size=5, language_mode=language_mode
        ),
    )


# TODO: fix https://github.com/isi-vista/adam/issues/917 which causes us to have to specify that we don't wish to include ME_HACK and YOU_HACK in our curriculum design


@pytest.mark.parametrize(
    "color_node,object_0_node,object_1_node",
    [
        (RED, BALL, BOOK),
        (BLUE, BALL, BOOK),
        (GREEN, BALL, BOOK),
        (BLACK, BALL, CAR),
        (WHITE, BALL, CAR),
    ],
)
@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize("learner", [integrated_learner_factory])
def test_subset_color_attribute(
    color_node, object_0_node, object_1_node, language_mode, learner
):
    color = property_variable(f"{color_node.handle}", color_node)
    object_0 = standard_object(
        f"{object_0_node.handle}", object_0_node, added_properties=[color]
    )
    object_1 = standard_object(
        f"{object_1_node.handle}", object_1_node, added_properties=[color]
    )

    color_object_template = _object_with_color_template(object_0, None)

    templates = [color_object_template, _object_with_color_template(object_1, None)]

    language_generator = phase1_language_generator(language_mode)

    color_train_curriculum = phase1_instances(
        f"{color.handle} Color Train",
        language_generator=language_generator,
        situations=chain(
            *[
                flatten(
                    [
                        sampled(
                            template,
                            chooser=PHASE1_CHOOSER_FACTORY(),
                            ontology=GAILA_PHASE_1_ONTOLOGY,
                            max_to_sample=2,
                            block_multiple_of_the_same_type=True,
                        )
                        for template in templates
                    ]
                )
            ]
        ),
    )

    color_test_curriculum = phase1_instances(
        f"{color.handle} Color Test",
        situations=sampled(
            color_object_template,
            chooser=PHASE1_TEST_CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            max_to_sample=1,
            block_multiple_of_the_same_type=True,
        ),
        language_generator=language_generator,
    )

    processing_learner = learner(language_mode)

    for (
        _,
        linguistic_description,
        perceptual_representation,
    ) in color_train_curriculum.instances():
        processing_learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )

    for (
        _,
        test_lingustics_description,
        test_perceptual_representation,
    ) in color_test_curriculum.instances():
        descriptions_from_learner = processing_learner.describe(
            test_perceptual_representation
        )
        gold = test_lingustics_description.as_token_sequence()
        assert descriptions_from_learner
        assert gold in [desc.as_token_sequence() for desc in descriptions_from_learner]


# hack: wo de and ni de are currently considered to be one word. This won't work for third person possession
# TODO: Fix this learning test. See: https://github.com/isi-vista/adam/issues/861
@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize("learner", [integrated_learner_factory])
def test_subset_my_attribute_learner_integrated(language_mode, learner):
    inanimate_object = standard_object(
        "object", INANIMATE_OBJECT, required_properties=[PERSON_CAN_HAVE]
    )

    language_generator = phase1_language_generator(language_mode)

    my_train_curriculum = phase1_instances(
        "my-train",
        situations=flatten(
            sampled(
                _x_has_y_template(
                    standard_object("speaker", person, added_properties=[IS_SPEAKER]),
                    inanimate_object,
                    syntax_hints=[IGNORE_HAS_AS_VERB],
                ),
                ontology=GAILA_PHASE_1_ONTOLOGY,
                chooser=PHASE1_CHOOSER_FACTORY(),
                max_to_sample=5,
                block_multiple_of_the_same_type=True,
            )
            for person in [MOM, DAD, BABY]
        ),
        language_generator=language_generator,
    )

    my_test_curriculum = phase1_instances(
        "my-test",
        situations=sampled(
            _x_has_y_template(
                standard_object(
                    "speaker",
                    PERSON,
                    banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
                    added_properties=[IS_SPEAKER],
                ),
                inanimate_object,
                syntax_hints=[IGNORE_HAS_AS_VERB],
            ),
            block_multiple_of_the_same_type=True,
            ontology=GAILA_PHASE_1_ONTOLOGY,
            chooser=PHASE1_TEST_CHOOSER_FACTORY(),
            max_to_sample=1,
        ),
        language_generator=language_generator,
    )

    processing_learner = learner(language_mode)

    for (
        _,
        linguistic_description,
        perceptual_representation,
    ) in my_train_curriculum.instances():
        processing_learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )

    for (
        _,
        test_lingustics_description,
        test_perceptual_representation,
    ) in my_test_curriculum.instances():
        descriptions_from_learner = processing_learner.describe(
            test_perceptual_representation
        )
        gold = test_lingustics_description.as_token_sequence()
        assert descriptions_from_learner
        assert gold in [desc.as_token_sequence() for desc in descriptions_from_learner]


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize("learner", [integrated_learner_factory])
def test_your_attribute_learner(language_mode, learner):
    person_0 = standard_object(
        "speaker",
        PERSON,
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
        added_properties=[IS_SPEAKER],
    )
    person_1 = standard_object(
        "addressee",
        PERSON,
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
        added_properties=[IS_ADDRESSEE],
    )
    inanimate_object = standard_object(
        "object", INANIMATE_OBJECT, required_properties=[PERSON_CAN_HAVE]
    )

    language_generator = phase1_language_generator(language_mode)

    your_train_curriculum = phase1_instances(
        "your-train",
        situations=sampled(
            _x_has_y_template(
                person_1,
                inanimate_object,
                background=[person_0],
                syntax_hints=[IGNORE_HAS_AS_VERB],
            ),
            block_multiple_of_the_same_type=True,
            ontology=GAILA_PHASE_1_ONTOLOGY,
            chooser=PHASE1_CHOOSER_FACTORY(),
            max_to_sample=5,
        ),
        language_generator=language_generator,
    )

    your_test_curriculum = phase1_instances(
        "your-test",
        situations=sampled(
            _x_has_y_template(
                person_1,
                inanimate_object,
                background=[person_0],
                syntax_hints=[IGNORE_HAS_AS_VERB],
            ),
            block_multiple_of_the_same_type=True,
            ontology=GAILA_PHASE_1_ONTOLOGY,
            chooser=PHASE1_TEST_CHOOSER_FACTORY(),
            max_to_sample=1,
        ),
        language_generator=language_generator,
    )

    process_learner = learner(language_mode)

    for (
        _,
        linguistic_description,
        perceptual_representation,
    ) in your_train_curriculum.instances():
        process_learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )

    for (
        _,
        test_lingustics_description,
        test_perceptual_representation,
    ) in your_test_curriculum.instances():
        descriptions_from_learner = process_learner.describe(
            test_perceptual_representation
        )
        gold = test_lingustics_description.as_token_sequence()
        assert descriptions_from_learner
        assert gold in [desc.as_token_sequence() for desc in descriptions_from_learner]
