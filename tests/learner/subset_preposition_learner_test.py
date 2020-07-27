import pytest

from adam.curriculum.curriculum_utils import (
    PHASE1_CHOOSER_FACTORY,
    PHASE1_TEST_CHOOSER_FACTORY,
    phase1_instances,
    standard_object,
)
from adam.curriculum.phase1_curriculum import _x_has_y_template
from adam.curriculum.preposition_curriculum import (
    _behind_template,
    _beside_template,
    _in_front_template,
    _in_template,
    _on_template,
    _over_template,
    _under_template,
)
from adam.language.language_utils import phase1_language_generator
from adam.learner import LearningExample
from adam.learner.integrated_learner import IntegratedTemplateLearner
from adam.learner.language_mode import LanguageMode
from adam.learner.prepositions import SubsetPrepositionLearner
from adam.learner.quantifers import QuantifierTemplateLearner
from adam.learner.relations import SubsetRelationLearnerNew
from adam.ontology.phase1_ontology import (
    BALL,
    BOOK,
    CUP,
    GAILA_PHASE_1_ONTOLOGY,
    MOM,
    PERSON,
    TABLE,
    WATER,
)
from adam.situation.templates.phase1_templates import object_variable, sampled
from immutablecollections import immutableset
from tests.learner import (
    LANGUAGE_MODE_TO_OBJECT_RECOGNIZER,
    LANGUAGE_MODE_TO_TEMPLATE_LEARNER_OBJECT_RECOGNIZER,
)


def subset_relation_language_factory(
    language_mode: LanguageMode
) -> SubsetPrepositionLearner:
    return SubsetPrepositionLearner(
        object_recognizer=LANGUAGE_MODE_TO_OBJECT_RECOGNIZER[language_mode],
        ontology=GAILA_PHASE_1_ONTOLOGY,
        language_mode=language_mode,
    )


def integrated_learner_factory(language_mode: LanguageMode):
    return IntegratedTemplateLearner(
        object_learner=LANGUAGE_MODE_TO_TEMPLATE_LEARNER_OBJECT_RECOGNIZER[language_mode],
        relation_learner=SubsetRelationLearnerNew(
            ontology=GAILA_PHASE_1_ONTOLOGY, beam_size=5, language_mode=language_mode
        ),
        number_learner=QuantifierTemplateLearner.pretrained_for_language_mode(
            language_mode
        ),
        language_mode=language_mode,
    )


def run_preposition_test(learner, situation_template, language_generator):
    train_curriculum = phase1_instances(
        "Preposition Unit Train",
        situations=sampled(
            situation_template,
            chooser=PHASE1_CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            max_to_sample=2,
        ),
        language_generator=language_generator,
    )
    test_curriculum = phase1_instances(
        "Preposition Unit Test",
        situations=sampled(
            situation_template,
            chooser=PHASE1_TEST_CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            max_to_sample=1,
        ),
        language_generator=language_generator,
    )

    for (
        _,
        linguistic_description,
        perceptual_representation,
    ) in train_curriculum.instances():
        learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )

    for (
        _,
        test_lingustics_description,
        test_perceptual_representation,
    ) in test_curriculum.instances():
        descriptions_from_learner = learner.describe(test_perceptual_representation)
        gold = test_lingustics_description.as_token_sequence()
        assert descriptions_from_learner
        assert gold in [desc.as_token_sequence() for desc in descriptions_from_learner]


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize(
    "learner",
    [pytest.mark.skip(subset_relation_language_factory), integrated_learner_factory],
)
def test_subset_preposition_on(language_mode, learner):
    ball = standard_object("ball", BALL)
    table = standard_object("table", TABLE)

    run_preposition_test(
        learner(language_mode),
        _on_template(ball, table, immutableset(), is_training=True),
        language_generator=phase1_language_generator(language_mode),
    )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize(
    "learner",
    [pytest.mark.skip(subset_relation_language_factory), integrated_learner_factory],
)
def test_subset_preposition_beside(language_mode, learner):
    ball = standard_object("ball", BALL)
    table = standard_object("table", TABLE)

    run_preposition_test(
        learner(language_mode),
        _beside_template(ball, table, immutableset(), is_training=True, is_right=True),
        language_generator=phase1_language_generator(language_mode),
    )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize(
    "learner",
    [pytest.mark.skip(subset_relation_language_factory), integrated_learner_factory],
)
def test_subset_preposition_under(language_mode, learner):
    ball = standard_object("ball", BALL)
    table = standard_object("table", TABLE)

    run_preposition_test(
        learner(language_mode),
        _under_template(ball, table, immutableset(), is_training=True, is_distal=True),
        language_generator=phase1_language_generator(language_mode),
    )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize(
    "learner",
    [pytest.mark.skip(subset_relation_language_factory), integrated_learner_factory],
)
def test_subset_preposition_over(language_mode, learner):
    ball = standard_object("ball", BALL)
    table = standard_object("table", TABLE)

    run_preposition_test(
        learner(language_mode),
        _over_template(ball, table, immutableset(), is_training=True, is_distal=True),
        language_generator=phase1_language_generator(language_mode),
    )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize(
    "learner",
    [pytest.mark.skip(subset_relation_language_factory), integrated_learner_factory],
)
def test_subset_preposition_in(language_mode, learner):
    water = object_variable("water", WATER)
    cup = standard_object("cup", CUP)

    run_preposition_test(
        learner(language_mode),
        _in_template(water, cup, immutableset(), is_training=True),
        language_generator=phase1_language_generator(language_mode),
    )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize(
    "learner",
    [pytest.mark.skip(subset_relation_language_factory), integrated_learner_factory],
)
def test_subset_preposition_behind(language_mode, learner):
    ball = standard_object("ball", BALL)
    table = standard_object("table", TABLE)

    run_preposition_test(
        learner(language_mode),
        _behind_template(
            ball,
            table,
            immutableset(),
            is_training=True,
            is_near=True,
            speaker_root_node=MOM,
        ),
        language_generator=phase1_language_generator(language_mode),
    )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize(
    "learner",
    [pytest.mark.skip(subset_relation_language_factory), integrated_learner_factory],
)
def test_subset_preposition_in_front(language_mode, learner):
    ball = standard_object("ball", BALL)
    table = standard_object("table", TABLE)

    run_preposition_test(
        learner(language_mode),
        _in_front_template(
            ball,
            table,
            immutableset(),
            is_training=True,
            is_near=True,
            speaker_root_node=MOM,
        ),
        language_generator=phase1_language_generator(language_mode),
    )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize(
    "learner",
    [pytest.mark.skip(subset_relation_language_factory), integrated_learner_factory],
)
def test_subset_preposition_has(language_mode, learner):
    person = standard_object("person", PERSON)
    cup = standard_object("cup", CUP)
    book = standard_object("book", BOOK)
    ball = standard_object("ball", BALL)

    language_generator = phase1_language_generator(language_mode)

    has_train_curriculum = []
    has_train_curriculum.extend(
        phase1_instances(
            "Has Unit Train",
            language_generator=language_generator,
            situations=sampled(
                _x_has_y_template(person, cup),
                chooser=PHASE1_CHOOSER_FACTORY(),
                ontology=GAILA_PHASE_1_ONTOLOGY,
                max_to_sample=1,
            ),
        ).instances()
    )
    has_train_curriculum.extend(
        phase1_instances(
            "Has Unit Train",
            language_generator=language_generator,
            situations=sampled(
                _x_has_y_template(person, book),
                chooser=PHASE1_CHOOSER_FACTORY(),
                ontology=GAILA_PHASE_1_ONTOLOGY,
                max_to_sample=1,
            ),
        ).instances()
    )

    has_test_curriculum = phase1_instances(
        "Has Unit Test",
        situations=sampled(
            _x_has_y_template(person, ball),
            chooser=PHASE1_CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            max_to_sample=1,
        ),
        language_generator=language_generator,
    )

    process_learner = learner(language_mode)
    for (_, linguistic_description, perceptual_representation) in has_train_curriculum:
        process_learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )

    for (
        _,
        test_lingustics_description,
        test_perceptual_representation,
    ) in has_test_curriculum.instances():
        descriptions_from_learner = process_learner.describe(
            test_perceptual_representation
        )
        gold = test_lingustics_description.as_token_sequence()
        assert descriptions_from_learner
        assert gold in [desc.as_token_sequence() for desc in descriptions_from_learner]
