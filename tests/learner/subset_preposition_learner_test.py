import pytest
from adam.language_specific.english.english_language_generator import (
    GAILA_PHASE_1_LANGUAGE_GENERATOR,
)
from adam.language_specific.chinese.chinese_language_generator import (
    GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR,
)
from adam.curriculum.curriculum_utils import (
    PHASE1_CHOOSER_FACTORY,
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
from adam.learner import LearningExample
from adam.learner.integrated_learner import IntegratedTemplateLearner
from adam.learner.objects import ObjectRecognizerAsTemplateLearner
from adam.learner.prepositions import SubsetPrepositionLearner
from adam.learner.relations import SubsetRelationLearnerNew
from adam.ontology import IS_ADDRESSEE, IS_SPEAKER
from adam.ontology.phase1_ontology import (
    BALL,
    BOOK,
    CUP,
    GAILA_PHASE_1_ONTOLOGY,
    LEARNER,
    MOM,
    PERSON,
    TABLE,
    WATER,
)
from adam.situation.templates.phase1_templates import object_variable, sampled
from immutablecollections import immutableset
from tests.learner import TEST_OBJECT_RECOGNIZER

OLD_SUBSET_PREPOSITION_LEARNER_FACTORY = lambda: SubsetPrepositionLearner(
    object_recognizer=TEST_OBJECT_RECOGNIZER, ontology=GAILA_PHASE_1_ONTOLOGY
)
NEW_SUBSET_RELATION_LEARNER_FACTORY = lambda: IntegratedTemplateLearner(
    object_learner=ObjectRecognizerAsTemplateLearner(
        object_recognizer=TEST_OBJECT_RECOGNIZER
    ),
    relation_learner=SubsetRelationLearnerNew(
        ontology=GAILA_PHASE_1_ONTOLOGY, beam_size=5
    ),
)

# TODO: fix Chinese for Integrated learner
LEARNER_FACTORIES = [
    OLD_SUBSET_PREPOSITION_LEARNER_FACTORY,
    # NEW_SUBSET_RELATION_LEARNER_FACTORY,
]


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
@pytest.mark.parametrize("learner_factory", LEARNER_FACTORIES)
def test_subset_preposition_on_learner(learner_factory, language_generator):
    ball = standard_object("ball", BALL)
    table = standard_object("table", TABLE)
    on_train_curriculum = phase1_instances(
        "Preposition Unit Train",
        situations=sampled(
            _on_template(ball, table, immutableset(), is_training=True),
            chooser=PHASE1_CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            max_to_sample=2,
        ),
        language_generator=language_generator,
    )
    on_test_curriculum = phase1_instances(
        "Preposition Unit Test",
        situations=sampled(
            _on_template(ball, table, immutableset(), is_training=False),
            chooser=PHASE1_CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            max_to_sample=1,
        ),
        language_generator=language_generator,
    )

    learner = learner_factory()
    for (
        _,
        linguistic_description,
        perceptual_representation,
    ) in on_train_curriculum.instances():
        learner.observe(
            LearningExample(perceptual_representation, linguistic_description),
            language_generator=language_generator,
        )

    for (
        _,
        test_lingustics_description,
        test_perceptual_representation,
    ) in on_test_curriculum.instances():
        descriptions_from_learner = learner.describe(
            test_perceptual_representation, language_generator=language_generator
        )
        gold = test_lingustics_description.as_token_sequence()
        assert descriptions_from_learner
        assert gold in [desc.as_token_sequence() for desc in descriptions_from_learner]


@pytest.mark.parametrize("learner_factory", LEARNER_FACTORIES)
@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_subset_preposition_beside_learner(learner_factory, language_generator):
    ball = standard_object("ball", BALL)
    table = standard_object("table", TABLE)
    beside_train_curriculum = phase1_instances(
        "Preposition Beside Unit Train",
        situations=sampled(
            _beside_template(
                ball, table, immutableset(), is_training=True, is_right=True
            ),
            chooser=PHASE1_CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            max_to_sample=2,
        ),
        language_generator=language_generator,
    )
    beside_test_curriculum = phase1_instances(
        "Preposition Beside Unit Test",
        situations=sampled(
            _beside_template(
                ball, table, immutableset(), is_training=False, is_right=True
            ),
            chooser=PHASE1_CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            max_to_sample=1,
        ),
        language_generator=language_generator,
    )

    learner = learner_factory()
    for (
        _,
        linguistic_description,
        perceptual_representation,
    ) in beside_train_curriculum.instances():
        learner.observe(
            LearningExample(perceptual_representation, linguistic_description),
            language_generator=language_generator,
        )

    for (
        _,
        test_linguistic_description,
        test_perceptual_representation,
    ) in beside_test_curriculum.instances():
        descriptions_from_learner = learner.describe(
            test_perceptual_representation, language_generator=language_generator
        )
        gold = test_linguistic_description.as_token_sequence()
        assert descriptions_from_learner
        assert gold in [desc.as_token_sequence() for desc in descriptions_from_learner]


@pytest.mark.parametrize("learner_factory", LEARNER_FACTORIES)
@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
def test_subset_preposition_under_learner(learner_factory, language_generator):
    ball = standard_object("ball", BALL)
    table = standard_object("table", TABLE)
    under_train_curriculum = phase1_instances(
        "Preposition Under Unit Train",
        situations=sampled(
            _under_template(
                ball, table, immutableset(), is_training=True, is_distal=True
            ),
            chooser=PHASE1_CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            max_to_sample=2,
        ),
        language_generator=language_generator,
    )
    under_test_curriculum = phase1_instances(
        "Preposition Under Unit Test",
        situations=sampled(
            _under_template(
                ball, table, immutableset(), is_training=False, is_distal=True
            ),
            chooser=PHASE1_CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            max_to_sample=1,
        ),
        language_generator=language_generator,
    )

    learner = learner_factory()
    for (
        _,
        linguistic_description,
        perceptual_representation,
    ) in under_train_curriculum.instances():
        learner.observe(
            LearningExample(perceptual_representation, linguistic_description),
            language_generator=language_generator,
        )

    for (
        _,
        test_linguistic_description,
        test_perceptual_representation,
    ) in under_test_curriculum.instances():
        descriptions_from_learner = learner.describe(
            test_perceptual_representation, language_generator=language_generator
        )
        gold = test_linguistic_description.as_token_sequence()
        assert descriptions_from_learner
        assert gold in [desc.as_token_sequence() for desc in descriptions_from_learner]


@pytest.mark.parametrize("learner_factory", LEARNER_FACTORIES)
@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_subset_preposition_over_learner(learner_factory, language_generator):
    ball = standard_object("ball", BALL)
    table = standard_object("table", TABLE)
    over_train_curriculum = phase1_instances(
        "Preposition Over Unit Train",
        situations=sampled(
            _over_template(ball, table, immutableset(), is_training=True, is_distal=True),
            chooser=PHASE1_CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            max_to_sample=2,
        ),
        language_generator=language_generator,
    )
    over_test_curriculum = phase1_instances(
        "Preposition Over Unit Test",
        situations=sampled(
            _over_template(
                ball, table, immutableset(), is_training=False, is_distal=True
            ),
            chooser=PHASE1_CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            max_to_sample=1,
        ),
        language_generator=language_generator,
    )

    learner = learner_factory()
    for (
        _,
        linguistic_description,
        perceptual_representation,
    ) in over_train_curriculum.instances():
        learner.observe(
            LearningExample(perceptual_representation, linguistic_description),
            language_generator=language_generator,
        )

    for (
        _,
        test_linguistic_description,
        test_perceptual_representation,
    ) in over_test_curriculum.instances():
        descriptions_from_learner = learner.describe(
            test_perceptual_representation, language_generator=language_generator
        )
        gold = test_linguistic_description.as_token_sequence()
        assert descriptions_from_learner
        assert gold in [desc.as_token_sequence() for desc in descriptions_from_learner]


@pytest.mark.parametrize("learner_factory", LEARNER_FACTORIES)
@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_subset_preposition_in_learner(learner_factory, language_generator):
    water = object_variable("water", WATER)
    cup = standard_object("cup", CUP)
    in_train_curriculum = phase1_instances(
        "Preposition In Unit Train",
        situations=sampled(
            _in_template(water, cup, immutableset(), is_training=True),
            chooser=PHASE1_CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            max_to_sample=2,
        ),
        language_generator=language_generator,
    )
    in_test_curriculum = phase1_instances(
        "Preposition In Unit Test",
        situations=sampled(
            _in_template(water, cup, immutableset(), is_training=False),
            chooser=PHASE1_CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            max_to_sample=1,
        ),
        language_generator=language_generator,
    )

    learner = learner_factory()

    for (
        _,
        linguistic_description,
        perceptual_representation,
    ) in in_train_curriculum.instances():
        learner.observe(
            LearningExample(perceptual_representation, linguistic_description),
            language_generator=language_generator,
        )

    for (
        _,
        test_linguistic_description,
        test_perceptual_representation,
    ) in in_test_curriculum.instances():
        descriptions_from_learner = learner.describe(
            test_perceptual_representation, language_generator=language_generator
        )
        gold = test_linguistic_description.as_token_sequence()
        assert descriptions_from_learner
        assert gold in [desc.as_token_sequence() for desc in descriptions_from_learner]


@pytest.mark.parametrize("learner_factory", LEARNER_FACTORIES)
@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_subset_preposition_behind_learner(learner_factory, language_generator):
    ball = standard_object("ball", BALL)
    table = standard_object("table", TABLE)
    learner_object = standard_object("learner", LEARNER, added_properties=[IS_ADDRESSEE])
    mom = standard_object("mom", MOM, added_properties=[IS_SPEAKER])
    behind_train_curriculum = phase1_instances(
        "Preposition Behind Unit Train",
        situations=sampled(
            _behind_template(
                ball,
                table,
                immutableset([learner_object, mom]),
                is_training=True,
                is_near=True,
            ),
            chooser=PHASE1_CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            max_to_sample=2,
        ),
        language_generator=language_generator,
    )
    behind_test_curriculum = phase1_instances(
        "Preposition Behind Unit Test",
        situations=sampled(
            _behind_template(
                ball,
                table,
                immutableset([learner_object, mom]),
                is_training=False,
                is_near=True,
            ),
            chooser=PHASE1_CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            max_to_sample=1,
        ),
        language_generator=language_generator,
    )

    learner = learner_factory()
    for (
        _,
        linguistic_description,
        perceptual_representation,
    ) in behind_train_curriculum.instances():
        learner.observe(
            LearningExample(perceptual_representation, linguistic_description),
            language_generator=language_generator,
        )

    for (
        _,
        test_linguistic_description,
        test_perceptual_representation,
    ) in behind_test_curriculum.instances():
        descriptions_from_learner = learner.describe(
            test_perceptual_representation, language_generator=language_generator
        )
        gold = test_linguistic_description.as_token_sequence()
        assert descriptions_from_learner
        assert gold in [desc.as_token_sequence() for desc in descriptions_from_learner]


@pytest.mark.parametrize("learner_factory", LEARNER_FACTORIES)
@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_subset_preposition_in_front_learner(learner_factory, language_generator):
    ball = standard_object("ball", BALL)
    table = standard_object("table", TABLE)
    learner_object = standard_object("learner", LEARNER, added_properties=[IS_ADDRESSEE])
    mom = standard_object("mom", MOM, added_properties=[IS_SPEAKER])
    in_front_train_curriculum = phase1_instances(
        "Preposition In Front Unit Train",
        situations=sampled(
            _in_front_template(
                ball,
                table,
                immutableset([learner_object, mom]),
                is_training=True,
                is_near=True,
            ),
            chooser=PHASE1_CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            max_to_sample=2,
        ),
        language_generator=language_generator,
    )
    in_front_test_curriculum = phase1_instances(
        "Preposition In Front Unit Test",
        situations=sampled(
            _in_front_template(
                ball,
                table,
                immutableset([learner_object, mom]),
                is_training=False,
                is_near=True,
            ),
            chooser=PHASE1_CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            max_to_sample=1,
        ),
        language_generator=language_generator,
    )

    learner = learner_factory()
    for (
        _,
        linguistic_description,
        perceptual_representation,
    ) in in_front_train_curriculum.instances():
        learner.observe(
            LearningExample(perceptual_representation, linguistic_description),
            language_generator=language_generator,
        )

    for (
        _,
        test_linguistic_description,
        test_perceptual_representation,
    ) in in_front_test_curriculum.instances():
        descriptions_from_learner = learner.describe(
            test_perceptual_representation, language_generator=language_generator
        )
        gold = test_linguistic_description.as_token_sequence()
        assert descriptions_from_learner
        assert gold in [desc.as_token_sequence() for desc in descriptions_from_learner]


@pytest.mark.parametrize("learner_factory", LEARNER_FACTORIES)
@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
def test_subset_preposition_has_learner(learner_factory, language_generator):
    person = standard_object("person", PERSON)
    cup = standard_object("cup", CUP)
    book = standard_object("book", BOOK)
    ball = standard_object("ball", BALL)

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

    learner = learner_factory()
    for (_, linguistic_description, perceptual_representation) in has_train_curriculum:
        learner.observe(
            LearningExample(perceptual_representation, linguistic_description),
            language_generator=language_generator,
        )

    for (
        _,
        test_lingustics_description,
        test_perceptual_representation,
    ) in has_test_curriculum.instances():
        descriptions_from_learner = learner.describe(
            test_perceptual_representation, language_generator=language_generator
        )
        gold = test_lingustics_description.as_token_sequence()
        assert descriptions_from_learner
        assert gold in [desc.as_token_sequence() for desc in descriptions_from_learner]
