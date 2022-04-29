import random
from adam.learner.integrated_learner import SymbolicIntegratedTemplateLearner
from adam.learner.relations import PursuitRelationLearner
from adam.ontology import IS_SPEAKER, IS_ADDRESSEE
import pytest
from adam.curriculum.phase1_curriculum import _x_has_y_template
from adam.ontology.phase2_ontology import GAILA_PHASE_2_ONTOLOGY
from immutablecollections import immutableset

from adam.curriculum.curriculum_utils import (
    standard_object,
    phase1_instances,
    CHOOSER_FACTORY,
)
from adam.curriculum.preposition_curriculum import (
    _on_template,
    _beside_template,
    _under_template,
    _over_template,
    _in_template,
    _behind_template,
    _in_front_template,
)
from adam.language.language_utils import phase1_language_generator
from adam.learner import LearningExample
from adam.learner.language_mode import LanguageMode
from adam.ontology.phase1_ontology import (
    BALL,
    TABLE,
    GAILA_PHASE_1_ONTOLOGY,
    WATER,
    CUP,
    MOM,
    PERSON,
    INANIMATE_OBJECT,
    PERSON_CAN_HAVE,
)
from adam.situation.templates.phase1_templates import sampled, object_variable
from tests.learner import LANGUAGE_MODE_TO_TEMPLATE_LEARNER_OBJECT_RECOGNIZER


def pursuit_learner_factory(language_mode: LanguageMode):
    rng = random.Random()
    rng.seed(0)
    return SymbolicIntegratedTemplateLearner(
        object_learner=LANGUAGE_MODE_TO_TEMPLATE_LEARNER_OBJECT_RECOGNIZER[language_mode],
        relation_learner=PursuitRelationLearner(
            learning_factor=0.05,
            graph_match_confirmation_threshold=0.7,
            lexicon_entry_threshold=0.7,
            rng=rng,
            smoothing_parameter=0.002,
            ontology=GAILA_PHASE_2_ONTOLOGY,
            language_mode=language_mode,
            min_continuous_feature_match_score=0.3,
            rank_gaze_higher=False,
        ),
    )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize("learner", [pursuit_learner_factory])
def test_pursuit_preposition_on_learner(language_mode, learner):
    ball = standard_object("ball", BALL)
    table = standard_object("table", TABLE)
    language_generator = phase1_language_generator(language_mode)
    on_train_curriculum = phase1_instances(
        "Preposition Unit Train",
        situations=sampled(
            _on_template(ball, table, immutableset(), is_training=True),
            chooser=CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            max_to_sample=10,
            block_multiple_of_the_same_type=True,
        ),
        language_generator=language_generator,
    )
    on_test_curriculum = phase1_instances(
        "Preposition Unit Test",
        situations=sampled(
            _on_template(ball, table, immutableset(), is_training=False),
            chooser=CHOOSER_FACTORY(),
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
    ) in on_train_curriculum.instances():
        # Get the object matches first - preposition learner can't learn without already recognized objects
        processing_learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )
    for (
        _,
        test_lingustics_description,
        test_perceptual_representation,
    ) in on_test_curriculum.instances():
        descriptions_from_learner = processing_learner.describe(
            test_perceptual_representation
        )
        gold = test_lingustics_description.as_token_sequence()
        assert descriptions_from_learner
        assert gold in [
            desc.as_token_sequence()
            for desc in descriptions_from_learner.description_to_confidence
        ]


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize("learner", [pursuit_learner_factory])
def test_pursuit_preposition_beside_learner(language_mode, learner):
    language_generator = phase1_language_generator(language_mode)
    ball = standard_object("ball", BALL)
    table = standard_object("table", TABLE)
    beside_train_curriculum = phase1_instances(
        "Preposition Beside Unit Train",
        situations=sampled(
            _beside_template(
                ball, table, immutableset(), is_training=True, is_right=True
            ),
            chooser=CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            max_to_sample=10,
            block_multiple_of_the_same_type=True,
        ),
        language_generator=language_generator,
    )
    beside_test_curriculum = phase1_instances(
        "Preposition Beside Unit Test",
        situations=sampled(
            _beside_template(
                ball, table, immutableset(), is_training=False, is_right=True
            ),
            chooser=CHOOSER_FACTORY(),
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
    ) in beside_train_curriculum.instances():
        processing_learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )

    for (
        _,
        test_linguistic_description,
        test_perceptual_representation,
    ) in beside_test_curriculum.instances():
        descriptions_from_learner = processing_learner.describe(
            test_perceptual_representation
        )
        gold = test_linguistic_description.as_token_sequence()
        assert descriptions_from_learner
        assert gold in [
            desc.as_token_sequence()
            for desc in descriptions_from_learner.description_to_confidence
        ]


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize("learner", [pursuit_learner_factory])
def test_pursuit_preposition_under_learner(language_mode, learner):
    ball = standard_object("ball", BALL)
    table = standard_object("table", TABLE)
    language_generator = phase1_language_generator(language_mode)
    under_train_curriculum = phase1_instances(
        "Preposition Under Unit Train",
        situations=sampled(
            _under_template(
                ball, table, immutableset(), is_training=True, is_distal=True
            ),
            chooser=CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            max_to_sample=10,
            block_multiple_of_the_same_type=True,
        ),
        language_generator=language_generator,
    )
    under_test_curriculum = phase1_instances(
        "Preposition Under Unit Test",
        situations=sampled(
            _under_template(
                ball, table, immutableset(), is_training=False, is_distal=True
            ),
            chooser=CHOOSER_FACTORY(),
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
    ) in under_train_curriculum.instances():
        processing_learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )

    for (
        _,
        test_linguistic_description,
        test_perceptual_representation,
    ) in under_test_curriculum.instances():
        descriptions_from_learner = processing_learner.describe(
            test_perceptual_representation
        )
        gold = test_linguistic_description.as_token_sequence()
        assert descriptions_from_learner
        assert gold in [
            desc.as_token_sequence()
            for desc in descriptions_from_learner.description_to_confidence
        ]


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize("learner", [pursuit_learner_factory])
def test_pursuit_preposition_over_learner(language_mode, learner):
    ball = standard_object("ball", BALL)
    table = standard_object("table", TABLE)
    language_generator = phase1_language_generator(language_mode)
    over_train_curriculum = phase1_instances(
        "Preposition Over Unit Train",
        situations=sampled(
            _over_template(ball, table, immutableset(), is_training=True, is_distal=True),
            chooser=CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            max_to_sample=10,
            block_multiple_of_the_same_type=True,
        ),
        language_generator=language_generator,
    )
    over_test_curriculum = phase1_instances(
        "Preposition Over Unit Test",
        situations=sampled(
            _over_template(
                ball, table, immutableset(), is_training=False, is_distal=True
            ),
            chooser=CHOOSER_FACTORY(),
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
    ) in over_train_curriculum.instances():
        processing_learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )

    for (
        _,
        test_linguistic_description,
        test_perceptual_representation,
    ) in over_test_curriculum.instances():
        descriptions_from_learner = processing_learner.describe(
            test_perceptual_representation
        )
        gold = test_linguistic_description.as_token_sequence()
        assert descriptions_from_learner
        assert gold in [
            desc.as_token_sequence()
            for desc in descriptions_from_learner.description_to_confidence
        ]


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize("learner", [pursuit_learner_factory])
def test_pursuit_preposition_in_learner(language_mode, learner):
    water = object_variable("water", WATER)
    cup = standard_object("cup", CUP)
    language_generator = phase1_language_generator(language_mode)
    in_train_curriculum = phase1_instances(
        "Preposition In Unit Train",
        situations=sampled(
            _in_template(water, cup, immutableset(), is_training=True),
            chooser=CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            max_to_sample=10,
            block_multiple_of_the_same_type=True,
        ),
        language_generator=language_generator,
    )
    in_test_curriculum = phase1_instances(
        "Preposition In Unit Test",
        situations=sampled(
            _in_template(water, cup, immutableset(), is_training=False),
            chooser=CHOOSER_FACTORY(),
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
    ) in in_train_curriculum.instances():
        processing_learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )

    for (
        _,
        test_linguistic_description,
        test_perceptual_representation,
    ) in in_test_curriculum.instances():
        descriptions_from_learner = processing_learner.describe(
            test_perceptual_representation
        )
        gold = test_linguistic_description.as_token_sequence()
        assert descriptions_from_learner
        assert gold in [
            desc.as_token_sequence()
            for desc in descriptions_from_learner.description_to_confidence
        ]


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize("learner", [pursuit_learner_factory])
def test_pursuit_preposition_behind_learner(language_mode, learner):
    ball = standard_object("ball", BALL)
    table = standard_object("table", TABLE)
    language_generator = phase1_language_generator(language_mode)
    speaker = standard_object("speaker", MOM, added_properties=[IS_SPEAKER])
    behind_train_curriculum = phase1_instances(
        "Preposition Behind Unit Train",
        situations=sampled(
            _behind_template(ball, table, [speaker], is_training=True, is_near=True),
            chooser=CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            max_to_sample=10,
            block_multiple_of_the_same_type=True,
        ),
        language_generator=language_generator,
    )
    behind_test_curriculum = phase1_instances(
        "Preposition Behind Unit Test",
        situations=sampled(
            _behind_template(ball, table, [speaker], is_training=False, is_near=True),
            chooser=CHOOSER_FACTORY(),
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
    ) in behind_train_curriculum.instances():
        processing_learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )

    for (
        _,
        test_linguistic_description,
        test_perceptual_representation,
    ) in behind_test_curriculum.instances():
        descriptions_from_learner = processing_learner.describe(
            test_perceptual_representation
        )
        gold = test_linguistic_description.as_token_sequence()
        assert descriptions_from_learner
        assert gold in [
            desc.as_token_sequence()
            for desc in descriptions_from_learner.description_to_confidence
        ]


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize("learner", [pursuit_learner_factory])
def test_pursuit_preposition_in_front_learner(language_mode, learner):
    ball = standard_object("ball", BALL)
    table = standard_object("table", TABLE)
    language_generator = phase1_language_generator(language_mode)
    speaker = standard_object("speaker", MOM, added_properties=[IS_SPEAKER])
    in_front_train_curriculum = phase1_instances(
        "Preposition In Front Unit Train",
        situations=sampled(
            _in_front_template(ball, table, [speaker], is_training=True, is_near=True),
            chooser=CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            max_to_sample=10,
            block_multiple_of_the_same_type=True,
        ),
        language_generator=language_generator,
    )
    in_front_test_curriculum = phase1_instances(
        "Preposition In Front Unit Test",
        situations=sampled(
            _in_front_template(ball, table, [speaker], is_training=False, is_near=True),
            chooser=CHOOSER_FACTORY(),
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
    ) in in_front_train_curriculum.instances():
        processing_learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )

    for (
        _,
        test_linguistic_description,
        test_perceptual_representation,
    ) in in_front_test_curriculum.instances():
        descriptions_from_learner = processing_learner.describe(
            test_perceptual_representation
        )
        gold = test_linguistic_description.as_token_sequence()
        assert descriptions_from_learner
        assert gold in [
            desc.as_token_sequence()
            for desc in descriptions_from_learner.description_to_confidence
        ]


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize("learner", [pursuit_learner_factory])
def test_pursuit_preposition_has_learner(language_mode, learner):
    person = standard_object(
        "person", PERSON, banned_properties=[IS_SPEAKER, IS_ADDRESSEE]
    )
    inanimate_object = standard_object(
        "inanimate-object", INANIMATE_OBJECT, required_properties=[PERSON_CAN_HAVE]
    )
    ball = standard_object("ball", BALL)

    language_generator = phase1_language_generator(language_mode)

    has_train_curriculum = phase1_instances(
        "Has Unit Train",
        situations=sampled(
            _x_has_y_template(person, inanimate_object),
            chooser=CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            max_to_sample=2,
            block_multiple_of_the_same_type=True,
        ),
        language_generator=language_generator,
    )

    has_test_curriculum = phase1_instances(
        "Has Unit Test",
        situations=sampled(
            _x_has_y_template(person, ball),
            chooser=CHOOSER_FACTORY(),
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
    ) in has_train_curriculum.instances():
        processing_learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )

    for (
        _,
        test_lingustics_description,
        test_perceptual_representation,
    ) in has_test_curriculum.instances():
        descriptions_from_learner = processing_learner.describe(
            test_perceptual_representation
        )
        gold = test_lingustics_description.as_token_sequence()
        assert descriptions_from_learner
        assert gold in [
            desc.as_token_sequence()
            for desc in descriptions_from_learner.description_to_confidence
        ]
