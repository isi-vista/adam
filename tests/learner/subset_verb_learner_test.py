from itertools import chain

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
from adam.curriculum.phase1_curriculum import (
    _make_come_down_template,
    make_drink_template,
    make_eat_template,
    make_fall_templates,
    make_fly_templates,
    make_give_templates,
    make_go_templates,
    make_jump_templates,
    make_move_templates,
    make_push_templates,
    make_put_templates,
    make_roll_templates,
    make_sit_templates,
    make_spin_templates,
    make_take_template,
    make_throw_templates,
)
from adam.learner import LearningExample
from adam.learner.integrated_learner import IntegratedTemplateLearner
from adam.learner.objects import ObjectRecognizerAsTemplateLearner
from adam.learner.verbs import SubsetVerbLearner, SubsetVerbLearnerNew
from adam.ontology import IS_SPEAKER, THING
from adam.ontology.phase1_ontology import (
    AGENT,
    ANIMATE,
    GAILA_PHASE_1_ONTOLOGY,
    GOAL,
    HAS_SPACE_UNDER,
    LEARNER,
    PERSON,
    GROUND,
    COME,
    CAN_JUMP,
    EDIBLE,
    SELF_MOVING,
)
from adam.situation import Action
from adam.situation.templates.phase1_situation_templates import (
    _go_under_template,
    _jump_over_template,
)
from adam.situation.templates.phase1_templates import Phase1SituationTemplate, sampled
from immutablecollections import immutableset
from tests.learner import TEST_OBJECT_RECOGNIZER


NOT_USED_LEARNER = IntegratedTemplateLearner(
    object_learner=ObjectRecognizerAsTemplateLearner(
        object_recognizer=TEST_OBJECT_RECOGNIZER
    ),
    action_learner=SubsetVerbLearnerNew(ontology=GAILA_PHASE_1_ONTOLOGY, beam_size=5),
)

LEARNER_FACTORIES = [
    lambda: SubsetVerbLearner(
        object_recognizer=TEST_OBJECT_RECOGNIZER, ontology=GAILA_PHASE_1_ONTOLOGY
    ),
    # TODO: implement the integrated learner for Chinese
    # lambda: IntegratedTemplateLearner(
    #    object_learner=ObjectRecognizerAsTemplateLearner(
    #        object_recognizer=TEST_OBJECT_RECOGNIZER
    #    ),
    #    action_learner=SubsetVerbLearnerNew(ontology=GAILA_PHASE_1_ONTOLOGY, beam_size=5),
    # ),
]

# VerbPursuitLearner(
#         learning_factor=0.5,
#         graph_match_confirmation_threshold=0.7,
#         lexicon_entry_threshold=0.7,
#         rng=rng,
#         smoothing_parameter=0.001,
#         ontology=GAILA_PHASE_1_ONTOLOGY,
#     )  # type: ignore


def run_verb_test(learner, situation_template, language_generator):
    train_curriculum = phase1_instances(
        "train",
        chain(
            *[
                sampled(
                    situation_template,
                    max_to_sample=10,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                )
            ]
        ),
        language_generator=language_generator,
    )
    test_curriculum = phase1_instances(
        "test",
        chain(
            *[
                sampled(
                    situation_template,
                    max_to_sample=1,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                )
            ]
        ),
        language_generator=language_generator,
    )

    for (
        _,
        linguistic_description,
        perceptual_representation,
    ) in train_curriculum.instances():
        # Get the object matches first - preposition learner can't learn without already recognized objects
        learner.observe(
            LearningExample(perceptual_representation, linguistic_description),
            language_generator=language_generator,
        )
    for (
        _,
        test_lingustics_description,
        test_perceptual_representation,
    ) in test_curriculum.instances():
        descriptions_from_learner = learner.describe(
            test_perceptual_representation, language_generator=language_generator
        )
        gold = test_lingustics_description.as_token_sequence()
        print(gold)
        assert descriptions_from_learner
        assert gold in [desc.as_token_sequence() for desc in descriptions_from_learner]


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
@pytest.mark.parametrize("learner_factory", LEARNER_FACTORIES)
def test_eat_simple(learner_factory, language_generator):
    learner = learner_factory()
    object_to_eat = standard_object("object_0", required_properties=[EDIBLE])
    eater = standard_object("eater_0", THING, required_properties=[ANIMATE])
    run_verb_test(
        learner,
        make_eat_template(eater, object_to_eat),
        language_generator=language_generator,
    )


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
@pytest.mark.parametrize("learner_factory", LEARNER_FACTORIES)
def test_drink(learner_factory, language_generator):
    learner = learner_factory()
    run_verb_test(learner, make_drink_template(), language_generator=language_generator)


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
@pytest.mark.parametrize("learner_factory", LEARNER_FACTORIES)
def test_sit(learner_factory, language_generator):
    for situation_template in make_sit_templates():
        learner = learner_factory()
        run_verb_test(learner, situation_template, language_generator=language_generator)


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR, GAILA_PHASE_1_LANGUAGE_GENERATOR],
)
@pytest.mark.parametrize("learner_factory", LEARNER_FACTORIES)
def test_put(learner_factory, language_generator):
    for situation_template in make_put_templates():
        learner = learner_factory()
        run_verb_test(learner, situation_template, language_generator=language_generator)


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
@pytest.mark.parametrize("learner_factory", LEARNER_FACTORIES)
def test_push(learner_factory, language_generator):
    for situation_template in make_push_templates():
        learner = learner_factory()
        run_verb_test(learner, situation_template, language_generator=language_generator)


# GO
@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
@pytest.mark.parametrize("learner_factory", LEARNER_FACTORIES)
def test_go(learner_factory, language_generator):
    goer = standard_object("goer", THING, required_properties=[ANIMATE])
    under_goal_reference = standard_object(
        "go-under-goal", THING, required_properties=[HAS_SPACE_UNDER]
    )

    under_templates = [
        _go_under_template(goer, under_goal_reference, [], is_distal=is_distal)
        for is_distal in (True, False)
    ]

    for situation_template in make_go_templates():
        learner = learner_factory()
        run_verb_test(learner, situation_template, language_generator=language_generator)

    for situation_template in under_templates:
        learner = learner_factory()
        run_verb_test(learner, situation_template, language_generator=language_generator)


# COME
@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
@pytest.mark.parametrize("learner_factory", LEARNER_FACTORIES)
def test_come(learner_factory, language_generator):
    movee = standard_object("movee", required_properties=[SELF_MOVING])
    learner = standard_object("leaner_0", LEARNER)
    speaker = standard_object("speaker", PERSON, added_properties=[IS_SPEAKER])
    object_ = standard_object("object_0", THING)
    ground = standard_object("ground", root_node=GROUND)

    come_to_speaker = Phase1SituationTemplate(
        "come-to-speaker",
        salient_object_variables=[movee, speaker],
        actions=[
            Action(COME, argument_roles_to_fillers=[(AGENT, movee), (GOAL, speaker)])
        ],
    )
    come_to_learner = Phase1SituationTemplate(
        "come-to-leaner",
        salient_object_variables=[movee],
        actions=[
            Action(COME, argument_roles_to_fillers=[(AGENT, movee), (GOAL, learner)])
        ],
    )
    come_to_object = Phase1SituationTemplate(
        "come-to-object",
        salient_object_variables=[movee, object_],
        actions=[
            Action(COME, argument_roles_to_fillers=[(AGENT, movee), (GOAL, object_)])
        ],
    )
    for situation_template in [
        _make_come_down_template(movee, object_, speaker, ground, immutableset()),
        come_to_speaker,
        come_to_learner,
        come_to_object,
    ]:
        learner = learner_factory()
        run_verb_test(learner, situation_template, language_generator=language_generator)


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
@pytest.mark.parametrize("learner_factory", LEARNER_FACTORIES)
def test_take(learner_factory, language_generator):
    learner = learner_factory()
    run_verb_test(learner, make_take_template(), language_generator=language_generator)


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
@pytest.mark.parametrize("learner_factory", LEARNER_FACTORIES)
def test_give(learner_factory, language_generator):
    for situation_template in make_give_templates():
        learner = learner_factory()
        run_verb_test(learner, situation_template, language_generator=language_generator)


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
@pytest.mark.parametrize("learner_factory", LEARNER_FACTORIES)
def test_spin(learner_factory, language_generator):
    for situation_template in make_spin_templates():
        learner = learner_factory()
        run_verb_test(learner, situation_template, language_generator=language_generator)


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
@pytest.mark.parametrize("learner_factory", LEARNER_FACTORIES)
def test_fall(learner_factory, language_generator):
    for situation_template in make_fall_templates():
        learner = learner_factory()
        run_verb_test(learner, situation_template, language_generator=language_generator)


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
@pytest.mark.parametrize("learner_factory", LEARNER_FACTORIES)
def test_throw(learner_factory, language_generator):
    for situation_template in make_throw_templates():
        learner = learner_factory()
        run_verb_test(learner, situation_template, language_generator=language_generator)


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
@pytest.mark.parametrize("learner_factory", LEARNER_FACTORIES)
def test_move(learner_factory, language_generator):
    for situation_template in make_move_templates():
        learner = learner_factory()
        run_verb_test(learner, situation_template, language_generator=language_generator)


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
@pytest.mark.parametrize("learner_factory", LEARNER_FACTORIES)
def test_jump(learner_factory, language_generator):
    jumper = standard_object("jumper_0", THING, required_properties=[CAN_JUMP])
    jumped_over = standard_object("jumped_over")
    for situation_template in make_jump_templates():
        learner = learner_factory()
        run_verb_test(learner, situation_template, language_generator=language_generator)
    for situation_template in [_jump_over_template(jumper, jumped_over, [])]:
        learner = learner_factory()
        run_verb_test(learner, situation_template, language_generator=language_generator)


@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
@pytest.mark.parametrize("learner_factory", LEARNER_FACTORIES)
def test_roll(learner_factory, language_generator):
    for situation_template in make_roll_templates():
        learner = learner_factory()
        run_verb_test(learner, situation_template, language_generator=language_generator)


# FLY
@pytest.mark.parametrize(
    "language_generator",
    [GAILA_PHASE_1_LANGUAGE_GENERATOR, GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR],
)
@pytest.mark.parametrize("learner_factory", LEARNER_FACTORIES)
def test_fly(learner_factory, language_generator):
    for situation_template in make_fly_templates():
        learner = learner_factory()
        run_verb_test(learner, situation_template, language_generator)
