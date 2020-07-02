from itertools import chain

import pytest
from adam.curriculum.curriculum_utils import (
    PHASE1_CHOOSER_FACTORY,
    PHASE1_TEST_CHOOSER_FACTORY,
    phase1_instances,
    standard_object,
)
from adam.curriculum.phase1_curriculum import (
    _make_come_down_template,
    make_push_templates,
    make_drink_template,
    make_eat_template,
    make_fall_templates,
    make_fly_templates,
    make_give_templates,
    make_go_templates,
    make_jump_templates,
    make_move_templates,
    make_put_templates,
    make_roll_templates,
    make_sit_templates,
    make_spin_templates,
    make_take_template,
    make_throw_templates,
)
from adam.learner import LearningExample
from adam.learner.integrated_learner import IntegratedTemplateLearner
from adam.learner.language_mode import LanguageMode
from adam.learner.objects import ObjectRecognizerAsTemplateLearner
from adam.learner.verbs import SubsetVerbLearner, SubsetVerbLearnerNew
from adam.ontology import IS_SPEAKER, THING
from adam.ontology.phase1_ontology import (
    INANIMATE_OBJECT,
    CAN_HAVE_THINGS_RESTING_ON_THEM,
    INANIMATE,
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
from tests.learner import phase1_language_generator, object_recognizer_factory


def subset_verb_language_factory(language_mode: LanguageMode) -> SubsetVerbLearner:
    return SubsetVerbLearner(
        object_recognizer=object_recognizer_factory(language_mode),
        ontology=GAILA_PHASE_1_ONTOLOGY,
        language_mode=language_mode,
    )


def integrated_learner_factory(language_mode: LanguageMode):
    return IntegratedTemplateLearner(
        object_learner=ObjectRecognizerAsTemplateLearner(
            object_recognizer=object_recognizer_factory(language_mode),
            language_mode=language_mode,
        ),
        action_learner=SubsetVerbLearnerNew(
            ontology=GAILA_PHASE_1_ONTOLOGY, beam_size=5, language_mode=language_mode
        ),
    )


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
                    chooser=PHASE1_TEST_CHOOSER_FACTORY(),
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


@pytest.mark.parametrize(
    "language_mode",
    [LanguageMode.ENGLISH, pytest.param(LanguageMode.CHINESE, marks=pytest.mark.xfail)],
)
@pytest.mark.parametrize(
    "learner", [subset_verb_language_factory, integrated_learner_factory]
)
def test_eat_simple(language_mode, learner):
    object_to_eat = standard_object("object_0", required_properties=[EDIBLE])
    eater = standard_object("eater_0", THING, required_properties=[ANIMATE])
    run_verb_test(
        learner(language_mode),
        make_eat_template(eater, object_to_eat),
        language_generator=phase1_language_generator(language_mode),
    )


@pytest.mark.parametrize(
    "language_mode",
    [LanguageMode.ENGLISH, pytest.param(LanguageMode.CHINESE, marks=pytest.mark.xfail)],
)
@pytest.mark.parametrize(
    "learner", [subset_verb_language_factory, integrated_learner_factory]
)
def test_drink(language_mode, learner):
    run_verb_test(
        learner(language_mode),
        make_drink_template(),
        language_generator=phase1_language_generator(language_mode),
    )


@pytest.mark.parametrize(
    "language_mode",
    [LanguageMode.ENGLISH, pytest.param(LanguageMode.CHINESE, marks=pytest.mark.xfail)],
)
@pytest.mark.parametrize(
    "learner", [subset_verb_language_factory, integrated_learner_factory]
)
def test_sit(language_mode, learner):
    for situation_template in make_sit_templates():
        run_verb_test(
            learner(language_mode),
            situation_template,
            language_generator=phase1_language_generator(language_mode),
        )


@pytest.mark.parametrize(
    "language_mode",
    [LanguageMode.ENGLISH, pytest.param(LanguageMode.CHINESE, marks=pytest.mark.xfail)],
)
@pytest.mark.parametrize(
    "learner", [subset_verb_language_factory, integrated_learner_factory]
)
def test_put(language_mode, learner):
    for situation_template in make_put_templates():
        run_verb_test(
            learner(language_mode),
            situation_template,
            language_generator=phase1_language_generator(language_mode),
        )


@pytest.mark.parametrize(
    "language_mode",
    [LanguageMode.ENGLISH, pytest.param(LanguageMode.CHINESE, marks=pytest.mark.xfail)],
)
@pytest.mark.parametrize(
    "learner", [subset_verb_language_factory, integrated_learner_factory]
)
def test_push(language_mode, learner):
    for situation_template in make_push_templates(
        agent=standard_object("pusher", THING, required_properties=[ANIMATE]),
        theme=standard_object("pushee", INANIMATE_OBJECT),
        push_surface=standard_object(
            "push_surface", THING, required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
        ),
        push_goal=standard_object("push_goal", INANIMATE_OBJECT),
        use_adverbial_path_modifier=False,
    ):
        run_verb_test(
            learner(language_mode),
            situation_template,
            language_generator=phase1_language_generator(language_mode),
        )


@pytest.mark.parametrize(
    "language_mode",
    [LanguageMode.ENGLISH, pytest.param(LanguageMode.CHINESE, marks=pytest.mark.xfail)],
)
@pytest.mark.parametrize(
    "learner", [subset_verb_language_factory, integrated_learner_factory]
)
def test_go(language_mode, learner):
    goer = standard_object("goer", THING, required_properties=[ANIMATE])
    under_goal_reference = standard_object(
        "go-under-goal", THING, required_properties=[HAS_SPACE_UNDER]
    )

    under_templates = [
        _go_under_template(goer, under_goal_reference, [], is_distal=is_distal)
        for is_distal in (True, False)
    ]

    for situation_template in make_go_templates():
        run_verb_test(
            learner(language_mode),
            situation_template,
            language_generator=phase1_language_generator(language_mode),
        )

    for situation_template in under_templates:
        run_verb_test(
            learner(language_mode),
            situation_template,
            language_generator=phase1_language_generator(language_mode),
        )


@pytest.mark.parametrize(
    "language_mode",
    [LanguageMode.ENGLISH, pytest.param(LanguageMode.CHINESE, marks=pytest.mark.xfail)],
)
@pytest.mark.parametrize(
    "learner", [subset_verb_language_factory, integrated_learner_factory]
)
def test_come(language_mode, learner):
    movee = standard_object("movee", required_properties=[SELF_MOVING])
    learner_obj = standard_object("leaner_0", LEARNER)
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
            Action(COME, argument_roles_to_fillers=[(AGENT, movee), (GOAL, learner_obj)])
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
        run_verb_test(
            learner(language_mode),
            situation_template,
            language_generator=phase1_language_generator(language_mode),
        )


@pytest.mark.parametrize(
    "language_mode",
    [LanguageMode.ENGLISH, pytest.param(LanguageMode.CHINESE, marks=pytest.mark.xfail)],
)
@pytest.mark.parametrize(
    "learner", [subset_verb_language_factory, integrated_learner_factory]
)
def test_take(language_mode, learner):
    run_verb_test(
        learner(language_mode),
        make_take_template(
            agent=standard_object("taker_0", THING, required_properties=[ANIMATE]),
            theme=standard_object("object_taken_0", required_properties=[INANIMATE]),
            use_adverbial_path_modifier=False,
        ),
        language_generator=phase1_language_generator(language_mode),
    )


@pytest.mark.parametrize(
    "language_mode",
    [LanguageMode.ENGLISH, pytest.param(LanguageMode.CHINESE, marks=pytest.mark.xfail)],
)
@pytest.mark.parametrize(
    "learner", [subset_verb_language_factory, integrated_learner_factory]
)
def test_give(language_mode, learner):
    for situation_template in make_give_templates():
        run_verb_test(
            learner(language_mode),
            situation_template,
            language_generator=phase1_language_generator(language_mode),
        )


@pytest.mark.parametrize(
    "language_mode",
    [LanguageMode.ENGLISH, pytest.param(LanguageMode.CHINESE, marks=pytest.mark.xfail)],
)
@pytest.mark.parametrize(
    "learner", [subset_verb_language_factory, integrated_learner_factory]
)
def test_spin(language_mode, learner):
    for situation_template in make_spin_templates():
        run_verb_test(
            learner(language_mode),
            situation_template,
            language_generator=phase1_language_generator(language_mode),
        )


@pytest.mark.parametrize(
    "language_mode",
    [LanguageMode.ENGLISH, pytest.param(LanguageMode.CHINESE, marks=pytest.mark.xfail)],
)
@pytest.mark.parametrize(
    "learner", [subset_verb_language_factory, integrated_learner_factory]
)
def test_fall(language_mode, learner):
    for situation_template in make_fall_templates():
        run_verb_test(
            learner(language_mode),
            situation_template,
            language_generator=phase1_language_generator(language_mode),
        )


@pytest.mark.parametrize(
    "language_mode",
    [LanguageMode.ENGLISH, pytest.param(LanguageMode.CHINESE, marks=pytest.mark.xfail)],
)
@pytest.mark.parametrize(
    "learner", [subset_verb_language_factory, integrated_learner_factory]
)
def test_throw(language_mode, learner):
    for situation_template in make_throw_templates():
        run_verb_test(
            learner(language_mode),
            situation_template,
            language_generator=phase1_language_generator(language_mode),
        )


@pytest.mark.parametrize(
    "language_mode",
    [LanguageMode.ENGLISH, pytest.param(LanguageMode.CHINESE, marks=pytest.mark.xfail)],
)
@pytest.mark.parametrize(
    "learner", [subset_verb_language_factory, integrated_learner_factory]
)
def test_move(language_mode, learner):
    for situation_template in make_move_templates():
        run_verb_test(
            learner(language_mode),
            situation_template,
            language_generator=phase1_language_generator(language_mode),
        )


@pytest.mark.parametrize(
    "language_mode",
    [LanguageMode.ENGLISH, pytest.param(LanguageMode.CHINESE, marks=pytest.mark.xfail)],
)
@pytest.mark.parametrize(
    "learner", [subset_verb_language_factory, integrated_learner_factory]
)
def test_jump(language_mode, learner):
    jumper = standard_object("jumper_0", THING, required_properties=[CAN_JUMP])
    jumped_over = standard_object("jumped_over")
    for situation_template in make_jump_templates():
        run_verb_test(
            learner(language_mode),
            situation_template,
            language_generator=phase1_language_generator(language_mode),
        )
    for situation_template in [_jump_over_template(jumper, jumped_over, [])]:
        run_verb_test(
            learner(language_mode),
            situation_template,
            language_generator=phase1_language_generator(language_mode),
        )


@pytest.mark.parametrize(
    "language_mode",
    [LanguageMode.ENGLISH, pytest.param(LanguageMode.CHINESE, marks=pytest.mark.xfail)],
)
@pytest.mark.parametrize(
    "learner", [subset_verb_language_factory, integrated_learner_factory]
)
def test_roll(language_mode, learner):
    for situation_template in make_roll_templates():
        run_verb_test(
            learner(language_mode),
            situation_template,
            language_generator=phase1_language_generator(language_mode),
        )


@pytest.mark.parametrize(
    "language_mode",
    [LanguageMode.ENGLISH, pytest.param(LanguageMode.CHINESE, marks=pytest.mark.xfail)],
)
@pytest.mark.parametrize(
    "learner", [subset_verb_language_factory, integrated_learner_factory]
)
def test_fly(language_mode, learner):
    for situation_template in make_fly_templates():
        run_verb_test(
            learner(language_mode),
            situation_template,
            phase1_language_generator(language_mode),
        )
