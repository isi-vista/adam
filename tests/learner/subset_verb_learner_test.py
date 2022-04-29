from itertools import chain

import pytest
from adam.curriculum.curriculum_utils import (
    CHOOSER_FACTORY,
    TEST_CHOOSER_FACTORY,
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
    make_throw_animacy_templates,
)
from adam.language.language_utils import phase1_language_generator
from adam.learner import LearningExample
from adam.learner.integrated_learner import SymbolicIntegratedTemplateLearner
from adam.learner.language_mode import LanguageMode
from adam.learner.verbs import SubsetVerbLearner
from adam.ontology import IS_SPEAKER, THING, IS_ADDRESSEE
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
    HOLLOW,
    PERSON_CAN_HAVE,
    LIQUID,
)
from adam.situation import Action
from adam.situation.templates.phase1_situation_templates import (
    _go_under_template,
    _jump_over_template,
)
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    sampled,
    object_variable,
)
from immutablecollections import immutableset
from tests.learner import LANGUAGE_MODE_TO_TEMPLATE_LEARNER_OBJECT_RECOGNIZER


def integrated_learner_factory(language_mode: LanguageMode):
    return SymbolicIntegratedTemplateLearner(
        object_learner=LANGUAGE_MODE_TO_TEMPLATE_LEARNER_OBJECT_RECOGNIZER[language_mode],
        action_learner=SubsetVerbLearner(
            ontology=GAILA_PHASE_1_ONTOLOGY,
            beam_size=5,
            language_mode=language_mode,
            min_continuous_feature_match_score=0.3,
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
                    max_to_sample=20,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=CHOOSER_FACTORY(),
                    block_multiple_of_the_same_type=True,
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
                    chooser=TEST_CHOOSER_FACTORY(),
                    block_multiple_of_the_same_type=True,
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
        assert gold in [
            desc.as_token_sequence()
            for desc in descriptions_from_learner.description_to_confidence
        ]


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize("learner", [integrated_learner_factory])
def test_eat_simple(language_mode, learner):
    object_to_eat = standard_object("object_0", required_properties=[EDIBLE])
    eater = standard_object(
        "eater_0",
        THING,
        required_properties=[ANIMATE],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )
    run_verb_test(
        learner(language_mode),
        make_eat_template(eater, object_to_eat),
        language_generator=phase1_language_generator(language_mode),
    )


def drink_test_template():
    object_0 = standard_object(
        "object_0",
        required_properties=[HOLLOW, PERSON_CAN_HAVE],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )
    liquid_0 = object_variable("liquid_0", required_properties=[LIQUID])
    person_0 = standard_object(
        "person_0", PERSON, banned_properties=[IS_SPEAKER, IS_ADDRESSEE]
    )
    return make_drink_template(person_0, liquid_0, object_0, None)


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize("learner", [integrated_learner_factory])
def test_drink(language_mode, learner):
    run_verb_test(
        learner(language_mode),
        drink_test_template(),
        language_generator=phase1_language_generator(language_mode),
    )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize("learner", [integrated_learner_factory])
def test_sit(language_mode, learner):
    for situation_template in make_sit_templates(None):
        run_verb_test(
            learner(language_mode),
            situation_template,
            language_generator=phase1_language_generator(language_mode),
        )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize("learner", [integrated_learner_factory])
def test_put(language_mode, learner):
    for situation_template in make_put_templates(None):
        run_verb_test(
            learner(language_mode),
            situation_template,
            language_generator=phase1_language_generator(language_mode),
        )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize("learner", [integrated_learner_factory])
def test_push(language_mode, learner):
    for situation_template in make_push_templates(
        agent=standard_object(
            "pusher",
            THING,
            required_properties=[ANIMATE],
            banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
        ),
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


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize("learner", [integrated_learner_factory])
def test_go(language_mode, learner):
    goer = standard_object("goer", THING, required_properties=[ANIMATE])
    under_goal_reference = standard_object(
        "go-under-goal", THING, required_properties=[HAS_SPACE_UNDER]
    )

    under_templates = [
        _go_under_template(goer, under_goal_reference, [], is_distal=is_distal)
        for is_distal in (True, False)
    ]

    for situation_template in make_go_templates(None):
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


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize("learner", [integrated_learner_factory])
def test_come(language_mode, learner):
    movee = standard_object(
        "movee",
        required_properties=[SELF_MOVING],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )
    learner_obj = standard_object("leaner_0", LEARNER)
    speaker = standard_object(
        "speaker",
        PERSON,
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
        added_properties=[IS_SPEAKER],
    )
    object_ = standard_object(
        "object_0", THING, banned_properties=[IS_SPEAKER, IS_ADDRESSEE]
    )
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


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize("learner", [integrated_learner_factory])
def test_take(language_mode, learner):
    run_verb_test(
        learner(language_mode),
        make_take_template(
            agent=standard_object(
                "taker_0",
                THING,
                required_properties=[ANIMATE],
                banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
            ),
            theme=standard_object("object_taken_0", required_properties=[INANIMATE]),
            use_adverbial_path_modifier=False,
        ),
        language_generator=phase1_language_generator(language_mode),
    )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize("learner", [integrated_learner_factory])
def test_give(language_mode, learner):
    for situation_template in make_give_templates(immutableset()):
        run_verb_test(
            learner(language_mode),
            situation_template,
            language_generator=phase1_language_generator(language_mode),
        )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize("learner", [integrated_learner_factory])
def test_spin(language_mode, learner):
    for situation_template in make_spin_templates(None):
        run_verb_test(
            learner(language_mode),
            situation_template,
            language_generator=phase1_language_generator(language_mode),
        )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize("learner", [integrated_learner_factory])
def test_fall(language_mode, learner):
    for situation_template in make_fall_templates(immutableset()):
        run_verb_test(
            learner(language_mode),
            situation_template,
            language_generator=phase1_language_generator(language_mode),
        )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize("learner", [integrated_learner_factory])
def test_throw(language_mode, learner):
    for situation_template in make_throw_templates(None):
        run_verb_test(
            learner(language_mode),
            situation_template,
            language_generator=phase1_language_generator(language_mode),
        )


@pytest.mark.parametrize(
    "language_mode",
    [LanguageMode.CHINESE, pytest.param(LanguageMode.ENGLISH, marks=pytest.mark.xfail)],
)
@pytest.mark.parametrize("learner", [integrated_learner_factory])
# this tests gei vs. dau X shang for Chinese throw to
# TODO: fix English implementation https://github.com/isi-vista/adam/issues/870
def test_throw_animacy(language_mode, learner):
    # shuffle both together for the train curriculum
    train_curriculum = phase1_instances(
        "train",
        chain(
            *[
                sampled(
                    situation_template=situation_template,
                    max_to_sample=10,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=CHOOSER_FACTORY(),
                    block_multiple_of_the_same_type=True,
                )
                for situation_template in make_throw_animacy_templates(None)
            ]
        ),
        language_generator=phase1_language_generator(language_mode),
    )
    # shuffle both together for test curriculum
    test_curriculum = phase1_instances(
        "test",
        chain(
            *[
                sampled(
                    situation_template=situation_template,
                    max_to_sample=1,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=CHOOSER_FACTORY(),
                    block_multiple_of_the_same_type=True,
                )
                for situation_template in make_throw_animacy_templates(None)
            ]
        ),
        language_generator=phase1_language_generator(language_mode),
    )
    # instantiate and test the learner
    learner = learner(language_mode)
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
        assert gold in [
            desc.as_token_sequence()
            for desc in descriptions_from_learner.description_to_confidence
        ]


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize("learner", [integrated_learner_factory])
def test_move(language_mode, learner):
    for situation_template in make_move_templates(None):
        run_verb_test(
            learner(language_mode),
            situation_template,
            language_generator=phase1_language_generator(language_mode),
        )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize("learner", [integrated_learner_factory])
def test_jump(language_mode, learner):

    jumper = standard_object(
        "jumper_0",
        THING,
        required_properties=[CAN_JUMP],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )
    jumped_over = standard_object(
        "jumped_over", banned_properties=[IS_SPEAKER, IS_ADDRESSEE]
    )

    for situation_template in make_jump_templates(None):

        run_verb_test(
            learner(language_mode),
            situation_template,
            language_generator=phase1_language_generator(language_mode),
        )
    for situation_template in [_jump_over_template(jumper, jumped_over, immutableset())]:
        run_verb_test(
            learner(language_mode),
            situation_template,
            language_generator=phase1_language_generator(language_mode),
        )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize("learner", [integrated_learner_factory])
def test_roll(language_mode, learner):
    for situation_template in make_roll_templates(None):
        run_verb_test(
            learner(language_mode),
            situation_template,
            language_generator=phase1_language_generator(language_mode),
        )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize("learner", [integrated_learner_factory])
def test_fly(language_mode, learner):
    for situation_template in make_fly_templates(immutableset()):
        run_verb_test(
            learner(language_mode),
            situation_template,
            phase1_language_generator(language_mode),
        )
