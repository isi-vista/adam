import random
from itertools import chain
from typing import Iterable

import pytest

from adam.curriculum.curriculum_utils import (
    PHASE1_CHOOSER,
    phase1_instances,
    standard_object,
)
from adam.language_specific.english.english_language_generator import (
    USE_ADVERBIAL_PATH_MODIFIER,
)
from adam.learner import LearningExample
from adam.learner.verbs import SubsetVerbLearner
from adam.ontology import THING
from adam.ontology.phase1_ontology import (
    AGENT,
    COOKIE,
    EAT,
    GAILA_PHASE_1_ONTOLOGY,
    MOM,
    PATIENT,
    ANIMATE,
    CAN_HAVE_THINGS_RESTING_ON_THEM,
    INANIMATE_OBJECT,
    CAN_BE_SAT_ON_BY_PEOPLE,
    bigger_than,
    SIT,
    SIT_GOAL,
    SIT_THING_SAT_ON,
    GOAL,
)
from adam.ontology.phase1_spatial_relations import (
    Region,
    GRAVITATIONAL_UP,
    EXTERIOR_BUT_IN_CONTACT,
)
from adam.situation import Action
from adam.situation.templates.phase1_templates import Phase1SituationTemplate, sampled
from tests.learner import TEST_OBJECT_RECOGNIZER

LEARNER_FACTORIES = [
    lambda: SubsetVerbLearner(
        object_recognizer=TEST_OBJECT_RECOGNIZER, ontology=GAILA_PHASE_1_ONTOLOGY
    )
]

# VerbPursuitLearner(
#         learning_factor=0.5,
#         graph_match_confirmation_threshold=0.7,
#         lexicon_entry_threshold=0.7,
#         rng=rng,
#         smoothing_parameter=0.001,
#         ontology=GAILA_PHASE_1_ONTOLOGY,
#     )  # type: ignore


def run_verb_test(learner, situation_template):
    train_curriculum = phase1_instances(
        "train",
        chain(
            *[
                sampled(
                    situation_template,
                    max_to_sample=25,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                )
            ]
        ),
    )
    test_curriculum = phase1_instances(
        "test",
        chain(
            *[
                sampled(
                    situation_template,
                    max_to_sample=1,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                )
            ]
        ),
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
        assert [desc.as_token_sequence() for desc in descriptions_from_learner][0] == gold


@pytest.mark.parametrize("learner_factory", LEARNER_FACTORIES)
def test_eat_simple(learner_factory):
    learner = learner_factory()
    rng = random.Random()
    rng.seed(0)

    mom = standard_object("mom", MOM)
    cookie = standard_object("cookie", COOKIE)

    # "Mom eats a cookie"
    eat_template = Phase1SituationTemplate(
        "eat-object",
        salient_object_variables=[cookie, mom],
        actions=[
            Action(EAT, argument_roles_to_fillers=[(AGENT, mom), (PATIENT, cookie)])
        ],
    )
    run_verb_test(learner, eat_template)


# Infinite loop
# @pytest.mark.parametrize("learner_factory", LEARNER_FACTORIES)
# def test_drink(learner_factory):
#     learner = learner_factory()
#     rng = random.Random()
#     rng.seed(0)
#
#     object_0 = standard_object("object_0", required_properties=[HOLLOW])
#     liquid_0 = object_variable("liquid_0", required_properties=[LIQUID])
#     person_0 = standard_object("person_0", PERSON)
#
#     drink_template = Phase1SituationTemplate(
#         "drink",
#         salient_object_variables=[liquid_0, person_0],
#         actions=[
#             Action(
#                 DRINK,
#                 argument_roles_to_fillers=[(AGENT, person_0), (THEME, liquid_0)],
#                 auxiliary_variable_bindings=[(DRINK_CONTAINER_AUX, object_0)],
#             )
#         ],
#     )
#
#     run_verb_test(learner, drink_template)


@pytest.mark.parametrize("learner_factory", LEARNER_FACTORIES)
def test_sit(learner_factory):
    learner = learner_factory()

    sitter = standard_object("sitter_0", THING, required_properties=[ANIMATE])
    sit_surface = standard_object(
        "surface", THING, required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
    )
    seat = standard_object(
        "sitting-surface", INANIMATE_OBJECT, required_properties=[CAN_BE_SAT_ON_BY_PEOPLE]
    )

    def make_templates() -> Iterable[Phase1SituationTemplate]:
        # we need two groups of templates because in general something can sit on
        # anything bigger than itself which has a surface,
        # but people also sit in chairs, etc., which are smaller than them.
        sittee_to_contraints = (
            (  # type: ignore
                "-on-big-thing",
                sit_surface,
                [bigger_than(sit_surface, sitter)],
            ),
            ("-on-seat", seat, []),
        )

        syntax_hints_options = (
            ("default", []),  # type: ignore
            ("adverbial-mod", [USE_ADVERBIAL_PATH_MODIFIER]),
        )

        for (name, sittee, constraints) in sittee_to_contraints:
            for (syntax_name, syntax_hints) in syntax_hints_options:
                yield Phase1SituationTemplate(
                    f"sit-intransitive-{name}-{syntax_name}",
                    salient_object_variables=[sitter],
                    actions=[
                        Action(
                            SIT,
                            argument_roles_to_fillers=[(AGENT, sitter)],
                            auxiliary_variable_bindings=[
                                (
                                    SIT_GOAL,
                                    Region(
                                        sittee,
                                        direction=GRAVITATIONAL_UP,
                                        distance=EXTERIOR_BUT_IN_CONTACT,
                                    ),
                                ),
                                (SIT_THING_SAT_ON, sittee),
                            ],
                        )
                    ],
                    constraining_relations=constraints,
                    syntax_hints=syntax_hints,
                )

                yield Phase1SituationTemplate(
                    f"sit-transitive-{name}-{syntax_name}",
                    salient_object_variables=[sitter, sittee],
                    actions=[
                        Action(
                            SIT,
                            argument_roles_to_fillers=[
                                (AGENT, sitter),
                                (
                                    GOAL,
                                    Region(
                                        sittee,
                                        direction=GRAVITATIONAL_UP,
                                        distance=EXTERIOR_BUT_IN_CONTACT,
                                    ),
                                ),
                            ],
                            auxiliary_variable_bindings=[(SIT_THING_SAT_ON, sittee)],
                        )
                    ],
                    constraining_relations=constraints,
                    syntax_hints=syntax_hints,
                )

    for situation_template in make_templates():
        run_verb_test(learner, situation_template)
