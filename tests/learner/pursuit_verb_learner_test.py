import random
from itertools import chain

import pytest

from adam.curriculum.curriculum_utils import (PHASE1_CHOOSER, phase1_instances, standard_object)
from adam.learner import LearningExample
from adam.learner.verbs import SubsetVerbLearner
from adam.ontology.phase1_ontology import (AGENT, COOKIE, EAT, GAILA_PHASE_1_ONTOLOGY, MOM, PATIENT)
from adam.situation import Action
from adam.situation.templates.phase1_templates import Phase1SituationTemplate, sampled
from learner import TEST_OBJECT_RECOGNIZER

LEARNERS = [
    SubsetVerbLearner(
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


@pytest.mark.parametrize("learner", LEARNERS)
def test_eat(learner):
    rng = random.Random()
    rng.seed(0)

    mom = standard_object("mom", MOM)
    cookie = standard_object("cookie", COOKIE)

    # "Mom eats a cookie"
    eat_object = Phase1SituationTemplate(
        "eat-object",
        salient_object_variables=[cookie, mom],
        actions=[
            Action(EAT, argument_roles_to_fillers=[(AGENT, mom), (PATIENT, cookie)])
        ],
    )

    eat_train_curriculum = phase1_instances(
        "eating",
        chain(
            *[
                sampled(
                    eat_object,
                    max_to_sample=25,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                )
            ]
        ),
    )
    eat_test_curriculum = phase1_instances(
        "eating test",
        chain(
            *[
                sampled(
                    eat_object,
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
    ) in eat_train_curriculum.instances():
        # Get the object matches first - preposition learner can't learn without already recognized objects
        learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )

    for (
        _,
        test_lingustics_description,
        test_perceptual_representation,
    ) in eat_test_curriculum.instances():
        descriptions_from_learner = learner.describe(test_perceptual_representation)
        gold = test_lingustics_description.as_token_sequence()
        assert descriptions_from_learner
        assert [desc.as_token_sequence() for desc in descriptions_from_learner][0] == gold
