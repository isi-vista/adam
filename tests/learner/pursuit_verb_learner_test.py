import random
from itertools import chain

from more_itertools import first

from adam.curriculum.curriculum_utils import (
    standard_object,
    phase1_instances,
    PHASE1_CHOOSER,
)
from adam.learner import LearningExample
from adam.learner.object_recognizer import ObjectRecognizer
from adam.learner.verbs import VerbPursuitLearner
from adam.ontology.phase1_ontology import (
    GAILA_PHASE_1_ONTOLOGY,
    EAT,
    AGENT,
    PATIENT,
    MOM,
    COOKIE,
)
from adam.perception.perception_graph import PerceptionGraphPattern, TemporalScope
from adam.situation import Action
from adam.situation.templates.phase1_templates import sampled, Phase1SituationTemplate


def test_pursuit_verb_eat_learner():
    rng = random.Random()
    rng.seed(0)
    learner = VerbPursuitLearner(
        learning_factor=0.5,
        graph_match_confirmation_threshold=0.7,
        lexicon_entry_threshold=0.7,
        rng=rng,
        smoothing_parameter=0.001,
        ontology=GAILA_PHASE_1_ONTOLOGY,
    )  # type: ignore

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

    # Set up object recognizer, given the two objects we 'already' recognize
    object_recognizer = ObjectRecognizer(
        {
            node.handle: PerceptionGraphPattern.from_schema(
                first(GAILA_PHASE_1_ONTOLOGY.structural_schemata(node))
            ).copy_with_temporal_scopes(
                required_temporal_scopes=[TemporalScope.BEFORE, TemporalScope.AFTER]
            )
            for node in [MOM, COOKIE]
        }
    )

    for (
        _,
        linguistic_description,
        perceptual_representation,
    ) in eat_train_curriculum.instances():
        # Get the object matches first - preposition learner can't learn without already recognized objects
        learner.observe(
            LearningExample(perceptual_representation, linguistic_description),
            object_recognizer=object_recognizer,
        )

    for (
        _,
        test_lingustics_description,
        test_perceptual_representation,
    ) in eat_test_curriculum.instances():
        descriptions_from_learner = learner.describe(
            test_perceptual_representation, object_recognizer
        )
        gold = test_lingustics_description.as_token_sequence()
        assert descriptions_from_learner
        assert [desc.as_token_sequence() for desc in descriptions_from_learner][0] == gold
