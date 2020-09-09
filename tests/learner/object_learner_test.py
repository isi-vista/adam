import logging
import random
from typing import Iterable

import pytest
from immutablecollections import immutableset
from more_itertools import flatten

from adam.curriculum.curriculum_utils import PHASE1_TEST_CHOOSER_FACTORY
from adam.curriculum.phase1_curriculum import PHASE1_CHOOSER_FACTORY, phase1_instances
from adam.curriculum.pursuit_curriculum import make_simple_pursuit_curriculum
from adam.language.dependency import LinearizedDependencyTree
from adam.language.language_generator import LanguageGenerator
from adam.language.language_utils import phase1_language_generator
from adam.language_specific.english.english_language_generator import (
    GAILA_PHASE_1_LANGUAGE_GENERATOR,
    IGNORE_COLORS,
)
from adam.learner import (
    LanguagePerceptionSemanticAlignment,
    LearningExample,
    PerceptionSemanticAlignment,
)
from adam.learner.alignments import LanguageConceptAlignment
from adam.learner.integrated_learner import IntegratedTemplateLearner
from adam.learner.language_mode import LanguageMode
from adam.learner.objects import (
    PursuitObjectLearnerNew,
    SubsetObjectLearnerNew,
    ProposeButVerifyObjectLearner,
)
from adam.learner.objects import SubsetObjectLearner
from adam.ontology import OntologyNode
from adam.ontology.phase1_ontology import (
    BALL,
    BOX,
    DOG,
    GAILA_PHASE_1_ONTOLOGY,
    GROUND,
    HAND,
    HEAD,
    HOUSE,
    MOM,
    on,
)
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    GAILA_PHASE_1_PERCEPTION_GENERATOR,
)
from adam.perception.perception_graph import PerceptionGraph
from adam.random_utils import RandomChooser
from adam.relation import flatten_relations
from adam.relation_dsl import negate
from adam.semantics import ObjectSemanticNode
from adam.situation import SituationObject
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    all_possible,
    color_variable,
    object_variable,
    sampled,
)


def subset_object_learner_factory(language_mode: LanguageMode):
    return SubsetObjectLearner(
        ontology=GAILA_PHASE_1_ONTOLOGY, language_mode=language_mode
    )


def integrated_learner_factory(language_mode: LanguageMode):
    return IntegratedTemplateLearner(
        object_learner=SubsetObjectLearnerNew(
            ontology=GAILA_PHASE_1_ONTOLOGY, beam_size=10, language_mode=language_mode
        )
    )


def integrated_learner_pv_factory(langage_mode: LanguageMode):
    rng = RandomChooser.for_seed(0)
    return IntegratedTemplateLearner(
        object_learner=ProposeButVerifyObjectLearner(
            rng=rng, language_mode=langage_mode, ontology=GAILA_PHASE_1_ONTOLOGY
        )
    )


def run_subset_learner_for_object(
    nodes: Iterable[OntologyNode],
    *,
    learner,
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ]
):
    colored_obj_objects = [
        object_variable(
            "obj-with-color", node, added_properties=[color_variable("color")]
        )
        for node in nodes
    ]

    obj_templates = [
        Phase1SituationTemplate(
            "colored-obj-object",
            salient_object_variables=[colored_obj_object],
            syntax_hints=[IGNORE_COLORS],
        )
        for colored_obj_object in colored_obj_objects
    ]

    obj_curriculum = phase1_instances(
        "all obj situations",
        flatten(
            [
                all_possible(
                    obj_template,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                )
                for obj_template in obj_templates
            ]
        ),
        language_generator=language_generator,
    )

    test_obj_curriculum = phase1_instances(
        "obj test",
        situations=sampled(
            obj_templates[0],
            chooser=PHASE1_TEST_CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            max_to_sample=1,
            block_multiple_of_the_same_type=True,
        ),
        language_generator=language_generator,
    )

    for training_stage in [obj_curriculum]:
        for (
            _,
            linguistic_description,
            perceptual_representation,
        ) in training_stage.instances():
            learner.observe(
                LearningExample(perceptual_representation, linguistic_description)
            )

    for test_instance_group in [test_obj_curriculum]:
        for (
            _,
            test_instance_language,
            test_instance_perception,
        ) in test_instance_group.instances():
            descriptions_from_learner = learner.describe(test_instance_perception)
            gold = test_instance_language.as_token_sequence()
            print(gold)
            assert gold in [
                desc.as_token_sequence() for desc in descriptions_from_learner
            ]


# tests learning "ball" in both languages
@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize(
    "learner",
    [
        pytest.param(
            subset_object_learner_factory,
            marks=pytest.mark.skip("No Longer Need to Test Old Learners"),
        ),
        integrated_learner_factory,
        integrated_learner_pv_factory,
    ],
)
def test_subset_learner(language_mode, learner):
    run_subset_learner_for_object(
        [BALL, DOG, BOX],
        learner=learner(language_mode),
        language_generator=phase1_language_generator(language_mode),
    )


# TODO: Figure out how to run this test over both chinese and english well
def test_subset_learner_subobject():
    mom = SituationObject.instantiate_ontology_node(
        ontology_node=MOM, ontology=GAILA_PHASE_1_ONTOLOGY
    )
    head = SituationObject.instantiate_ontology_node(
        ontology_node=HEAD, ontology=GAILA_PHASE_1_ONTOLOGY
    )
    hand = SituationObject.instantiate_ontology_node(
        ontology_node=HAND, ontology=GAILA_PHASE_1_ONTOLOGY
    )
    ball = SituationObject.instantiate_ontology_node(
        ontology_node=BALL, ontology=GAILA_PHASE_1_ONTOLOGY
    )
    house = SituationObject.instantiate_ontology_node(
        ontology_node=HOUSE, ontology=GAILA_PHASE_1_ONTOLOGY
    )
    ground = SituationObject.instantiate_ontology_node(
        ontology_node=GROUND, ontology=GAILA_PHASE_1_ONTOLOGY
    )

    mom_situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=immutableset([mom])
    )

    floating_head_situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=immutableset([head]),
        other_objects=immutableset([ground]),
        always_relations=flatten_relations(negate(on(head, ground))),
    )

    # Need to include some extra situations so that the learner will prune its semantics for 'a'
    # away and not recognize it as an object.
    floating_hand_situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=immutableset([hand]),
        other_objects=immutableset([ground]),
        always_relations=flatten_relations(negate(on(hand, ground))),
    )

    floating_ball_situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=immutableset([ball]),
        other_objects=immutableset([ground]),
        always_relations=flatten_relations(negate(on(ball, ground))),
    )

    floating_house_situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=immutableset([house]),
        other_objects=immutableset([ground]),
        always_relations=flatten_relations(negate(on(house, ground))),
    )

    object_learner = SubsetObjectLearnerNew(
        ontology=GAILA_PHASE_1_ONTOLOGY, beam_size=5, language_mode=LanguageMode.ENGLISH
    )

    for situation in [
        mom_situation,
        floating_head_situation,
        floating_hand_situation,
        floating_ball_situation,
        floating_house_situation,
    ]:
        perceptual_representation = GAILA_PHASE_1_PERCEPTION_GENERATOR.generate_perception(
            situation, chooser=RandomChooser.for_seed(0)
        )
        for linguistic_description in GAILA_PHASE_1_LANGUAGE_GENERATOR.generate_language(
            situation, chooser=RandomChooser.for_seed(0)
        ):
            perception_graph = PerceptionGraph.from_frame(
                perceptual_representation.frames[0]
            )

            object_learner.learn_from(
                LanguagePerceptionSemanticAlignment(
                    language_concept_alignment=LanguageConceptAlignment.create_unaligned(
                        language=linguistic_description
                    ),
                    perception_semantic_alignment=PerceptionSemanticAlignment(
                        perception_graph=perception_graph, semantic_nodes=[]
                    ),
                )
            )

    mom_perceptual_representation = GAILA_PHASE_1_PERCEPTION_GENERATOR.generate_perception(
        mom_situation, chooser=RandomChooser.for_seed(0)
    )
    perception_graph = PerceptionGraph.from_frame(mom_perceptual_representation.frames[0])
    enriched = object_learner.enrich_during_description(
        PerceptionSemanticAlignment.create_unaligned(perception_graph)
    )

    semantic_node_types_and_debug_strings = {
        (type(semantic_node), semantic_node.concept.debug_string)
        for semantic_node in enriched.semantic_nodes
    }
    assert (ObjectSemanticNode, "Mom") in semantic_node_types_and_debug_strings
    assert (ObjectSemanticNode, "head") in semantic_node_types_and_debug_strings
    assert (ObjectSemanticNode, "hand") in semantic_node_types_and_debug_strings


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
def test_pursuit_object_learner(language_mode):
    target_objects = [
        BALL,
        # PERSON,
        # CHAIR,
        # TABLE,
        DOG,
        # BIRD,
        BOX,
    ]

    language_generator = phase1_language_generator(language_mode)

    target_test_templates = []
    for obj in target_objects:
        # Create train and test templates for the target objects
        test_obj_object = object_variable("obj-with-color", obj)
        test_template = Phase1SituationTemplate(
            "colored-obj-object",
            salient_object_variables=[test_obj_object],
            syntax_hints=[IGNORE_COLORS],
        )
        target_test_templates.extend(
            all_possible(
                test_template,
                chooser=PHASE1_CHOOSER_FACTORY(),
                ontology=GAILA_PHASE_1_ONTOLOGY,
            )
        )
    rng = random.Random()
    rng.seed(0)

    # We can use this to generate the actual pursuit curriculum
    train_curriculum = make_simple_pursuit_curriculum(
        target_objects=target_objects,
        num_instances=30,
        num_objects_in_instance=3,
        num_noise_instances=0,
        language_generator=language_generator,
    )

    test_obj_curriculum = phase1_instances(
        "obj test",
        situations=target_test_templates,
        language_generator=language_generator,
    )

    # All parameters should be in the range 0-1.
    # Learning factor works better when kept < 0.5
    # Graph matching threshold doesn't seem to matter that much, as often seems to be either a
    # complete or a very small match.
    # The lexicon threshold works better between 0.07-0.3, but we need to play around with it because we end up not
    # lexicalize items sufficiently because of diminishing lexicon probability through training
    rng = random.Random()
    rng.seed(0)
    learner = IntegratedTemplateLearner(
        object_learner=PursuitObjectLearnerNew(
            learning_factor=0.05,
            graph_match_confirmation_threshold=0.7,
            lexicon_entry_threshold=0.7,
            rng=rng,
            smoothing_parameter=0.002,
            ontology=GAILA_PHASE_1_ONTOLOGY,
            language_mode=language_mode,
        )
    )
    for training_stage in [train_curriculum]:
        for (
            _,
            linguistic_description,
            perceptual_representation,
        ) in training_stage.instances():
            learner.observe(
                LearningExample(perceptual_representation, linguistic_description)
            )

    for test_instance_group in [test_obj_curriculum]:
        for (
            _,
            test_instance_language,
            test_instance_perception,
        ) in test_instance_group.instances():
            logging.info("lang: %s", test_instance_language)
            descriptions_from_learner = learner.describe(test_instance_perception)
            gold = test_instance_language.as_token_sequence()
            assert gold in [
                desc.as_token_sequence() for desc in descriptions_from_learner
            ]


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
def test_pursuit_object_learner_with_gaze(language_mode):
    target_objects = [
        BALL,
        # PERSON,
        # CHAIR,
        # TABLE,
        DOG,
        # BIRD,
        BOX,
    ]

    language_generator = phase1_language_generator(language_mode)

    target_test_templates = []
    for obj in target_objects:
        # Create train and test templates for the target objects
        test_obj_object = object_variable("obj-with-color", obj)
        test_template = Phase1SituationTemplate(
            "colored-obj-object",
            salient_object_variables=[test_obj_object],
            syntax_hints=[IGNORE_COLORS],
            gazed_objects=[test_obj_object],
        )
        target_test_templates.extend(
            all_possible(
                test_template,
                chooser=PHASE1_CHOOSER_FACTORY(),
                ontology=GAILA_PHASE_1_ONTOLOGY,
            )
        )
    rng = random.Random()
    rng.seed(0)

    # We can use this to generate the actual pursuit curriculum
    train_curriculum = make_simple_pursuit_curriculum(
        target_objects=target_objects,
        num_instances=30,
        num_objects_in_instance=3,
        num_noise_instances=0,
        language_generator=language_generator,
        add_gaze=True,
    )

    test_obj_curriculum = phase1_instances(
        "obj test",
        situations=target_test_templates,
        language_generator=language_generator,
    )

    # All parameters should be in the range 0-1.
    # Learning factor works better when kept < 0.5
    # Graph matching threshold doesn't seem to matter that much, as often seems to be either a
    # complete or a very small match.
    # The lexicon threshold works better between 0.07-0.3, but we need to play around with it because we end up not
    # lexicalize items sufficiently because of diminishing lexicon probability through training
    rng = random.Random()
    rng.seed(0)
    learner = IntegratedTemplateLearner(
        object_learner=PursuitObjectLearnerNew(
            learning_factor=0.05,
            graph_match_confirmation_threshold=0.7,
            lexicon_entry_threshold=0.7,
            rng=rng,
            smoothing_parameter=0.002,
            ontology=GAILA_PHASE_1_ONTOLOGY,
            language_mode=language_mode,
            rank_gaze_higher=True,
        )
    )
    for training_stage in [train_curriculum]:
        for (
            _,
            linguistic_description,
            perceptual_representation,
        ) in training_stage.instances():
            learner.observe(
                LearningExample(perceptual_representation, linguistic_description)
            )

    for test_instance_group in [test_obj_curriculum]:
        for (
            _,
            test_instance_language,
            test_instance_perception,
        ) in test_instance_group.instances():
            logging.info("lang: %s", test_instance_language)
            descriptions_from_learner = learner.describe(test_instance_perception)
            gold = test_instance_language.as_token_sequence()
            assert gold in [
                desc.as_token_sequence() for desc in descriptions_from_learner
            ]
