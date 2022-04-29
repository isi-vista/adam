import logging
from platform import python_implementation
import random
from typing import Any, Iterable, Sequence, Tuple

import pytest
from immutablecollections import immutabledict, immutableset
from more_itertools import flatten

from adam.curriculum import InstanceGroup, ExplicitWithoutSituationInstanceGroup
from adam.curriculum.curriculum_utils import TEST_CHOOSER_FACTORY
from adam.curriculum.phase1_curriculum import CHOOSER_FACTORY, phase1_instances
from adam.curriculum.pursuit_curriculum import make_simple_pursuit_curriculum
from adam.language import TokenSequenceLinguisticDescription
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
from adam.learner.integrated_learner import (
    SymbolicIntegratedTemplateLearner,
    SimulatedIntegratedTemplateLearner,
)
from adam.learner.language_mode import LanguageMode
from adam.learner.objects import (
    PursuitObjectLearner,
    SubsetObjectLearner,
    ProposeButVerifyObjectLearner,
    CrossSituationalObjectLearner,
)
from adam.ontology import OntologyNode, THING
from adam.ontology.phase1_ontology import (
    BALL,
    BLACK,
    BLUE,
    BOX,
    DOG,
    GAILA_PHASE_1_ONTOLOGY,
    GREEN,
    GROUND,
    HAND,
    HEAD,
    HOUSE,
    MOM,
    on,
    PHASE_1_CONCEPT,
    RED,
)
from adam.perception.perception_graph_nodes import (
    ContinuousNode,
    StrokeGNNRecognitionNode,
)
from adam.perception.visual_perception import (
    ClusterPerception,
    VisualPerceptionRepresentation,
    VisualPerceptionFrame,
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


def integrated_learner_factory(language_mode: LanguageMode):
    return SymbolicIntegratedTemplateLearner(
        object_learner=SubsetObjectLearner(
            ontology=GAILA_PHASE_1_ONTOLOGY,
            beam_size=10,
            language_mode=language_mode,
            min_continuous_feature_match_score=0.3,
        )
    )


def integrated_learner_sim_factory(language_mode: LanguageMode):
    return SimulatedIntegratedTemplateLearner(
        object_learner=SubsetObjectLearner(
            ontology=GAILA_PHASE_1_ONTOLOGY,
            beam_size=10,
            language_mode=language_mode,
            min_continuous_feature_match_score=0.1,
        )
    )


def integrated_learner_pv_factory(langage_mode: LanguageMode):
    rng = RandomChooser.for_seed(0)
    return SymbolicIntegratedTemplateLearner(
        object_learner=ProposeButVerifyObjectLearner(
            rng=rng,
            language_mode=langage_mode,
            ontology=GAILA_PHASE_1_ONTOLOGY,
            min_continuous_feature_match_score=0.3,
        )
    )


def integrated_learner_cs_factory(langage_mode: LanguageMode):
    return SymbolicIntegratedTemplateLearner(
        object_learner=CrossSituationalObjectLearner(
            smoothing_parameter=0.001,
            # The expected number of meanings is the number of subtypes of THING.
            # This should probably be set as a default value for the cross-situational object learner if possible...
            expected_number_of_meanings=len(
                GAILA_PHASE_1_ONTOLOGY.nodes_with_properties(THING)
            ),
            language_mode=langage_mode,
            ontology=GAILA_PHASE_1_ONTOLOGY,
            min_continuous_feature_match_score=0.3,
        )
    )


def run_subset_learner_for_object(
    nodes: Iterable[OntologyNode],
    *,
    learner,
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
):
    colored_obj_objects = [
        object_variable(
            "obj-with-color",
            node,
            added_properties=[
                color_variable("color", required_properties=[PHASE_1_CONCEPT])
            ],
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
                    chooser=CHOOSER_FACTORY(),
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
            chooser=TEST_CHOOSER_FACTORY(),
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
            assert gold in [
                desc.as_token_sequence()
                for desc in descriptions_from_learner.description_to_confidence
            ]


# tests learning "ball" in both languages
@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize(
    "learner",
    [
        integrated_learner_factory,
        integrated_learner_pv_factory,
        integrated_learner_cs_factory,
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

    object_learner = SubsetObjectLearner(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        beam_size=5,
        language_mode=LanguageMode.ENGLISH,
        min_continuous_feature_match_score=0.3,
    )

    for situation in [
        mom_situation,
        floating_head_situation,
        floating_hand_situation,
        floating_ball_situation,
        floating_house_situation,
    ]:
        perceptual_representation = (
            GAILA_PHASE_1_PERCEPTION_GENERATOR.generate_perception(
                situation, chooser=RandomChooser.for_seed(0)
            )
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

    mom_perceptual_representation = (
        GAILA_PHASE_1_PERCEPTION_GENERATOR.generate_perception(
            mom_situation, chooser=RandomChooser.for_seed(0)
        )
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
    # assert (ObjectSemanticNode, "head") in semantic_node_types_and_debug_strings
    # assert (ObjectSemanticNode, "hand") in semantic_node_types_and_debug_strings


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
                chooser=CHOOSER_FACTORY(),
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
    learner = SymbolicIntegratedTemplateLearner(
        object_learner=PursuitObjectLearner(
            learning_factor=0.05,
            graph_match_confirmation_threshold=0.7,
            lexicon_entry_threshold=0.7,
            rng=rng,
            smoothing_parameter=0.002,
            ontology=GAILA_PHASE_1_ONTOLOGY,
            language_mode=language_mode,
            min_continuous_feature_match_score=0.3,
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
                desc.as_token_sequence()
                for desc in descriptions_from_learner.description_to_confidence
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
                chooser=CHOOSER_FACTORY(),
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
    learner = SymbolicIntegratedTemplateLearner(
        object_learner=PursuitObjectLearner(
            learning_factor=0.05,
            graph_match_confirmation_threshold=0.7,
            lexicon_entry_threshold=0.7,
            rng=rng,
            smoothing_parameter=0.002,
            ontology=GAILA_PHASE_1_ONTOLOGY,
            language_mode=language_mode,
            min_continuous_feature_match_score=0.3,
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
                desc.as_token_sequence()
                for desc in descriptions_from_learner.description_to_confidence
            ]


def _make_fake_simulated_curriculum_with_continuous_features(
    cube_language: TokenSequenceLinguisticDescription,
) -> Tuple[
    Sequence[
        InstanceGroup[
            Any,
            TokenSequenceLinguisticDescription,
            VisualPerceptionFrame,
        ]
    ],
    Sequence[
        InstanceGroup[
            Any,
            TokenSequenceLinguisticDescription,
            VisualPerceptionFrame,
        ]
    ],
]:
    """
    Create and return a fake simulated curriculum in English, both train and test, with continuous
    features.

    This is based on samples from the real simulated curriciulum, but with continuous features
    added (for testing) and stroke information removed (for space).

    Arguments:
        cube_language:
            The linguistic description (in your langauge mode) for a scene with just a cube in it.
    """

    def one_cube_perception(
        *,
        viewpoint_id: int,
        color: OntologyNode,
        cromulence: float,
        centroid_x: float = 164.0,
        centroid_y: float = 138.6,
        std: float = 21.3,
    ) -> VisualPerceptionRepresentation[VisualPerceptionFrame]:
        properties = immutableset(
            [
                color,
                StrokeGNNRecognitionNode(
                    weight=1.0, object_recognized="cube", confidence=1.0
                ),
                ContinuousNode(
                    weight=1.0,
                    label="cromulence",
                    value=cromulence,
                ),
            ]
        )
        return VisualPerceptionRepresentation.single_frame(
            VisualPerceptionFrame(
                (
                    ClusterPerception(
                        cluster_id="0",
                        viewpoint_id=viewpoint_id,
                        sub_object_id=0,
                        strokes=immutableset(),
                        adjacent_strokes=immutabledict(),
                        properties=immutableset(properties),
                        centroid_x=centroid_x,
                        centroid_y=centroid_y,
                        std=std,
                    ),
                )
            )
        )

    train_percepta = [
        one_cube_perception(
            viewpoint_id=2,
            color=RED,
            cromulence=-1.0399841062404955,
            centroid_x=164.07928309927087,
            centroid_y=138.6924419756045,
            std=21.348674694574363,
        ),
        one_cube_perception(
            viewpoint_id=3,
            color=BLACK,
            cromulence=0.7504511958064572,
            centroid_x=104.08874672175489,
            centroid_y=100.66993528901608,
            std=21.548431197312045,
        ),
        one_cube_perception(
            viewpoint_id=4,
            color=BLACK,
            cromulence=0.9405647163912139,
            centroid_x=204.40925978830117,
            centroid_y=103.65682960241853,
            std=55.46661855606659,
        ),
        one_cube_perception(
            viewpoint_id=5,
            color=BLUE,
            cromulence=-1.9510351886538364,
            centroid_x=141.6972104005249,
            centroid_y=113.18199826684022,
            std=21.523024137061366,
        ),
        one_cube_perception(
            viewpoint_id=6,
            color=GREEN,
            cromulence=-1.302179506862318,
            centroid_x=170.38334628069387,
            centroid_y=127.90266211342467,
            std=25.465418718680052,
        ),
        one_cube_perception(
            viewpoint_id=7,
            color=GREEN,
            cromulence=0.12784040316728537,
            centroid_x=119.79480966188844,
            centroid_y=122.07781463475229,
            std=7.079007701857896,
        ),
        one_cube_perception(
            viewpoint_id=8,
            color=GREEN,
            cromulence=-0.3162425923435822,
            centroid_x=141.31220346026583,
            centroid_y=113.29999423125005,
            std=18.953288075496648,
        ),
        one_cube_perception(
            viewpoint_id=9,
            color=RED,
            cromulence=-0.016801157504288795,
            centroid_x=123.77352738856209,
            centroid_y=124.60147261143791,
            std=19.208465499613734,
        ),
        one_cube_perception(
            viewpoint_id=10,
            color=RED,
            cromulence=-0.85304392757358,
            centroid_x=110.72804171091398,
            centroid_y=123.13186108624421,
            std=14.572257903195126,
        ),
    ]

    test_percepta = [
        one_cube_perception(
            viewpoint_id=10,
            color=RED,
            cromulence=0.30471707975443135,
            centroid_x=110.72804171091398,
            centroid_y=123.13186108624421,
            std=14.572257903195126,
        ),
        one_cube_perception(
            viewpoint_id=11,
            color=RED,
            cromulence=-1.0399841062404955,
            centroid_x=120.66285987033817,
            centroid_y=131.2268791382163,
            std=16.18104373599222,
        ),
        one_cube_perception(
            viewpoint_id=12,
            color=BLUE,
            cromulence=0.7504511958064572,
            centroid_x=121.98681538377386,
            centroid_y=114.41089059914648,
            std=23.613332528818297,
        ),
        one_cube_perception(
            viewpoint_id=13,
            color=GREEN,
            cromulence=0.9405647163912139,
            centroid_x=140.52336809242834,
            centroid_y=136.72839927405838,
            std=20.640603170394897,
        ),
    ]

    return (
        [
            ExplicitWithoutSituationInstanceGroup(
                name="cubes_train",
                instances=tuple(
                    (cube_language, perception) for perception in train_percepta
                ),
            )
        ],
        [
            ExplicitWithoutSituationInstanceGroup(
                name="cubes_test",
                instances=tuple(
                    (cube_language, perception) for perception in test_percepta
                ),
            )
        ],
    )


# It doesn't make sense to parameterize this test since we only support English at the moment.
@pytest.mark.skipif(python_implementation() != "CPython", reason="requires SciPy")
def test_subset_learner_learns_continuous_feature():
    learner = integrated_learner_sim_factory(LanguageMode.ENGLISH)
    (
        train_curriculum,
        test_curriculum,
    ) = _make_fake_simulated_curriculum_with_continuous_features(
        TokenSequenceLinguisticDescription(("a", "cube"))
    )

    for training_stage in train_curriculum:
        for (
            _,
            linguistic_description,
            perceptual_representation,
        ) in training_stage.instances():
            learner.observe(
                LearningExample(perceptual_representation, linguistic_description)
            )

    for test_instance_group in test_curriculum:
        for (
            _,
            test_instance_language,
            test_instance_perception,
        ) in test_instance_group.instances():
            logging.info("lang: %s", test_instance_language)
            descriptions_from_learner = learner.describe(test_instance_perception)
            gold = test_instance_language.as_token_sequence()
            assert gold in [
                desc.as_token_sequence()
                for desc in descriptions_from_learner.description_to_confidence
            ]
