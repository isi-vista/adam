import logging
from typing import Callable, Optional, Mapping, Iterable, Tuple

from adam.curriculum.curriculum_utils import Phase1InstanceGroup
from adam.curriculum.imprecise_descriptions_curriculum import (
    make_imprecise_size_curriculum,
    make_imprecise_temporal_descriptions,
    make_subtle_verb_distinctions_curriculum,
)
import random
from adam.learner.objects import PursuitObjectLearnerNew
from adam.curriculum.phase2_curriculum import (
    build_functionally_defined_objects_curriculum,
    build_gaila_m13_curriculum,
    build_m13_shuffled_curriculum,
)
from adam.curriculum.preposition_curriculum import make_prepositions_curriculum
from adam.curriculum.verbs_with_dynamic_prepositions_curriculum import (
    make_verb_with_dynamic_prepositions_curriculum,
)
from adam.experiment.experiment_utils import (
    build_each_object_by_itself_curriculum_train,
    build_each_object_by_itself_curriculum_test,
    build_debug_curriculum_train,
    build_debug_curriculum_test,
    build_generics_curriculum,
    build_m6_prepositions_curriculum,
    build_pursuit_curriculum,
)
from adam.language.dependency import LinearizedDependencyTree
from adam.language.language_generator import LanguageGenerator
from adam.language.language_utils import phase2_language_generator
from adam.language_specific.english import ENGLISH_DETERMINERS
from adam.learner.attributes import SubsetAttributeLearner, SubsetAttributeLearnerNew
from adam.learner.integrated_learner import IntegratedTemplateLearner
from adam.learner.language_mode import LanguageMode
from adam.learner.relations import SubsetRelationLearnerNew
from adam.learner.verbs import SubsetVerbLearner, SubsetVerbLearnerNew
from adam.ontology.phase2_ontology import GAILA_PHASE_2_ONTOLOGY
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    GAILA_PHASE_1_PERCEPTION_GENERATOR,
)
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

from adam.curriculum.m6_curriculum import make_m6_curriculum
from adam.curriculum.phase1_curriculum import (
    build_gaila_phase1_object_curriculum,
    build_gaila_phase1_attribute_curriculum,
    build_gaila_phase1_relation_curriculum,
    build_gaila_phase1_verb_curriculum,
    build_gaila_phase_1_curriculum,
)
from adam.experiment import Experiment, execute_experiment
from adam.experiment.observer import LearningProgressHtmlLogger, CandidateAccuracyObserver
from adam.learner import TopLevelLanguageLearner
from adam.learner.object_recognizer import ObjectRecognizer
from adam.learner.prepositions import SubsetPrepositionLearner
from adam.learner.pursuit import HypothesisLogger
from adam.learner.objects import (
    ObjectPursuitLearner,
    SubsetObjectLearner,
    SubsetObjectLearnerNew,
    ObjectRecognizerAsTemplateLearner,
)
from adam.ontology.phase1_ontology import (
    GAILA_PHASE_1_ONTOLOGY,
    ME_HACK,
    YOU_HACK,
    PHASE_1_CURRICULUM_OBJECTS,
)
from adam.random_utils import RandomChooser

LANGUAGE_GEN = LanguageGenerator[  # pylint: disable=invalid-name
    HighLevelSemanticsSituation, LinearizedDependencyTree
]

CURRICULUM_BUILDER = Callable[  # pylint: disable=invalid-name
    [Optional[int], Optional[int], LANGUAGE_GEN], Iterable[Phase1InstanceGroup]
]


def log_experiment_entry_point(params: Parameters) -> None:
    experiment_name = params.string("experiment")
    debug_log_dir = params.optional_creatable_directory("debug_log_directory")

    graph_logger: Optional[HypothesisLogger]
    if debug_log_dir:
        logging.info("Debug graphs will be written to %s", debug_log_dir)
        graph_logger = HypothesisLogger(debug_log_dir, enable_graph_rendering=True)
    else:
        graph_logger = None

    logger = LearningProgressHtmlLogger.create_logger(params)

    language_mode = params.enum(
        "language_mode", LanguageMode, default=LanguageMode.ENGLISH
    )

    (training_instance_groups, test_instance_groups) = curriculum_from_params(
        params, language_mode
    )

    execute_experiment(
        Experiment(
            name=experiment_name,
            training_stages=training_instance_groups,
            learner_factory=learner_factory_from_params(
                params, graph_logger, language_mode
            ),
            pre_example_training_observers=[
                logger.pre_observer(),
                CandidateAccuracyObserver("pre-acc-observer"),
            ],
            post_example_training_observers=[logger.post_observer()],
            test_instance_groups=test_instance_groups,
            test_observers=[logger.test_observer()],
            sequence_chooser=RandomChooser.for_seed(0),
        ),
        log_path=params.optional_creatable_directory("hypothesis_log_dir"),
        log_hypotheses_every_n_examples=params.integer(
            "log_hypothesis_every_n_steps", default=250
        ),
        log_learner_state=params.boolean("log_learner_state", default=True),
        learner_logging_path=params.optional_creatable_directory("experiment_group_dir"),
        starting_point=params.integer("starting_point", default=-1),
        point_to_log=params.integer("point_to_log", default=0),
        load_learner_state=params.optional_existing_file("learner_state_path"),
    )


def learner_factory_from_params(
    params: Parameters,
    graph_logger: Optional[HypothesisLogger],
    language_mode: LanguageMode = LanguageMode.ENGLISH,
) -> Callable[[], TopLevelLanguageLearner]:  # type: ignore
    learner_type = params.string(
        "learner",
        [
            "pursuit",
            "object-subset",
            "preposition-subset",
            "attribute-subset",
            "verb-subset",
            "integrated-learner",
            "integrated-learner-recognizer",
            "pursuit-gaze",
        ],
    )

    beam_size = params.positive_integer("beam_size", default=10)

    if language_mode == LanguageMode.CHINESE and learner_type not in [
        "integrated-learner",
        "integrated-learner-recognizer",
    ]:
        raise RuntimeError("Only able to test Chinese with integrated learner.")

    rng = random.Random()
    rng.seed(0)
    perception_generator = GAILA_PHASE_1_PERCEPTION_GENERATOR

    objects = [YOU_HACK, ME_HACK]
    objects.extend(PHASE_1_CURRICULUM_OBJECTS)

    # Eval hack! This is specific to the Phase 1 ontology
    object_recognizer = ObjectRecognizer.for_ontology_types(
        objects,
        determiners=ENGLISH_DETERMINERS,
        ontology=GAILA_PHASE_1_ONTOLOGY,
        language_mode=language_mode,
        perception_generator=perception_generator,
    )

    if learner_type == "pursuit":
        return lambda: ObjectPursuitLearner.from_parameters(
            params.namespace("pursuit"), graph_logger=graph_logger
        )
    elif learner_type == "pursuit-gaze":
        return lambda: IntegratedTemplateLearner(
            object_learner=PursuitObjectLearnerNew(
                learning_factor=0.05,
                graph_match_confirmation_threshold=0.7,
                lexicon_entry_threshold=0.7,
                rng=rng,
                smoothing_parameter=0.002,
                ontology=GAILA_PHASE_2_ONTOLOGY,
                language_mode=language_mode,
                rank_gaze_higher=True,
            )
        )
    elif learner_type == "object-subset":
        return lambda: SubsetObjectLearner(
            ontology=GAILA_PHASE_1_ONTOLOGY, language_mode=LanguageMode.ENGLISH
        )
    elif learner_type == "attribute-subset":
        return lambda: SubsetAttributeLearner(
            ontology=GAILA_PHASE_1_ONTOLOGY,
            object_recognizer=object_recognizer,
            language_mode=LanguageMode.ENGLISH,
        )
    elif learner_type == "preposition-subset":
        return lambda: SubsetPrepositionLearner(
            # graph_logger=graph_logger,
            object_recognizer=object_recognizer,
            ontology=GAILA_PHASE_1_ONTOLOGY,
            language_mode=LanguageMode.ENGLISH,
        )
    elif learner_type == "verb-subset":
        return lambda: SubsetVerbLearner(
            ontology=GAILA_PHASE_1_ONTOLOGY,
            object_recognizer=object_recognizer,
            language_mode=LanguageMode.ENGLISH,
        )
    elif learner_type == "integrated-learner":
        return lambda: IntegratedTemplateLearner(
            object_learner=SubsetObjectLearnerNew(
                ontology=GAILA_PHASE_2_ONTOLOGY,
                beam_size=beam_size,
                language_mode=language_mode,
            ),
            attribute_learner=SubsetAttributeLearnerNew(
                ontology=GAILA_PHASE_2_ONTOLOGY,
                beam_size=beam_size,
                language_mode=language_mode,
            ),
            relation_learner=SubsetRelationLearnerNew(
                ontology=GAILA_PHASE_2_ONTOLOGY,
                beam_size=beam_size,
                language_mode=language_mode,
            ),
            action_learner=SubsetVerbLearnerNew(
                ontology=GAILA_PHASE_2_ONTOLOGY,
                beam_size=beam_size,
                language_mode=language_mode,
            ),
        )
    elif learner_type == "integrated-learner-recognizer":
        return lambda: IntegratedTemplateLearner(
            object_learner=ObjectRecognizerAsTemplateLearner(
                object_recognizer=object_recognizer, language_mode=language_mode
            ),
            attribute_learner=SubsetAttributeLearnerNew(
                ontology=GAILA_PHASE_2_ONTOLOGY,
                beam_size=beam_size,
                language_mode=language_mode,
            ),
            relation_learner=SubsetRelationLearnerNew(
                ontology=GAILA_PHASE_2_ONTOLOGY,
                beam_size=beam_size,
                language_mode=language_mode,
            ),
            action_learner=SubsetVerbLearnerNew(
                ontology=GAILA_PHASE_2_ONTOLOGY,
                beam_size=beam_size,
                language_mode=language_mode,
            ),
        )
    else:
        raise RuntimeError("can't happen")


def curriculum_from_params(
    params: Parameters, language_mode: LanguageMode = LanguageMode.ENGLISH
):
    str_to_train_test_curriculum: Mapping[
        str, Tuple[CURRICULUM_BUILDER, Optional[CURRICULUM_BUILDER]]
    ] = {
        "m6-deniz": (make_m6_curriculum, None),
        "each-object-by-itself": (
            build_each_object_by_itself_curriculum_train,
            build_each_object_by_itself_curriculum_test,
        ),
        "pursuit": (
            build_pursuit_curriculum,
            build_each_object_by_itself_curriculum_test,
        ),
        "m6-preposition": (build_m6_prepositions_curriculum, None),
        "m9-objects": (build_gaila_phase1_object_curriculum, None),
        "m9-attributes": (build_gaila_phase1_attribute_curriculum, None),
        "m9-relations": (build_gaila_phase1_relation_curriculum, None),
        "m9-events": (build_gaila_phase1_verb_curriculum, None),
        "m9-debug": (build_debug_curriculum_train, build_debug_curriculum_test),
        "m9-complete": (build_gaila_phase_1_curriculum, None),
        "m13-imprecise-size": (make_imprecise_size_curriculum, None),
        "m13-imprecise-temporal": (make_imprecise_temporal_descriptions, None),
        "m13-subtle-verb-distinction": (make_subtle_verb_distinctions_curriculum, None),
        "m13-object-restrictions": (build_functionally_defined_objects_curriculum, None),
        "m13-functionally-defined-objects": (
            build_functionally_defined_objects_curriculum,
            None,
        ),
        "m13-generics": (build_generics_curriculum, None),
        "m13-complete": (build_gaila_m13_curriculum, None),
        "m13-verbs-with-dynamic-prepositions": (
            make_verb_with_dynamic_prepositions_curriculum,
            None,
        ),
        "m13-shuffled": (build_m13_shuffled_curriculum, build_gaila_m13_curriculum),
        "m13-relations": (make_prepositions_curriculum, None),
    }

    curriculum_name = params.string("curriculum", str_to_train_test_curriculum.keys())
    language_generator = phase2_language_generator(language_mode)

    if params.has_namespace("pursuit-curriculum-params"):
        pursuit_curriculum_params = params.namespace("pursuit-curriculum-params")
    else:
        pursuit_curriculum_params = Parameters.empty()

    (training_instance_groups, test_instance_groups) = str_to_train_test_curriculum[
        curriculum_name
    ]

    num_samples = params.optional_positive_integer("num_samples")
    num_noise_objects = params.optional_positive_integer("num_noise_objects")

    return (
        training_instance_groups(num_samples, num_noise_objects, language_generator)
        if curriculum_name != "pursuit"
        else training_instance_groups(
            num_samples,
            num_noise_objects,
            language_generator,
            pursuit_curriculum_params=pursuit_curriculum_params,
        ),
        test_instance_groups(num_samples, num_noise_objects, language_generator)
        if test_instance_groups
        else [],
    )


if __name__ == "__main__":
    parameters_only_entry_point(log_experiment_entry_point)
