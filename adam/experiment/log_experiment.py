import logging
import pickle
import random
from pathlib import Path
from typing import Callable, Optional, Mapping, Tuple, cast, Any

from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

from adam.curriculum.curriculum_from_files import phase3_load_from_disk
from adam.curriculum.imprecise_descriptions_curriculum import (
    make_imprecise_size_curriculum,
    make_imprecise_temporal_descriptions,
    make_subtle_verb_distinctions_curriculum,
)
from adam.curriculum.m6_curriculum import make_m6_curriculum
from adam.curriculum.phase1_curriculum import (
    build_gaila_phase1_object_curriculum,
    build_gaila_phase1_attribute_curriculum,
    build_classifier_curriculum,
    build_gaila_phase1_relation_curriculum,
    build_gaila_phase1_verb_curriculum,
    build_gaila_phase_1_curriculum,
)
from adam.curriculum.phase2_curriculum import (
    build_functionally_defined_objects_curriculum,
    build_gaila_m13_curriculum,
    build_m13_shuffled_curriculum,
    integrated_pursuit_learner_experiment_curriculum,
    build_object_learner_experiment_curriculum_train,
    build_pursuit_curriculum,
    integrated_pursuit_learner_experiment_test,
)
from adam.curriculum.preposition_curriculum import make_prepositions_curriculum
from adam.curriculum.verbs_with_dynamic_prepositions_curriculum import (
    make_verb_with_dynamic_prepositions_curriculum,
)
from adam.experiment import Experiment, execute_experiment
from adam.experiment.curriculum_repository import (
    read_experiment_curriculum,
)
from adam.experiment.experiment_utils import (
    build_each_object_by_itself_curriculum_train,
    build_each_object_by_itself_curriculum_test,
    build_debug_curriculum_train,
    build_debug_curriculum_test,
    build_generics_curriculum,
    build_m6_prepositions_curriculum,
    build_functionally_defined_objects_train_curriculum,
    build_actions_and_generics_curriculum,
    observer_states_by_most_recent,
    build_object_learner_factory,
    build_attribute_learner_factory,
    build_relation_learner_factory,
    build_action_learner_factory,
    build_plural_learner_factory,
    build_affordance_learner_factory,
)
from adam.experiment.observer import LearningProgressHtmlLogger, YAMLLogger
from adam.language import LinguisticDescriptionT
from adam.language.language_utils import (
    phase2_language_generator,
    integrated_experiment_language_generator,
)
from adam.language_specific.english import ENGLISH_DETERMINERS
from adam.learner import TopLevelLanguageLearner
from adam.learner.affordances import MappingAffordanceLearner
from adam.learner.attributes import SubsetAttributeLearner, PursuitAttributeLearner
from adam.learner.functional_learner import FunctionalLearner
from adam.learner.generics import SimpleGenericsLearner
from adam.learner.integrated_learner import (
    SymbolicIntegratedTemplateLearner,
    SimulatedIntegratedTemplateLearner,
)
from adam.learner.language_mode import LanguageMode
from adam.learner.object_recognizer import ObjectRecognizer
from adam.learner.objects import PursuitObjectLearner, ProposeButVerifyObjectLearner
from adam.learner.objects import SubsetObjectLearner, ObjectRecognizerAsTemplateLearner
from adam.learner.pursuit import HypothesisLogger
from adam.learner.relations import SubsetRelationLearner
from adam.learner.template_learner import TemplateLearner
from adam.learner.verbs import SubsetVerbLearner
from adam.ontology.phase1_ontology import (
    GAILA_PHASE_1_ONTOLOGY,
    ME_HACK,
    YOU_HACK,
    PHASE_1_CURRICULUM_OBJECTS,
)
from adam.ontology.phase2_ontology import GAILA_PHASE_2_ONTOLOGY
from adam.perception import PerceptionT
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    GAILA_PHASE_1_PERCEPTION_GENERATOR,
)
from adam.perception.perception_graph import GraphLogger
from adam.random_utils import RandomChooser
from adam.situation import SituationT

SYMBOLIC = "symbolic"
SIMULATED = "simulated"


def log_experiment_entry_point(params: Parameters) -> None:
    debug_perception_log_dir = params.optional_creatable_directory(
        "debug_perception_log_dir"
    )
    perception_graph_logger: Optional[GraphLogger]
    if debug_perception_log_dir:
        logging.info(
            "Debug perception graphs will be written to %s", debug_perception_log_dir
        )
        perception_graph_logger = GraphLogger(
            debug_perception_log_dir, enable_graph_rendering=True
        )
    else:
        perception_graph_logger = None

    execute_experiment(
        experiment=experiment_from_params(params),
        log_path=params.optional_creatable_directory("hypothesis_log_dir"),
        log_hypotheses_every_n_examples=params.integer(
            "log_hypothesis_every_n_steps", default=250
        ),
        log_learner_state=params.boolean("log_learner_state", default=True),
        learner_logging_path=params.optional_creatable_directory("experiment_group_dir"),
        starting_point=params.integer("starting_point", default=0),
        point_to_log=params.integer("point_to_log", default=0),
        load_learner_state=params.optional_existing_file("learner_state_path"),
        resume_from_latest_logged_state=params.boolean(
            "resume_from_latest_logged_state", default=False
        ),
        debug_learner_pickling=params.boolean("debug_learner_pickling", default=False),
        perception_graph_logger=perception_graph_logger,
    )


def experiment_from_params(
    params: Parameters,
) -> Experiment[SituationT, LinguisticDescriptionT, PerceptionT]:
    experiment_name = params.string("experiment")
    debug_log_dir = params.optional_creatable_directory("debug_log_directory")

    graph_logger: Optional[HypothesisLogger]
    if debug_log_dir:
        logging.info("Debug graphs will be written to %s", debug_log_dir)
        graph_logger = HypothesisLogger(debug_log_dir, enable_graph_rendering=True)
    else:
        graph_logger = None

    language_mode = params.enum(
        "language_mode", LanguageMode, default=LanguageMode.ENGLISH
    )
    experiment_group_dir = params.optional_creatable_directory("experiment_group_dir")
    resume_from_last_logged_state = params.boolean(
        "resume_from_latest_logged_state", default=False
    )

    test_observer = []  # type: ignore
    pre_observer = []  # type: ignore
    post_observer = []  # type: ignore
    experiment_type = params.string(
        "experiment_type", valid_options=[SYMBOLIC, SIMULATED], default=SYMBOLIC
    )
    if experiment_type == SYMBOLIC:
        logging.info("Symbolic Experiment Section")
        logger = LearningProgressHtmlLogger.create_logger(params)

        curriculum_repository_path = params.optional_existing_directory(
            "load_from_curriculum_repository"
        )

        if curriculum_repository_path:
            curriculum = read_experiment_curriculum(
                curriculum_repository_path, params, language_mode
            )
            (training_instance_groups, test_instance_groups) = (
                curriculum.train_curriculum,
                curriculum.test_curriculum,
            )
        else:
            (training_instance_groups, test_instance_groups) = curriculum_from_params(
                params, language_mode
            )

        # Check if we have explicit observer states to load
        observers_state = params.optional_existing_file("observers_state_path")

        if resume_from_last_logged_state and observers_state:
            raise RuntimeError(
                "Can not resume from last logged state and provide explicit observer state paths"
            )

        if resume_from_last_logged_state:
            if not experiment_group_dir:
                raise RuntimeError(
                    "experiment_group_dir must be specified when resume_from_last_logged_state is true."
                )

            # Try to Load Observers
            for _, observers_state_path in observer_states_by_most_recent(
                cast(Path, experiment_group_dir) / "observer_state", "observers_state_at_"
            ):
                try:
                    with observers_state_path.open("rb") as f:
                        observers_holder = pickle.load(f)
                        pre_observer = observers_holder.pre_observers
                        post_observer = observers_holder.post_observers
                        test_observer = observers_holder.test_observers
                except OSError:
                    logging.warning(
                        "Unable to open observer state at %s; skipping.",
                        str(observers_state_path),
                    )
                except pickle.UnpicklingError:
                    logging.warning(
                        "Couldn't unpickle observer state at %s; skipping.",
                        str(observers_state_path),
                    )

            if not pre_observer and not post_observer and not test_observer:
                logging.warning("Reverting to default observers.")
                pre_observer = [
                    logger.pre_observer(  # type: ignore
                        params=params.namespace_or_empty("pre_observer"),
                        experiment_group_dir=experiment_group_dir,
                    )
                ]

                post_observer = [
                    logger.post_observer(  # type: ignore
                        params=params.namespace_or_empty("post_observer"),
                        experiment_group_dir=experiment_group_dir,
                    )
                ]

                test_observer = [
                    logger.test_observer(  # type: ignore
                        params=params.namespace_or_empty("post_observer"),
                        experiment_group_dir=experiment_group_dir,
                    )
                ]

        elif observers_state:
            try:
                with observers_state.open("rb") as f:
                    observers_holder = pickle.load(f)
                    pre_observer = observers_holder.pre_observers
                    post_observer = observers_holder.post_observers
                    test_observer = observers_holder.test_observers
            except OSError:
                logging.warning(
                    "Unable to open observer state at %s; skipping.", str(observers_state)
                )
            except pickle.UnpicklingError:
                logging.warning(
                    "Couldn't unpickle observer state at %s; skipping.",
                    str(observers_state),
                )
        else:
            pre_observer = [
                logger.pre_observer(  # type: ignore
                    params=params.namespace_or_empty("pre_observer"),
                    experiment_group_dir=experiment_group_dir,
                )
            ]

            post_observer = [
                logger.post_observer(  # type: ignore
                    params=params.namespace_or_empty("post_observer"),
                    experiment_group_dir=experiment_group_dir,
                )
            ]

            test_observer = [
                logger.test_observer(  # type: ignore
                    params=params.namespace_or_empty("test_observer"),
                    experiment_group_dir=experiment_group_dir,
                )
            ]
    else:
        logging.info("Simulated Experiment Section")
        (training_instance_groups, test_instance_groups) = curriculum_from_params(
            params, language_mode
        )
        yaml_observer_pre = YAMLLogger.from_params(  # type: ignore
            "pre_observer", params.namespace_or_empty("pre_observer")
        )
        if yaml_observer_pre:
            pre_observer.append(yaml_observer_pre)

        yaml_observer_post = YAMLLogger.from_params(  # type: ignore
            "post_observer", params.namespace_or_empty("post_observer")
        )
        if yaml_observer_post:
            post_observer.append(yaml_observer_post)

        yaml_observer_test = YAMLLogger.from_params(  # type: ignore
            "test_observer", params.namespace_or_empty("test_observer")
        )
        if yaml_observer_test:
            test_observer.append(yaml_observer_test)

    return Experiment(
        name=experiment_name,
        training_stages=training_instance_groups,
        learner_factory=learner_factory_from_params(params, graph_logger, language_mode),
        pre_example_training_observers=pre_observer,
        post_example_training_observers=post_observer,
        test_instance_groups=test_instance_groups,
        test_observers=test_observer,
        sequence_chooser=RandomChooser.for_seed(
            params.integer("sequence_chooser_seed", default=0)
        ),
    )


def learner_factory_from_params(
    params: Parameters,
    graph_logger: Optional[HypothesisLogger],  # pylint: disable=unused-argument
    language_mode: LanguageMode = LanguageMode.ENGLISH,
) -> Callable[[], TopLevelLanguageLearner]:  # type: ignore
    learner_type = params.string(
        "learner",
        [
            "integrated-learner",
            "integrated-learner-recognizer-without-generics",
            "integrated-learner-recognizer",
            "pursuit-gaze",
            "integrated-object-only",
            "integrated-learner-params",
            "integrated-pursuit-attribute-only",
            "simulated-integrated-learner-params",
        ],
    )

    beam_size = params.positive_integer("beam_size", default=10)
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

    if learner_type == "pursuit-gaze":
        return lambda: SymbolicIntegratedTemplateLearner(
            object_learner=PursuitObjectLearner(
                learning_factor=0.05,
                graph_match_confirmation_threshold=0.7,
                lexicon_entry_threshold=0.7,
                rng=rng,
                smoothing_parameter=0.002,
                ontology=GAILA_PHASE_2_ONTOLOGY,
                language_mode=language_mode,
                min_continuous_feature_match_score=0.7,
                rank_gaze_higher=True,
            ),
            attribute_learner=SubsetAttributeLearner(
                ontology=GAILA_PHASE_2_ONTOLOGY,
                beam_size=beam_size,
                language_mode=language_mode,
                min_continuous_feature_match_score=0.7,
            ),
            relation_learner=SubsetRelationLearner(
                ontology=GAILA_PHASE_2_ONTOLOGY,
                beam_size=beam_size,
                language_mode=language_mode,
                min_continuous_feature_match_score=0.7,
            ),
            action_learner=SubsetVerbLearner(
                ontology=GAILA_PHASE_2_ONTOLOGY,
                beam_size=beam_size,
                language_mode=language_mode,
                min_continuous_feature_match_score=0.7,
            ),
        )
    elif learner_type == "integrated-learner":
        return lambda: SymbolicIntegratedTemplateLearner(
            object_learner=SubsetObjectLearner(
                ontology=GAILA_PHASE_2_ONTOLOGY,
                beam_size=beam_size,
                language_mode=language_mode,
                min_continuous_feature_match_score=0.7,
            ),
            attribute_learner=SubsetAttributeLearner(
                ontology=GAILA_PHASE_2_ONTOLOGY,
                beam_size=beam_size,
                language_mode=language_mode,
                min_continuous_feature_match_score=0.7,
            ),
            relation_learner=SubsetRelationLearner(
                ontology=GAILA_PHASE_2_ONTOLOGY,
                beam_size=beam_size,
                language_mode=language_mode,
                min_continuous_feature_match_score=0.7,
            ),
            action_learner=SubsetVerbLearner(
                ontology=GAILA_PHASE_2_ONTOLOGY,
                beam_size=beam_size,
                language_mode=language_mode,
                min_continuous_feature_match_score=0.7,
            ),
            functional_learner=FunctionalLearner(language_mode=language_mode),
        )
    elif learner_type == "integrated-learner-recognizer":
        return lambda: SymbolicIntegratedTemplateLearner(
            object_learner=ObjectRecognizerAsTemplateLearner(
                object_recognizer=object_recognizer, language_mode=language_mode
            ),
            attribute_learner=SubsetAttributeLearner(
                ontology=GAILA_PHASE_2_ONTOLOGY,
                beam_size=beam_size,
                language_mode=language_mode,
                min_continuous_feature_match_score=0.7,
            ),
            relation_learner=SubsetRelationLearner(
                ontology=GAILA_PHASE_2_ONTOLOGY,
                beam_size=beam_size,
                language_mode=language_mode,
                min_continuous_feature_match_score=0.7,
            ),
            action_learner=SubsetVerbLearner(
                ontology=GAILA_PHASE_2_ONTOLOGY,
                beam_size=beam_size,
                language_mode=language_mode,
                min_continuous_feature_match_score=0.7,
            ),
            functional_learner=FunctionalLearner(language_mode=language_mode),
            generics_learner=SimpleGenericsLearner(),
        )
    elif learner_type == "ic":
        return lambda: SymbolicIntegratedTemplateLearner(
            object_learner=ObjectRecognizerAsTemplateLearner(
                object_recognizer=object_recognizer, language_mode=language_mode
            ),
            attribute_learner=SubsetAttributeLearner(
                ontology=GAILA_PHASE_2_ONTOLOGY,
                beam_size=beam_size,
                language_mode=language_mode,
                min_continuous_feature_match_score=0.7,
            ),
            relation_learner=SubsetRelationLearner(
                ontology=GAILA_PHASE_2_ONTOLOGY,
                beam_size=beam_size,
                language_mode=language_mode,
                min_continuous_feature_match_score=0.7,
            ),
            action_learner=SubsetVerbLearner(
                ontology=GAILA_PHASE_2_ONTOLOGY,
                beam_size=beam_size,
                language_mode=language_mode,
                min_continuous_feature_match_score=0.7,
            ),
            functional_learner=FunctionalLearner(language_mode=language_mode),
        )
    elif learner_type == "integrated-object-only":
        object_learner_type = params.string(
            "object_learner_type",
            valid_options=["subset", "pbv", "pursuit"],
            default="subset",
        )

        if params.has_namespace("learner_params"):
            learner_params = params.namespace("learner_params")
        else:
            learner_params = params.empty(namespace_prefix="learner_params")

        object_learner_factory: Callable[[], TemplateLearner]
        if object_learner_type == "subset":

            def subset_factory() -> SubsetObjectLearner:
                return SubsetObjectLearner(  # type: ignore
                    ontology=GAILA_PHASE_2_ONTOLOGY,
                    beam_size=beam_size,
                    language_mode=language_mode,
                )

            object_learner_factory = subset_factory

        elif object_learner_type == "pbv":

            def pbv_factory() -> ProposeButVerifyObjectLearner:
                return ProposeButVerifyObjectLearner.from_params(  # type: ignore
                    learner_params
                )

            object_learner_factory = pbv_factory
        elif object_learner_type == "pursuit":

            def pursuit_factory() -> PursuitObjectLearner:
                return PursuitObjectLearner(  # type: ignore
                    learning_factor=learner_params.floating_point("learning_factor"),
                    graph_match_confirmation_threshold=learner_params.floating_point(
                        "graph_match_confirmation_threshold"
                    ),
                    lexicon_entry_threshold=learner_params.floating_point(
                        "lexicon_entry_threshold"
                    ),
                    rng=rng,
                    smoothing_parameter=learner_params.floating_point(
                        "smoothing_parameter"
                    ),
                    ontology=GAILA_PHASE_2_ONTOLOGY,
                    language_mode=language_mode,
                )

            object_learner_factory = pursuit_factory
        else:
            raise RuntimeError(f"Invalid Object Learner Type Selected: {learner_type}")
        return lambda: SymbolicIntegratedTemplateLearner(
            object_learner=object_learner_factory()
        )
    elif learner_type == "integrated-learner-params":
        object_learner = build_object_learner_factory(  # type:ignore
            params.namespace_or_empty("object_learner"),
            beam_size,
            language_mode,
            graph_logger,
        )
        attribute_learner = build_attribute_learner_factory(  # type:ignore
            params.namespace_or_empty("attribute_learner"),
            beam_size,
            language_mode,
            graph_logger,
        )
        relation_learner = build_relation_learner_factory(  # type:ignore
            params.namespace_or_empty("relation_learner"),
            beam_size,
            language_mode,
            graph_logger,
        )
        action_learner = build_action_learner_factory(  # type:ignore
            params.namespace_or_empty("action_learner"), beam_size, language_mode
        )
        plural_learner = build_plural_learner_factory(  # type:ignore
            params.namespace_or_empty("plural_learner"), beam_size, language_mode
        )
        return lambda: SymbolicIntegratedTemplateLearner(
            object_learner=object_learner,
            attribute_learner=attribute_learner,
            relation_learner=relation_learner,
            action_learner=action_learner,
            functional_learner=FunctionalLearner(language_mode=language_mode)
            if params.boolean("include_functional_learner", default=True)
            else None,
            generics_learner=SimpleGenericsLearner()
            if params.boolean("include_generics_learner", default=True)
            else None,
            plural_learner=plural_learner,
            suppress_error=params.boolean("suppress_error", default=True),
        )
    elif learner_type == "integrated-pursuit-attribute-only":
        return lambda: SymbolicIntegratedTemplateLearner(
            object_learner=ObjectRecognizerAsTemplateLearner(
                object_recognizer=object_recognizer, language_mode=language_mode
            ),
            attribute_learner=PursuitAttributeLearner(
                learning_factor=0.05,
                graph_match_confirmation_threshold=0.7,
                lexicon_entry_threshold=0.7,
                rng=rng,
                smoothing_parameter=0.002,
                rank_gaze_higher=False,
                ontology=GAILA_PHASE_1_ONTOLOGY,
                language_mode=language_mode,
                min_continuous_feature_match_score=0.7,
            ),
        )
    elif learner_type == "simulated-integrated-learner-params":
        object_learner = build_object_learner_factory(  # type:ignore
            params.namespace_or_empty("object_learner"),
            beam_size,
            language_mode,
            graph_logger,
        )
        attribute_learner = build_attribute_learner_factory(  # type:ignore
            params.namespace_or_empty("attribute_learner"),
            beam_size,
            language_mode,
            graph_logger,
        )
        relation_learner = build_relation_learner_factory(  # type:ignore
            params.namespace_or_empty("relation_learner"),
            beam_size,
            language_mode,
            graph_logger,
        )
        action_learner = build_action_learner_factory(  # type:ignore
            params.namespace_or_empty("action_learner"), beam_size, language_mode
        )
        plural_learner = build_plural_learner_factory(  # type:ignore
            params.namespace_or_empty("plural_learner"), beam_size, language_mode
        )
        affordance_learner = build_affordance_learner_factory(  # type: ignore
            params.namespace_or_empty("affordance_learner"), beam_size, language_mode
        )
        return lambda: SimulatedIntegratedTemplateLearner(
            object_learner=object_learner,
            attribute_learner=attribute_learner,
            relation_learner=relation_learner,
            action_learner=action_learner,
            functional_learner=FunctionalLearner(language_mode=language_mode)
            if params.boolean("include_functional_learner", default=True)
            else None,
            generics_learner=SimpleGenericsLearner()
            if params.boolean("include_generics_learner", default=True)
            else None,
            plural_learner=plural_learner,
            suppress_error=params.boolean("suppress_error", default=True),
            affordance_learner=affordance_learner,
            mapping_affordance_learner=MappingAffordanceLearner(
                language_mode=language_mode,
                min_continuous_feature_match_score=0.3,  # Unused for this learner
            )
            if params.boolean("include_mapping_affordance_learner", default=False)
            else None,
        )
    else:
        raise RuntimeError("can't happen")


def curriculum_from_params(
    params: Parameters, language_mode: LanguageMode = LanguageMode.ENGLISH
):
    # The typing here is marked as 'Any' because MyPy doesn't seem to be able to verify it well
    # It's a complicated callable that should probably be a Protocol but JL spent ~1 hour
    # trying to get typing working to minimal success so instead I've just refactored all curriculum to follow
    # Callable[[Optional[int], Optional[int], LanguageGenerator, Parameters]], Sequence[InstanceGroup]]
    # As best as I can. We just don't have accurate type checking here any more :(
    str_to_train_test_curriculum: Mapping[str, Tuple[Any, Optional[Any]]] = {
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
        "chinese-classifiers": (build_classifier_curriculum, None),
        "m9-relations": (build_gaila_phase1_relation_curriculum, None),
        "m9-events": (build_gaila_phase1_verb_curriculum, None),
        "m9-debug": (build_debug_curriculum_train, build_debug_curriculum_test),
        "m9-complete": (build_gaila_phase_1_curriculum, None),
        "m13-imprecise-size": (make_imprecise_size_curriculum, None),
        "m13-imprecise-temporal": (make_imprecise_temporal_descriptions, None),
        "m13-subtle-verb-distinction": (make_subtle_verb_distinctions_curriculum, None),
        "m13-object-restrictions": (build_functionally_defined_objects_curriculum, None),
        "m13-functionally-defined-objects": (
            build_functionally_defined_objects_train_curriculum,
            build_functionally_defined_objects_curriculum,
        ),
        "m13-generics": (build_generics_curriculum, None),
        "m13-complete": (build_gaila_m13_curriculum, None),
        "m13-verbs-with-dynamic-prepositions": (
            make_verb_with_dynamic_prepositions_curriculum,
            None,
        ),
        "m13-shuffled": (build_m13_shuffled_curriculum, build_gaila_m13_curriculum),
        "m13-relations": (make_prepositions_curriculum, None),
        "actions-and-generics-curriculum": (build_actions_and_generics_curriculum, None),
        "m15-object-noise-experiments": (
            build_object_learner_experiment_curriculum_train,
            build_each_object_by_itself_curriculum_test,
        ),
        "m18-integrated-learners-experiment": (
            integrated_pursuit_learner_experiment_curriculum,
            integrated_pursuit_learner_experiment_test,
        ),
        "phase3": (
            phase3_load_from_disk,
            phase3_load_from_disk,
        ),
    }

    curriculum_name = params.string("curriculum", str_to_train_test_curriculum.keys())
    language_generator = (
        integrated_experiment_language_generator(language_mode)
        if curriculum_name == "m18-integrated-learners-experiment"
        else phase2_language_generator(language_mode)
    )

    (training_instance_groups, test_instance_groups) = str_to_train_test_curriculum[
        curriculum_name
    ]

    num_samples = params.optional_positive_integer("num_samples")
    # We need to be able to accept 0 as the number of noise objects but optional_integer doesn't currently
    # support specifying a range of acceptable values: https://github.com/isi-vista/vistautils/issues/142
    num_noise_objects = params.optional_integer("num_noise_objects")

    # optional argument to use path instead of goal
    if curriculum_name in [
        "m13-complete",
        "m13-shuffled",
        "m13-verbs-with-dynamic-prepositions",
    ]:
        return (
            training_instance_groups(
                num_samples,
                num_noise_objects,
                language_generator,
                params.boolean("use-path-instead-of-goal", default=False),
            ),
            test_instance_groups(num_samples, num_noise_objects, language_generator)
            if test_instance_groups
            else [],
        )

    return (
        training_instance_groups(
            num_samples,
            num_noise_objects,
            language_generator,
            params=params.namespace_or_empty(
                "pursuit-curriculum-params"
                if curriculum_name == "pursuit"
                else "train_curriculum"
            ),
        ),
        test_instance_groups(
            num_samples,
            num_noise_objects,
            language_generator,
            params=params.namespace_or_empty("test_curriculum"),
        )
        if test_instance_groups
        else [],
    )


if __name__ == "__main__":
    parameters_only_entry_point(log_experiment_entry_point)
