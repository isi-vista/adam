import logging
from itertools import repeat
from typing import Callable, Optional

from adam.language_specific.english import ENGLISH_DETERMINERS
from adam.learner.attributes import SubsetAttributeLearner
from adam.learner.verbs import SubsetVerbLearner
from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

from adam.curriculum.m6_curriculum import (
    M6_PREPOSITION_SUBCURRICULUM_GENERATORS,
    instantiate_subcurricula,
    make_m6_curriculum,
    M6_CURRICULUM_ALL_OBJECTS,
)
from adam.curriculum.phase1_curriculum import (
    _make_each_object_by_itself_curriculum,
    build_gaila_phase1_object_curriculum,
    build_gaila_phase1_attribute_curriculum,
    build_gaila_phase1_relation_curriculum,
    build_gaila_phase1_verb_curriculum,
    _make_put_on_speaker_addressee_body_part_curriculum,
)
from adam.curriculum.pursuit_curriculum import make_simple_pursuit_curriculum
from adam.experiment import Experiment, execute_experiment
from adam.experiment.observer import LearningProgressHtmlLogger, CandidateAccuracyObserver
from adam.learner import LanguageLearner
from adam.learner.object_recognizer import ObjectRecognizer
from adam.learner.prepositions import SubsetPrepositionLearner
from adam.learner.pursuit import HypothesisLogger
from adam.learner.objects import ObjectPursuitLearner, SubsetObjectLearner
from adam.ontology.phase1_ontology import (
    GAILA_PHASE_1_ONTOLOGY,
    PHASE_1_CURRICULUM_OBJECTS,
)
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    GAILA_M6_PERCEPTION_GENERATOR,
)
from adam.random_utils import RandomChooser


def log_experiment_entry_point(params: Parameters) -> None:
    experiment_group_dir = params.creatable_directory("experiment_group_dir")
    experiment_name = params.string("experiment")
    debug_log_dir = params.optional_creatable_directory("debug_log_directory")

    graph_logger: Optional[HypothesisLogger]
    if debug_log_dir:
        logging.info("Debug graphs will be written to %s", debug_log_dir)
        graph_logger = HypothesisLogger(debug_log_dir, enable_graph_rendering=True)
    else:
        graph_logger = None

    logger = LearningProgressHtmlLogger.create_logger(
        output_dir=experiment_group_dir, experiment_name=experiment_name
    )

    (training_instance_groups, test_instance_groups) = curriculum_from_params(params)

    execute_experiment(
        Experiment(
            name=experiment_name,
            training_stages=training_instance_groups,
            learner_factory=learner_factory_from_params(params, graph_logger),
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
    )


def learner_factory_from_params(
    params: Parameters, graph_logger: Optional[HypothesisLogger]
) -> Callable[[], LanguageLearner]:  # type: ignore
    learner_type = params.string(
        "learner",
        [
            "pursuit",
            "object-subset",
            "preposition-subset",
            "attribute-subset",
            "verb-subset",
        ],
    )

    # Eval hack! This is specific to the Phase 1 ontology
    object_recognizer = ObjectRecognizer.for_ontology_types(
        PHASE_1_CURRICULUM_OBJECTS,
        determiners=ENGLISH_DETERMINERS,
        ontology=GAILA_PHASE_1_ONTOLOGY,
    )

    if learner_type == "pursuit":
        return lambda: ObjectPursuitLearner.from_parameters(
            params.namespace("pursuit"), graph_logger=graph_logger
        )
    elif learner_type == "object-subset":
        return lambda: SubsetObjectLearner(ontology=GAILA_PHASE_1_ONTOLOGY)
    elif learner_type == "attribute-subset":
        return lambda: SubsetAttributeLearner(
            ontology=GAILA_PHASE_1_ONTOLOGY, object_recognizer=object_recognizer
        )
    elif learner_type == "preposition-subset":
        return lambda: SubsetPrepositionLearner(
            # graph_logger=graph_logger,
            object_recognizer=object_recognizer,
            ontology=GAILA_PHASE_1_ONTOLOGY,
        )
    elif learner_type == "verb-subset":
        return lambda: SubsetVerbLearner(
            ontology=GAILA_PHASE_1_ONTOLOGY, object_recognizer=object_recognizer
        )
    else:
        raise RuntimeError("can't happen")


def curriculum_from_params(params: Parameters):
    curriculum_name = params.string(
        "curriculum",
        [
            "m6-deniz",
            "each-object-by-itself",
            "pursuit",
            "m6-preposition",
            "m9-objects",
            "m9-attributes",
            "m9-relations",
            "m9-events",
            "m9-debug",
        ],
    )
    if curriculum_name == "m6-deniz":
        return (make_m6_curriculum(), [])
    elif curriculum_name == "each-object-by-itself":
        return (
            # We show the learned each item 6 times,
            # because pursuit won't lexicalize anything it hasn't seen five times.
            list(
                repeat(
                    _make_each_object_by_itself_curriculum(
                        perception_generator=GAILA_M6_PERCEPTION_GENERATOR
                    ),
                    10,
                )
            ),
            [
                _make_each_object_by_itself_curriculum(
                    perception_generator=GAILA_M6_PERCEPTION_GENERATOR
                )
            ],
        )
    elif curriculum_name == "pursuit":
        pursuit_curriculum_params = params.namespace("pursuit-curriculum-params")
        num_instances = pursuit_curriculum_params.integer("num_instances")
        num_noise_instances = pursuit_curriculum_params.integer("num_noise_instances")
        num_objects_in_instance = pursuit_curriculum_params.integer(
            "num_objects_in_instance"
        )
        return (
            [
                make_simple_pursuit_curriculum(
                    target_objects=M6_CURRICULUM_ALL_OBJECTS,
                    num_instances=num_instances,
                    num_objects_in_instance=num_objects_in_instance,
                    num_noise_instances=num_noise_instances,
                    perception_generator=GAILA_M6_PERCEPTION_GENERATOR,
                )
            ],
            [
                _make_each_object_by_itself_curriculum(
                    perception_generator=GAILA_M6_PERCEPTION_GENERATOR
                )
            ],
        )
    elif curriculum_name == "m6-preposition":
        return (instantiate_subcurricula(M6_PREPOSITION_SUBCURRICULUM_GENERATORS), [])
    elif curriculum_name == "m9-objects":
        return (build_gaila_phase1_object_curriculum(), [])
    elif curriculum_name == "m9-attributes":
        return (build_gaila_phase1_attribute_curriculum(), [])
    elif curriculum_name == "m9-relations":
        return (build_gaila_phase1_relation_curriculum(), [])
    elif curriculum_name == "m9-events":
        return (build_gaila_phase1_verb_curriculum(), [])
    elif curriculum_name == "m9-debug":
        return ([_make_put_on_speaker_addressee_body_part_curriculum()], [])
    else:
        raise RuntimeError("Can't happen")


if __name__ == "__main__":
    parameters_only_entry_point(log_experiment_entry_point)
