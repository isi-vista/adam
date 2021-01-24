import logging
from adam.language_specific.english import ENGLISH_DETERMINERS
from adam.learner import LanguageMode
from adam.learner.attributes import SubsetAttributeLearnerNew, PursuitAttributeLearnerNew
from adam.learner.object_recognizer import ObjectRecognizer
from adam.learner.objects import (
    SubsetObjectLearnerNew,
    ProposeButVerifyObjectLearner,
    PursuitObjectLearnerNew,
    ObjectRecognizerAsTemplateLearner,
    CrossSituationalObjectLearner,
)
from adam.learner.plurals import SubsetPluralLearnerNew
from adam.learner.relations import SubsetRelationLearnerNew, PursuitRelationLearnerNew
from adam.learner.template_learner import TemplateLearner
from adam.learner.verbs import SubsetVerbLearnerNew
from adam.ontology.integrated_learner_experiement_ontology import (
    INTEGRATED_EXPERIMENT_ONTOLOGY,
    INTEGRATED_EXPERIMENT_CURRICULUM_OBJECTS,
)

from pathlib import Path

from itertools import repeat, chain
from typing import Sequence, Optional, Iterable, Tuple, List
import random

from adam.curriculum.curriculum_utils import (
    Phase1InstanceGroup,
    PHASE1_CHOOSER_FACTORY,
    phase1_instances,
    standard_object,
)
from adam.curriculum.m6_curriculum import (
    M6_PREPOSITION_SUBCURRICULUM_GENERATORS,
    instantiate_subcurricula,
)

from adam.ontology.phase2_ontology import GAILA_PHASE_2_ONTOLOGY
from adam.ontology import IS_SPEAKER, IS_ADDRESSEE, THING
from adam.ontology.phase1_ontology import (
    INANIMATE_OBJECT,
    CAN_BE_SAT_ON_BY_PEOPLE,
    ANIMATE,
    PHASE_1_CURRICULUM_OBJECTS,
)
from adam.ontology.phase1_ontology import GAILA_PHASE_1_ONTOLOGY

from adam.curriculum.phase1_curriculum import (
    _make_each_object_by_itself_curriculum,
    _make_put_on_speaker_addressee_body_part_curriculum,
    _make_generic_statements_curriculum,
    _make_drink_curriculum,
    make_sit_transitive,
    make_sit_template_intransitive,
    _make_fly_curriculum,
    _make_jump_curriculum,
    _make_sit_curriculum,
    _make_eat_curriculum,
)
from adam.language.dependency import LinearizedDependencyTree
from adam.language.language_generator import LanguageGenerator
from adam.random_utils import RandomChooser
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.situation.templates.phase1_templates import sampled
from vistautils.parameters import Parameters
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    GAILA_PHASE_1_PERCEPTION_GENERATOR,
    GAILA_PHASE_2_PERCEPTION_GENERATOR,
    INTEGRATED_EXPERIMENT_PERCEPTION_GENERATOR,
)
from immutablecollections import immutabledict


# Experiment Utils
def observer_states_by_most_recent(
    pickled_path: Path, pickel_name: str
) -> Iterable[Tuple[int, Path]]:
    paths = []
    for logged_state_path in pickled_path.glob(f"{pickel_name}*.pkl"):
        if not logged_state_path.is_file():
            logging.warning("Skipping non-file learner state %s.", str(logged_state_path))
            continue
        iteration_number_string = logged_state_path.name.replace(
            f"{pickel_name}", ""
        ).replace(".pkl", "")
        try:
            iteration_number = int(iteration_number_string)
        except ValueError:
            logging.warning(
                "Skipping Observer state file with bad iteration number %s.",
                iteration_number_string,
            )
            continue
        paths.append((iteration_number, logged_state_path))
        logged_state_path.stat()
    return sorted(paths, reverse=True)


def restore_report_state(path: Path, num_instances: int) -> None:
    """
    Restore the output report files to the appropriate number of instances to match the loaded observer
    """
    lines_to_write_back: List[str] = []
    with path.open("r") as file:
        for num, line in enumerate(file):
            if num < num_instances:
                lines_to_write_back.append(line.strip())

    # Remove the old file
    path.unlink()

    # Replace it with our new one
    path.write_text("\n".join(lines_to_write_back))


def restore_html_state(path: Path, num_instance: int) -> None:
    """
    Restore an experiment HTML output to a specific number of instances to match the experiment state
    """
    lines_to_write_back: List[str] = []
    count_instances: int = 0
    with path.open("r") as file:
        for line in file:
            lines_to_write_back.append(line)
            if "</table>" in line:
                count_instances = count_instances + 1

            if count_instances >= num_instance:
                break

    # Remove the old file
    path.unlink()

    # Replace it with our new one
    path.write_text("".join(lines_to_write_back))


ONTOLOGY_STR_TO_ONTOLOGY = immutabledict(  # type: ignore
    [
        (
            "phase1",
            (
                GAILA_PHASE_1_ONTOLOGY,
                PHASE_1_CURRICULUM_OBJECTS,
                GAILA_PHASE_1_PERCEPTION_GENERATOR,
            ),
        ),
        (
            "phase2",
            (
                GAILA_PHASE_2_ONTOLOGY,
                PHASE_1_CURRICULUM_OBJECTS,
                GAILA_PHASE_2_PERCEPTION_GENERATOR,
            ),
        ),
        (
            "integrated_experiment",
            (
                INTEGRATED_EXPERIMENT_ONTOLOGY,
                INTEGRATED_EXPERIMENT_CURRICULUM_OBJECTS,
                INTEGRATED_EXPERIMENT_PERCEPTION_GENERATOR,
            ),
        ),
    ]
)


def build_object_learner_factory(
    params: Parameters, beam_size: int, language_mode: LanguageMode
) -> TemplateLearner:
    learner_type = params.string(
        "learner_type",
        valid_options=["subset", "pbv", "cross-situational", "pursuit", "recognizer"],
        default="subset",
    )
    ontology, objects, perception_gen = ONTOLOGY_STR_TO_ONTOLOGY[
        params.string(
            "ontology", valid_options=ONTOLOGY_STR_TO_ONTOLOGY.keys(), default="phase2"
        )
    ]

    if learner_type == "subset":
        return SubsetObjectLearnerNew(
            ontology=ontology, beam_size=beam_size, language_mode=language_mode
        )
    elif learner_type == "pbv":
        chooser = RandomChooser.for_seed(
            params.optional_integer("random_seed", default=0)
        )
        return ProposeButVerifyObjectLearner(
            graph_match_confirmation_threshold=params.floating_point(
                "graph_match_confirmation_threshold", default=0.8
            ),
            rng=chooser,
            ontology=ontology,
            language_mode=language_mode,
        )
    elif learner_type == "cross-situational":
        return CrossSituationalObjectLearner(
            graph_match_confirmation_threshold=params.floating_point(
                "graph_match_confirmation_threshold"
            ),
            lexicon_entry_threshold=params.floating_point("lexicon_entry_threshold"),
            smoothing_parameter=params.floating_point("smoothing_parameter"),
            expected_number_of_meanings=len(ontology.nodes_with_properties(THING)),
            ontology=ontology,
            language_mode=language_mode,
        )
    elif learner_type == "pursuit":
        rng = random.Random()
        rng.seed(params.integer("random_seed", default=0))
        return PursuitObjectLearnerNew(
            learning_factor=params.floating_point("learning_factor"),
            graph_match_confirmation_threshold=params.floating_point(
                "graph_match_confirmation_threshold"
            ),
            lexicon_entry_threshold=params.floating_point("lexicon_entry_threshold"),
            rng=rng,
            smoothing_parameter=params.floating_point("smoothing_parameter"),
            ontology=ontology,
            language_mode=language_mode,
            rank_gaze_higher=params.boolean("rank_gaze_higher", default=False),
        )
    elif learner_type == "recognizer":
        object_recognizer = ObjectRecognizer.for_ontology_types(
            objects,
            determiners=ENGLISH_DETERMINERS,
            ontology=ontology,
            language_mode=language_mode,
            perception_generator=perception_gen,
        )
        return ObjectRecognizerAsTemplateLearner(
            object_recognizer=object_recognizer, language_mode=language_mode
        )
    else:
        raise RuntimeError("Object learner type invalid")


def build_attribute_learner_factory(
    params: Parameters, beam_size: int, language_mode: LanguageMode
) -> Optional[TemplateLearner]:
    learner_type = params.string(
        "learner_type", valid_options=["subset", "pursuit", "none"], default="subset"
    )
    ontology, _, _ = ONTOLOGY_STR_TO_ONTOLOGY[
        params.string(
            "ontology", valid_options=ONTOLOGY_STR_TO_ONTOLOGY.keys(), default="phase2"
        )
    ]

    if learner_type == "subset":
        return SubsetAttributeLearnerNew(
            ontology=ontology, beam_size=beam_size, language_mode=language_mode
        )
    elif learner_type == "pursuit":
        rng = random.Random()
        rng.seed(params.integer("random_seed", default=0))
        return PursuitAttributeLearnerNew(
            learning_factor=params.floating_point("learning_factor"),
            graph_match_confirmation_threshold=params.floating_point(
                "graph_match_confirmation_threshold"
            ),
            lexicon_entry_threshold=params.floating_point("lexicon_entry_threshold"),
            rng=rng,
            smoothing_parameter=params.floating_point("smoothing_parameter"),
            ontology=ontology,
            language_mode=language_mode,
        )
    elif learner_type == "none":
        # We don't want to include this learner type.
        return None
    else:
        raise RuntimeError("Attribute learner type invalid.")


def build_relation_learner_factory(
    params: Parameters, beam_size: int, language_mode: LanguageMode
) -> Optional[TemplateLearner]:
    learner_type = params.string(
        "learner_type", valid_options=["subset", "pursuit", "none"], default="subset"
    )
    ontology, _, _ = ONTOLOGY_STR_TO_ONTOLOGY[
        params.string(
            "ontology", valid_options=ONTOLOGY_STR_TO_ONTOLOGY.keys(), default="phase2"
        )
    ]

    if learner_type == "subset":
        return SubsetRelationLearnerNew(
            ontology=ontology, beam_size=beam_size, language_mode=language_mode
        )
    elif learner_type == "pursuit":
        rng = random.Random()
        rng.seed(params.integer("random_seed", default=0))
        return PursuitRelationLearnerNew(
            learning_factor=params.floating_point("learning_factor"),
            graph_match_confirmation_threshold=params.floating_point(
                "graph_match_confirmation_threshold"
            ),
            lexicon_entry_threshold=params.floating_point("lexicon_entry_threshold"),
            rng=rng,
            smoothing_parameter=params.floating_point("smoothing_parameter"),
            ontology=ontology,
            language_mode=language_mode,
        )
    elif learner_type == "none":
        # We don't want to include this learner type.
        return None
    else:
        raise RuntimeError("Relation learner type invalid ")


def build_action_learner_factory(
    params: Parameters, beam_size: int, language_mode: LanguageMode
) -> Optional[TemplateLearner]:
    learner_type = params.string(
        "learner_type", valid_options=["subset", "none"], default="subset"
    )
    ontology, _, _ = ONTOLOGY_STR_TO_ONTOLOGY[
        params.string(
            "ontology", valid_options=ONTOLOGY_STR_TO_ONTOLOGY.keys(), default="phase2"
        )
    ]

    if learner_type == "subset":
        return SubsetVerbLearnerNew(
            ontology=ontology, beam_size=beam_size, language_mode=language_mode
        )
    elif learner_type == "none":
        # We don't want to include this learner type.
        return None
    else:
        raise RuntimeError("Action learner type invalid ")


def build_plural_learner_factory(
    params: Parameters, beam_size: int, language_mode: LanguageMode
) -> Optional[TemplateLearner]:
    learner_type = params.string(
        "learner_type", valid_options=["subset", "none"], default="subset"
    )
    ontology, _, _ = ONTOLOGY_STR_TO_ONTOLOGY[
        params.string(
            "ontology", valid_options=ONTOLOGY_STR_TO_ONTOLOGY.keys(), default="phase2"
        )
    ]

    if learner_type == "subset":
        return SubsetPluralLearnerNew(
            ontology=ontology, beam_size=beam_size, language_mode=language_mode
        )
    elif learner_type == "none":
        # We don't want to include this learner type.
        return None
    else:
        raise RuntimeError("Plural learner type invalid ")


# Curriculum Construction
def build_each_object_by_itself_curriculum_train(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Sequence[Phase1InstanceGroup]:
    # We show the learned each item 6 times,
    # because pursuit won't lexicalize anything it hasn't seen five times.
    return list(
        repeat(
            _make_each_object_by_itself_curriculum(
                num_samples, num_noise_objects, language_generator
            ),
            10,
        )
    )


def build_each_object_by_itself_curriculum_test(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
    *,
    params: Parameters = Parameters.empty(),
) -> Sequence[Phase1InstanceGroup]:
    # pylint: disable=unused-argument
    return [
        _make_each_object_by_itself_curriculum(
            num_samples, num_noise_objects, language_generator
        )
    ]


def build_generics_curriculum(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Sequence[Phase1InstanceGroup]:
    return [
        _make_generic_statements_curriculum(
            num_samples, num_noise_objects, language_generator
        )
    ]


def build_actions_and_generics_curriculum(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Sequence[Phase1InstanceGroup]:
    # pylint: disable=unused-argument
    return [
        _make_eat_curriculum(10, 0, language_generator),
        _make_drink_curriculum(10, 0, language_generator),
        _make_sit_curriculum(10, 0, language_generator),
        _make_jump_curriculum(10, 0, language_generator),
        _make_fly_curriculum(10, 0, language_generator),
        _make_generic_statements_curriculum(
            num_samples=20, noise_objects=0, language_generator=language_generator
        ),
    ]


def build_m6_prepositions_curriculum(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Sequence[Phase1InstanceGroup]:
    return instantiate_subcurricula(
        M6_PREPOSITION_SUBCURRICULUM_GENERATORS,
        num_samples,
        num_noise_objects,
        language_generator,
    )


def _make_sit_on_curriculum(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    sitter = standard_object(
        "sitter_0",
        THING,
        required_properties=[ANIMATE],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )
    seat = standard_object(
        "sitting-surface", INANIMATE_OBJECT, required_properties=[CAN_BE_SAT_ON_BY_PEOPLE]
    )
    return phase1_instances(
        "sit_on",
        chain(
            *[
                sampled(
                    make_sit_template_intransitive(
                        sitter, seat, num_noise_objects, surface=False, syntax_hints=False
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples if num_samples else 25,
                    block_multiple_of_the_same_type=True,
                ),
                sampled(
                    make_sit_transitive(
                        sitter, seat, num_noise_objects, surface=False, syntax_hints=False
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples if num_samples else 25,
                    block_multiple_of_the_same_type=True,
                ),
            ]
        ),
        language_generator=language_generator,
    )


def build_functionally_defined_objects_train_curriculum(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Sequence[Phase1InstanceGroup]:
    return [
        _make_sit_on_curriculum(num_samples, num_noise_objects, language_generator),
        _make_drink_curriculum(num_samples, num_noise_objects, language_generator),
    ]


def build_debug_curriculum_train(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Sequence[Phase1InstanceGroup]:
    return [
        _make_put_on_speaker_addressee_body_part_curriculum(
            num_samples, num_noise_objects, language_generator
        )
    ]


def build_debug_curriculum_test(  # pylint: disable=unused-argument
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Sequence[Phase1InstanceGroup]:
    # pylint: disable=unused-argument
    return []
