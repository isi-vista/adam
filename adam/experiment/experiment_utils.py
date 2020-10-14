import logging

from pathlib import Path

from itertools import repeat, chain
from more_itertools import only
from typing import Sequence, Optional, Iterable, Tuple, List
import random

from adam.curriculum import AblatedLanguageSituationsInstanceGroup
from adam.curriculum.curriculum_utils import (
    Phase1InstanceGroup,
    PHASE1_CHOOSER_FACTORY,
    phase1_instances,
    standard_object,
)
from adam.curriculum.m6_curriculum import (
    M6_PREPOSITION_SUBCURRICULUM_GENERATORS,
    instantiate_subcurricula,
    M6_CURRICULUM_ALL_OBJECTS,
)
from adam.curriculum.phase2_curriculum import make_multiple_object_situation
from adam.language_specific.english.english_language_generator import IGNORE_COLORS

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
)
from adam.curriculum.pursuit_curriculum import make_simple_pursuit_curriculum
from adam.language.dependency import LinearizedDependencyTree
from adam.language.language_generator import LanguageGenerator
from adam.random_utils import RandomChooser
from adam.situation import SituationObject
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.situation.templates.phase1_templates import sampled
from vistautils.parameters import Parameters
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator,
    GazePerceivedNoisily,
)


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
) -> Sequence[Phase1InstanceGroup]:
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


def build_pursuit_curriculum(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
    *,
    pursuit_curriculum_params: Parameters = Parameters.empty(),
) -> Sequence[Phase1InstanceGroup]:

    num_instances = pursuit_curriculum_params.integer(
        "num_instances", default=num_samples if num_samples else 10
    )
    num_noise_instances = pursuit_curriculum_params.integer(
        "num_noise_instances", default=num_noise_objects if num_noise_objects else 2
    )
    num_objects_in_instance = pursuit_curriculum_params.integer(
        "num_objects_in_instance", default=3
    )
    add_gaze = pursuit_curriculum_params.boolean("add_gaze", default=False)
    prob_given = pursuit_curriculum_params.floating_point("prob_given", default=1.0)
    prob_not_given = pursuit_curriculum_params.floating_point(
        "prob_not_given", default=0.0
    )
    rng = random.Random()
    rng.seed(0)
    gaze_perciever = GazePerceivedNoisily(
        rng=rng,
        prob_gaze_perceived_given_gaze=prob_given,
        prob_gaze_perceived_given_not_gaze=prob_not_given,
    )
    perception_generator = HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator(
        ontology=GAILA_PHASE_2_ONTOLOGY, gaze_strategy=gaze_perciever
    )
    return [
        make_simple_pursuit_curriculum(
            target_objects=M6_CURRICULUM_ALL_OBJECTS,
            num_instances=num_instances,
            num_objects_in_instance=num_objects_in_instance,
            num_noise_instances=num_noise_instances,
            language_generator=language_generator,
            add_gaze=add_gaze,
            perception_generator=perception_generator,
        )
    ]


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


def build_object_learner_experiment_curriculum_train(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
    *,
    params: Parameters = Parameters.empty(),
) -> Sequence[Phase1InstanceGroup]:
    situations = make_multiple_object_situation(
        num_samples, num_noise_objects, language_generator
    )
    accurate_language_chance = params.floating_point(
        "accurate_language_percentage", default=0.5
    )
    output_situations = []
    random.seed(params.integer("random_seed", default=0))
    rng = RandomChooser.for_seed(params.integer("language_random_seed", default=0))
    for (situation, language, perception) in situations.instances():
        if random.random() <= accurate_language_chance:
            output_language = language
        else:
            # Make Invalid Language
            if situation and isinstance(situation, HighLevelSemanticsSituation):
                # First, gather all OntologyNodes which aren't already present in the situation
                present_ontology_nodes = [
                    _object.ontology_node for _object in situation.all_objects
                ]
                valid_other_objects = [
                    node
                    for node in PHASE_1_CURRICULUM_OBJECTS
                    if node not in present_ontology_nodes
                ]
                # Then choose one at random
                chosen_ontology_node = rng.choice(valid_other_objects)
                # Make a fake situation with just this object in it, ignoring colors
                wrong_situation = HighLevelSemanticsSituation(
                    ontology=GAILA_PHASE_2_ONTOLOGY,
                    salient_objects=[
                        SituationObject.instantiate_ontology_node(
                            chosen_ontology_node, ontology=GAILA_PHASE_2_ONTOLOGY
                        )
                    ],
                    syntax_hints=[IGNORE_COLORS],
                )
                # Generate the language as if it came from this fake situation rather than the original one
                fake_language = only(
                    language_generator.generate_language(wrong_situation, chooser=rng)
                )
                output_language = LinearizedDependencyTree(
                    dependency_tree=fake_language.dependency_tree,
                    surface_token_order=fake_language.surface_token_order,
                    accurate=False,
                )

            else:
                raise RuntimeError(
                    f"Unable to make invalid language without a situation of type HighlevelSemanticsSituation. Got situation: {situation}"
                )

        output_situations.append((situation, output_language, perception))
    return [
        AblatedLanguageSituationsInstanceGroup(
            name=f"{situations.name()}_ablated", instances=output_situations
        )
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
    return []
