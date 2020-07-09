from itertools import repeat
from typing import Sequence

from adam.curriculum.curriculum_utils import Phase1InstanceGroup
from adam.curriculum.m6_curriculum import (
    M6_PREPOSITION_SUBCURRICULUM_GENERATORS,
    instantiate_subcurricula,
    M6_CURRICULUM_ALL_OBJECTS,
)
from adam.curriculum.phase1_curriculum import (
    _make_each_object_by_itself_curriculum,
    _make_put_on_speaker_addressee_body_part_curriculum,
    _make_generic_statements_curriculum,
)
from adam.curriculum.pursuit_curriculum import make_simple_pursuit_curriculum
from adam.language.dependency import LinearizedDependencyTree
from adam.language.language_generator import LanguageGenerator
from adam.language_specific.english.english_language_generator import (
    GAILA_PHASE_1_LANGUAGE_GENERATOR,
)
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    GAILA_M6_PERCEPTION_GENERATOR,
)
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from vistautils.parameters import Parameters


def build_each_object_by_itself_curriculum_train(
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ] = GAILA_PHASE_1_LANGUAGE_GENERATOR
) -> Sequence[Phase1InstanceGroup]:
    # We show the learned each item 6 times,
    # because pursuit won't lexicalize anything it hasn't seen five times.
    return list(
        repeat(
            _make_each_object_by_itself_curriculum(
                perception_generator=GAILA_M6_PERCEPTION_GENERATOR,
                language_generator=language_generator,
            ),
            10,
        )
    )


def build_each_object_by_itself_curriculum_test(
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ] = GAILA_PHASE_1_LANGUAGE_GENERATOR
) -> Sequence[Phase1InstanceGroup]:
    return [
        _make_each_object_by_itself_curriculum(
            perception_generator=GAILA_M6_PERCEPTION_GENERATOR,
            language_generator=language_generator,
        )
    ]


def build_generics_curriculum(
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ] = GAILA_PHASE_1_LANGUAGE_GENERATOR
) -> Sequence[Phase1InstanceGroup]:
    return [_make_generic_statements_curriculum(language_generator=language_generator)]


def build_m6_prepositions_curriculum(
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ] = GAILA_PHASE_1_LANGUAGE_GENERATOR
) -> Sequence[Phase1InstanceGroup]:
    return instantiate_subcurricula(
        M6_PREPOSITION_SUBCURRICULUM_GENERATORS, language_generator=language_generator
    )


def build_pursuit_curriculum(
    *,
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ] = GAILA_PHASE_1_LANGUAGE_GENERATOR,
    pursuit_curriculum_params: Parameters = Parameters.empty(),
) -> Sequence[Phase1InstanceGroup]:

    num_instances = pursuit_curriculum_params.integer("num_instances")
    num_noise_instances = pursuit_curriculum_params.integer("num_noise_instances")
    num_objects_in_instance = pursuit_curriculum_params.integer("num_objects_in_instance")

    return [
        make_simple_pursuit_curriculum(
            target_objects=M6_CURRICULUM_ALL_OBJECTS,
            num_instances=num_instances,
            num_objects_in_instance=num_objects_in_instance,
            num_noise_instances=num_noise_instances,
            perception_generator=GAILA_M6_PERCEPTION_GENERATOR,
            language_generator=language_generator,
        )
    ]


def build_debug_curriculum_train(
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ] = GAILA_PHASE_1_LANGUAGE_GENERATOR
) -> Sequence[Phase1InstanceGroup]:
    return [
        _make_put_on_speaker_addressee_body_part_curriculum(
            language_generator=language_generator
        )
    ]


def build_debug_curriculum_test(  # pylint: disable=unused-argument
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ] = GAILA_PHASE_1_LANGUAGE_GENERATOR
) -> Sequence[Phase1InstanceGroup]:
    return []
