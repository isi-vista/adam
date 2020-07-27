from itertools import repeat
from typing import Optional, Sequence

from adam.curriculum.curriculum_utils import Phase1InstanceGroup
from adam.curriculum.m6_curriculum import (
    M6_CURRICULUM_ALL_OBJECTS,
    M6_PREPOSITION_SUBCURRICULUM_GENERATORS,
    instantiate_subcurricula,
)
from adam.curriculum.phase1_curriculum import (
    _make_each_object_by_itself_curriculum,
    _make_generic_statements_curriculum,
    _make_put_on_speaker_addressee_body_part_curriculum,
)
from adam.curriculum.pursuit_curriculum import make_simple_pursuit_curriculum
from adam.language.dependency import LinearizedDependencyTree
from adam.language.language_generator import LanguageGenerator
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    GAILA_M6_PERCEPTION_GENERATOR,
)
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from vistautils.parameters import Parameters


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
