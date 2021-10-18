"""
Curricula for M6 Milestone
In this curriculum, we cover the following:
* Object vocabulary:
“mommy, daddy, baby, book, house, car, water, ball, juice, cup, box, chair, head, milk,
hand, dog, truck, door, hat, table, cookie, bird”
* Modifier vocabulary:
basic color terms (red, blue, green, white, black…), one, two, my, your)
"""
import random as r
from itertools import chain
from typing import Sequence, List, Optional

from more_itertools import flatten

from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.language.language_generator import LanguageGenerator
from adam.language.dependency import LinearizedDependencyTree

from adam.curriculum import ExplicitWithSituationInstanceGroup
from adam.curriculum.curriculum_utils import (
    CHOOSER_FACTORY,
    Phase1InstanceGroup,
    phase1_instances,
    standard_object,
    make_noise_objects,
)
from adam.curriculum.phase1_curriculum import (
    _make_each_object_by_itself_curriculum,
    _make_object_on_ground_curriculum,
    _make_objects_with_colors_curriculum,
)
from adam.curriculum.preposition_curriculum import (
    _behind_template,
    _beside_template,
    _on_template,
    _over_template,
    _under_template,
)
from adam.ontology import IS_ADDRESSEE, IS_SPEAKER
from adam.ontology.phase1_ontology import (
    BALL,
    BIRD,
    BOOK,
    BOX,
    CAR,
    CHAIR,
    COOKIE,
    CUP,
    GAILA_PHASE_1_ONTOLOGY,
    HAT,
    HOUSE,
    LEARNER,
    MOM,
    TABLE,
    TRUCK,
    DAD,
    BABY,
    WATER,
    HAND,
    DOG,
    MILK,
    HEAD,
    JUICE,
    DOOR,
)
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    GAILA_M6_PERCEPTION_GENERATOR,
)
from adam.situation.templates.phase1_templates import sampled

M6_PREPOSITION_CURRICULUM_SMALL_OBJECTS = [BALL, CUP, BOX, HAT, BOOK, COOKIE, BIRD]
M6_PREPOSITION_CURRICULUM_LARGER_OBJECTS = [TABLE, HOUSE, CAR, CHAIR, TRUCK]
M6_PREPOSITION_CURRICULUM_OBJECTS = list(
    chain(
        M6_PREPOSITION_CURRICULUM_SMALL_OBJECTS, M6_PREPOSITION_CURRICULUM_LARGER_OBJECTS
    )
)

M6_CURRICULUM_ALL_OBJECTS = [
    MOM,
    DAD,
    BABY,
    BOOK,
    HOUSE,
    CAR,
    WATER,
    BALL,
    JUICE,
    CUP,
    BOX,
    # TODO: https://github.com/isi-vista/adam/issues/946
    # CHAIR,
    HEAD,
    MILK,
    HAND,
    DOG,
    TRUCK,
    DOOR,
    HAT,
    TABLE,
    COOKIE,
    BIRD,
]

# Create object variables for objects to use in prepositions
SMALL_OBJECT_VARS = [
    standard_object("small_" + str(i), obj)
    for i, obj in enumerate(M6_PREPOSITION_CURRICULUM_SMALL_OBJECTS)
]
LARGE_OBJECT_VARS = [
    standard_object("large_" + str(i), obj)
    for i, obj in enumerate([TABLE, HOUSE, CAR, CHAIR, TRUCK])
]


def _make_m6_on_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    return phase1_instances(
        "Preposition on",
        situations=chain(
            *[
                sampled(
                    _on_template(
                        object_1,
                        object_2,
                        make_noise_objects(noise_objects),
                        is_training=True,
                    ),
                    chooser=CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    max_to_sample=num_samples if num_samples else 1,
                    block_multiple_of_the_same_type=True,
                )
                for object_1 in r.sample(SMALL_OBJECT_VARS, 3)
                for object_2 in r.sample(LARGE_OBJECT_VARS, 3)
            ]
        ),
        perception_generator=GAILA_M6_PERCEPTION_GENERATOR,
        language_generator=language_generator,
    )


def _make_m6_beside_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    return phase1_instances(
        "Preposition on",
        situations=chain(
            *[
                sampled(
                    _beside_template(
                        object_1,
                        object_2,
                        make_noise_objects(noise_objects),
                        is_training=True,
                        is_right=True,
                    ),
                    chooser=CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    max_to_sample=num_samples if num_samples else 1,
                    block_multiple_of_the_same_type=True,
                )
                for object_1 in r.sample(SMALL_OBJECT_VARS, 3)
                for object_2 in r.sample(LARGE_OBJECT_VARS, 3)
            ]
        ),
        perception_generator=GAILA_M6_PERCEPTION_GENERATOR,
        language_generator=language_generator,
    )


def _make_m6_under_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    return phase1_instances(
        "Preposition under",
        situations=chain(
            *[
                sampled(
                    _under_template(
                        object_1,
                        object_2,
                        make_noise_objects(noise_objects),
                        is_training=True,
                        is_distal=True,
                    ),
                    chooser=CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    max_to_sample=num_samples if num_samples else 1,
                    block_multiple_of_the_same_type=True,
                )
                for object_1 in r.sample(SMALL_OBJECT_VARS, 3)
                for object_2 in r.sample(LARGE_OBJECT_VARS, 3)
            ]
        ),
        perception_generator=GAILA_M6_PERCEPTION_GENERATOR,
        language_generator=language_generator,
    )


def _make_m6_over_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    return phase1_instances(
        "Preposition over",
        situations=chain(
            *[
                sampled(
                    _over_template(
                        object_1,
                        object_2,
                        make_noise_objects(noise_objects),
                        is_training=True,
                        is_distal=True,
                    ),
                    chooser=CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    max_to_sample=num_samples if num_samples else 1,
                    block_multiple_of_the_same_type=True,
                )
                for object_1 in r.sample(SMALL_OBJECT_VARS, 3)
                for object_2 in r.sample(LARGE_OBJECT_VARS, 3)
            ]
        ),
        perception_generator=GAILA_M6_PERCEPTION_GENERATOR,
        language_generator=language_generator,
    )


def _make_m6_behind_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    learner_object = standard_object("learner", LEARNER, added_properties=[IS_ADDRESSEE])
    mom = standard_object("mom", MOM, added_properties=[IS_SPEAKER])
    background = [learner_object, mom]
    background.extend(make_noise_objects(noise_objects))

    return phase1_instances(
        "Preposition behind",
        situations=chain(
            *[
                sampled(
                    _behind_template(
                        object_1, object_2, background, is_training=True, is_near=True
                    ),
                    chooser=CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    max_to_sample=num_samples if num_samples else 1,
                    block_multiple_of_the_same_type=True,
                )
                for object_1 in r.sample(SMALL_OBJECT_VARS, 3)
                for object_2 in r.sample(LARGE_OBJECT_VARS, 3)
            ]
        ),
        perception_generator=GAILA_M6_PERCEPTION_GENERATOR,
        language_generator=language_generator,
    )


def _make_m6_in_front_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    learner_object = standard_object("learner", LEARNER, added_properties=[IS_ADDRESSEE])
    mom = standard_object("mom", MOM, added_properties=[IS_SPEAKER])
    background = [learner_object, mom]
    background.extend(make_noise_objects(noise_objects))

    return phase1_instances(
        "Preposition behind",
        situations=chain(
            *[
                sampled(
                    _behind_template(
                        object_1, object_2, background, is_training=True, is_near=True
                    ),
                    chooser=CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    max_to_sample=num_samples if num_samples else 1,
                    block_multiple_of_the_same_type=True,
                )
                for object_1 in r.sample(SMALL_OBJECT_VARS, 3)
                for object_2 in r.sample(LARGE_OBJECT_VARS, 3)
            ]
        ),
        perception_generator=GAILA_M6_PERCEPTION_GENERATOR,
        language_generator=language_generator,
    )


M6_PREPOSITION_SUBCURRICULUM_GENERATORS = [
    _make_m6_on_curriculum,
    _make_m6_beside_curriculum,
    _make_m6_under_curriculum,
    _make_m6_over_curriculum,
    _make_m6_behind_curriculum,
    _make_m6_in_front_curriculum,
]


M6_SUBCURRICULUM_GENERATORS = list(
    chain(
        [
            [  # Single objects
                _make_each_object_by_itself_curriculum,
                # Objects with modifiers
                # Colors
                _make_objects_with_colors_curriculum,
                _make_object_on_ground_curriculum,
            ],
            M6_PREPOSITION_SUBCURRICULUM_GENERATORS,
        ]
    )
)


def _make_m6_mixed_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    r.seed(0)
    all_instances = flatten(  # type: ignore
        make_m6_curriculum(num_samples, noise_objects, language_generator)  # type: ignore
    )
    r.shuffle(all_instances)  # type: ignore
    return ExplicitWithSituationInstanceGroup("m6_mixed", tuple(all_instances))


def instantiate_subcurricula(
    subcurricula,
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> List[Phase1InstanceGroup]:
    return [
        subcurriculum(num_samples, num_noise_objects, language_generator)
        for subcurriculum in subcurricula
    ]


def make_m6_curriculum(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Sequence[Phase1InstanceGroup]:
    return instantiate_subcurricula(
        M6_SUBCURRICULUM_GENERATORS, num_samples, num_noise_objects, language_generator
    ) + [_make_m6_mixed_curriculum(num_samples, num_noise_objects, language_generator)]
