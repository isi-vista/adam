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

from immutablecollections import immutableset

from adam.curriculum import ExplicitWithSituationInstanceGroup
from adam.curriculum.curriculum_utils import (
    PHASE1_CHOOSER,
    Phase1InstanceGroup,
    phase1_instances,
    standard_object,
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
from adam.situation.templates.phase1_templates import sampled

r.seed(0)

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
    CHAIR,
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


def _make_m6_on_curriculum() -> Phase1InstanceGroup:
    return phase1_instances(
        "Preposition on",
        situations=chain(
            *[
                sampled(
                    _on_template(object_1, object_2, immutableset(), is_training=True),
                    chooser=PHASE1_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    max_to_sample=1,
                )
                for object_1 in r.sample(SMALL_OBJECT_VARS, 3)
                for object_2 in r.sample(LARGE_OBJECT_VARS, 3)
            ]
        ),
    )


def _make_m6_beside_curriculum() -> Phase1InstanceGroup:
    return phase1_instances(
        "Preposition on",
        situations=chain(
            *[
                sampled(
                    _beside_template(
                        object_1,
                        object_2,
                        immutableset(),
                        is_training=True,
                        is_right=True,
                    ),
                    chooser=PHASE1_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    max_to_sample=1,
                )
                for object_1 in r.sample(SMALL_OBJECT_VARS, 3)
                for object_2 in r.sample(LARGE_OBJECT_VARS, 3)
            ]
        ),
    )


def _make_m6_under_curriculum() -> Phase1InstanceGroup:
    return phase1_instances(
        "Preposition under",
        situations=chain(
            *[
                sampled(
                    _under_template(
                        object_1,
                        object_2,
                        immutableset(),
                        is_training=True,
                        is_distal=True,
                    ),
                    chooser=PHASE1_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    max_to_sample=1,
                )
                for object_1 in r.sample(SMALL_OBJECT_VARS, 3)
                for object_2 in r.sample(LARGE_OBJECT_VARS, 3)
            ]
        ),
    )


def _make_m6_over_curriculum() -> Phase1InstanceGroup:
    return phase1_instances(
        "Preposition over",
        situations=chain(
            *[
                sampled(
                    _over_template(
                        object_1,
                        object_2,
                        immutableset(),
                        is_training=True,
                        is_distal=True,
                    ),
                    chooser=PHASE1_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    max_to_sample=1,
                )
                for object_1 in r.sample(SMALL_OBJECT_VARS, 3)
                for object_2 in r.sample(LARGE_OBJECT_VARS, 3)
            ]
        ),
    )


def _make_m6_behind_curriculum() -> Phase1InstanceGroup:
    learner_object = standard_object("learner", LEARNER, added_properties=[IS_ADDRESSEE])
    mom = standard_object("mom", MOM, added_properties=[IS_SPEAKER])

    return phase1_instances(
        "Preposition behind",
        situations=chain(
            *[
                sampled(
                    _behind_template(
                        object_1,
                        object_2,
                        immutableset([learner_object, mom]),
                        is_training=True,
                        is_near=True,
                    ),
                    chooser=PHASE1_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    max_to_sample=1,
                )
                for object_1 in r.sample(SMALL_OBJECT_VARS, 3)
                for object_2 in r.sample(LARGE_OBJECT_VARS, 3)
            ]
        ),
    )


def _make_m6_in_front_curriculum() -> Phase1InstanceGroup:
    learner_object = standard_object("learner", LEARNER, added_properties=[IS_ADDRESSEE])
    mom = standard_object("mom", MOM, added_properties=[IS_SPEAKER])

    return phase1_instances(
        "Preposition behind",
        situations=chain(
            *[
                sampled(
                    _behind_template(
                        object_1,
                        object_2,
                        immutableset([learner_object, mom]),
                        is_training=True,
                        is_near=True,
                    ),
                    chooser=PHASE1_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    max_to_sample=1,
                )
                for object_1 in r.sample(SMALL_OBJECT_VARS, 3)
                for object_2 in r.sample(LARGE_OBJECT_VARS, 3)
            ]
        ),
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


def _make_m6_mixed_curriculum() -> Phase1InstanceGroup:
    all_instances = [
        instance
        for instance_group in M6_SUBCURRICULUM_GENERATORS
        for instance in instance_group().instances()  # type: ignore
    ]
    r.shuffle(all_instances)
    return ExplicitWithSituationInstanceGroup("m6_mixed", tuple(all_instances))


def instantiate_subcurricula(subcurricula):
    return [subcurriculum() for subcurriculum in subcurricula]


def make_m6_curriculum():
    return instantiate_subcurricula(M6_SUBCURRICULUM_GENERATORS) + [
        _make_m6_mixed_curriculum()
    ]
