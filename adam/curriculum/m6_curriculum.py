"""
Curricula for M6 Milestone
In this curriculum, we cover the following:
* Object vocabulary:
“mommy, daddy, baby, book, house, car, water, ball, juice, cup, box, chair, head, milk,
hand, dog, truck, door, hat, table, cookie, bird”
* Modifier vocabulary:
basic color terms (red, blue, green, white, black…), one, two, my, your)
"""
from itertools import chain
from typing import List, Callable

from immutablecollections import immutableset

from adam.curriculum import ExplicitWithSituationInstanceGroup
from adam.curriculum.curriculum_utils import Phase1InstanceGroup, standard_object, PHASE1_CHOOSER, phase1_instances
from adam.curriculum.phase1_curriculum import _make_each_object_by_itself_curriculum, \
    _make_objects_with_colors_curriculum, _make_multiple_objects_curriculum, _make_object_on_ground_curriculum
from adam.curriculum.preposition_curriculum import _on_template, _beside_template, _under_template, _over_template, \
    _behind_template
from adam.curriculum.pursuit_curriculum import make_simple_pursuit_curriculum
import random as r

from adam.ontology import IS_ADDRESSEE, IS_SPEAKER
from adam.ontology.phase1_ontology import BIRD, BALL, CUP, BOX, HAT, COOKIE, BOOK, TRUCK, CHAIR, CAR, HOUSE, TABLE, \
    GAILA_PHASE_1_ONTOLOGY, MOM, LEARNER
from adam.situation.templates.phase1_templates import Phase1SituationTemplate, sampled

r.seed(0)

# Create object variables for objects to use in prepositions
small_objects = [BALL, CUP, BOX, HAT, BOOK, COOKIE, BIRD]
large_objects = [TABLE, HOUSE, CAR, CHAIR, TRUCK]
small_object_vars = [standard_object('small_' + str(i), obj) for i, obj in enumerate(small_objects)]
large_object_vars = [standard_object('large_' + str(i), obj) for i, obj in enumerate(large_objects)]


def _make_m6_on_curriculum() -> Phase1InstanceGroup:
    return phase1_instances(
        "Preposition on",
        situations=chain(
            *[sampled(
                _on_template(object_1, object_2, immutableset(), is_training=True),
                chooser=PHASE1_CHOOSER,
                ontology=GAILA_PHASE_1_ONTOLOGY,
                max_to_sample=1
            )
                for object_1 in r.sample(small_object_vars, 3) for object_2 in r.sample(large_object_vars, 3)
            ],)
    )


def _make_m6_beside_curriculum() -> Phase1InstanceGroup:
    return phase1_instances(
        "Preposition on",
        situations=chain(
            *[sampled(
                _beside_template(object_1, object_2, immutableset(), is_training=True, is_right=True),
                chooser=PHASE1_CHOOSER,
                ontology=GAILA_PHASE_1_ONTOLOGY,
                max_to_sample=1
            )
                for object_1 in r.sample(small_object_vars, 3) for object_2 in r.sample(large_object_vars, 3)
            ],)
    )


def _make_m6_under_curriculum() -> Phase1InstanceGroup:
    return phase1_instances(
        "Preposition under",
        situations=chain(
            *[sampled(
                _under_template(object_1, object_2, immutableset(), is_training=True, is_distal=True),
                chooser=PHASE1_CHOOSER,
                ontology=GAILA_PHASE_1_ONTOLOGY,
                max_to_sample=1
            )
                for object_1 in r.sample(small_object_vars, 3) for object_2 in r.sample(large_object_vars, 3)
            ],)
    )


def _make_m6_over_curriculum() -> Phase1InstanceGroup:
    return phase1_instances(
        "Preposition over",
        situations=chain(
            *[sampled(
                _over_template(object_1, object_2, immutableset(), is_training=True, is_distal=True),
                chooser=PHASE1_CHOOSER,
                ontology=GAILA_PHASE_1_ONTOLOGY,
                max_to_sample=1
            )
                for object_1 in r.sample(small_object_vars, 3) for object_2 in r.sample(large_object_vars, 3)
            ],)
    )


def _make_m6_behind_curriculum() -> Phase1InstanceGroup:
    learner_object = standard_object("learner", LEARNER, added_properties=[IS_ADDRESSEE])
    mom = standard_object("mom", MOM, added_properties=[IS_SPEAKER])

    return phase1_instances(
        "Preposition behind",
        situations=chain(
            *[sampled(
                _behind_template(object_1, object_2, immutableset([learner_object, mom]), is_training=True, is_near=True),
                chooser=PHASE1_CHOOSER,
                ontology=GAILA_PHASE_1_ONTOLOGY,
                max_to_sample=1
            )
                for object_1 in r.sample(small_object_vars, 3) for object_2 in r.sample(large_object_vars, 3)
            ],)
    )


def _make_m6_in_front_curriculum() -> Phase1InstanceGroup:
    learner_object = standard_object("learner", LEARNER, added_properties=[IS_ADDRESSEE])
    mom = standard_object("mom", MOM, added_properties=[IS_SPEAKER])

    return phase1_instances(
        "Preposition behind",
        situations=chain(
            *[sampled(
                _behind_template(object_1, object_2, immutableset([learner_object, mom]), is_training=True, is_near=True),
                chooser=PHASE1_CHOOSER,
                ontology=GAILA_PHASE_1_ONTOLOGY,
                max_to_sample=1
            )
                for object_1 in r.sample(small_object_vars, 3) for object_2 in r.sample(large_object_vars, 3)
            ],)
    )


m6_instance_groups: List[Callable] = [
        # Single objects
        _make_each_object_by_itself_curriculum,

        # Objects with modifiers
        # Colors
        _make_objects_with_colors_curriculum,
        # One, two, many
        # _make_multiple_objects_curriculum,

        # Prepositions
        _make_object_on_ground_curriculum,
        _make_m6_on_curriculum,
        _make_m6_beside_curriculum,
        _make_m6_under_curriculum,
        _make_m6_over_curriculum,
        _make_m6_behind_curriculum,
        _make_m6_in_front_curriculum
    ]


def _make_m6_mixed_curriculum() -> Phase1InstanceGroup:
    all_instances = [instance for instance_group in m6_instance_groups for instance in instance_group().instances()]
    r.shuffle(all_instances)
    return ExplicitWithSituationInstanceGroup('m6_mixed', tuple(all_instances))


def make_m6_curriculum():
    return [instance_group() for instance_group in m6_instance_groups] + [_make_m6_mixed_curriculum()]
