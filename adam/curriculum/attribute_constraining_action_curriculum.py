from itertools import chain

from immutablecollections import immutableset

from adam.curriculum.curriculum_utils import (
    Phase1InstanceGroup,
    standard_object,
    phase1_instances,
    PHASE1_CHOOSER_FACTORY,
)
from adam.curriculum.phase1_curriculum import make_eat_template
from adam.ontology import THING
from adam.ontology.phase1_ontology import (
    PERSON,
    EDIBLE,
    NONHUMAN_ANIMAL,
    GAILA_PHASE_1_ONTOLOGY,
    ANIMATE,
)
from adam.situation.templates.phase1_templates import sampled


def make_human_eat_curriculum(
    num_samples: int = 5, *, noise_objects: int = 0
) -> Phase1InstanceGroup:
    object_to_eat = standard_object("object_0", required_properties=[EDIBLE])
    human = standard_object("eater_0", PERSON)
    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )

    return phase1_instances(
        "Human-Eat-Curriculum",
        # Essen
        sampled(
            make_eat_template(human, object_to_eat, background),
            max_to_sample=num_samples,
            ontology=GAILA_PHASE_1_ONTOLOGY,
            chooser=PHASE1_CHOOSER_FACTORY(),
        ),
    )


def make_animal_eat_curriculum(
    num_samples: int = 5, *, noise_objects: int = 0
) -> Phase1InstanceGroup:
    object_to_eat = standard_object("object_0", required_properties=[EDIBLE])
    animal = standard_object("eater_0", NONHUMAN_ANIMAL)
    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )

    return phase1_instances(
        "Animal-Eat-Curriculum",
        # Fressen
        sampled(
            make_eat_template(animal, object_to_eat, background),
            max_to_sample=num_samples,
            ontology=GAILA_PHASE_1_ONTOLOGY,
            chooser=PHASE1_CHOOSER_FACTORY(),
        ),
    )


def make_german_eat_test_curriculum(
    num_samples: int = 5, *, noise_objects: int = 0
) -> Phase1InstanceGroup:

    object_to_eat = standard_object("object_0", required_properties=[EDIBLE])
    eater = standard_object("eater_0", THING, required_properties=[ANIMATE])
    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )

    return phase1_instances(
        "german-eating",
        chain(
            *[
                sampled(
                    make_eat_template(eater, object_to_eat, background),
                    max_to_sample=num_samples,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                )
            ]
        ),
    )


def make_german_eat_train(num_samples: int = 5, *, noise_objects: int = 0):
    return [
        make_human_eat_curriculum(num_samples=num_samples, noise_objects=noise_objects),
        make_animal_eat_curriculum(num_samples=num_samples, noise_objects=noise_objects),
    ]


def make_german_complete(num_samples: int = 5, *, noise_objects: int = 0):
    return make_german_eat_train(num_samples=num_samples, noise_objects=noise_objects) + [
        make_german_eat_test_curriculum(
            num_samples=num_samples, noise_objects=noise_objects
        )
    ]
