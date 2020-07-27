from itertools import chain
from typing import Optional, Sequence

from adam.curriculum.curriculum_utils import (
    PHASE1_CHOOSER_FACTORY,
    Phase1InstanceGroup,
    make_noise_objects,
    phase1_instances,
    standard_object,
)
from adam.curriculum.phase1_curriculum import make_eat_template
from adam.language.dependency import LinearizedDependencyTree
from adam.language.language_generator import LanguageGenerator
from adam.ontology import IS_ADDRESSEE, IS_SPEAKER, THING
from adam.ontology.phase1_ontology import (
    ANIMATE,
    EDIBLE,
    GAILA_PHASE_1_ONTOLOGY,
    NONHUMAN_ANIMAL,
    PERSON,
)
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.situation.templates.phase1_templates import sampled


def make_human_eat_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    object_to_eat = standard_object("object_0", required_properties=[EDIBLE])
    human = standard_object(
        "eater_0", PERSON, banned_properties=[IS_SPEAKER, IS_ADDRESSEE]
    )
    background = make_noise_objects(noise_objects)

    return phase1_instances(
        "Human-Eat-Curriculum",
        # Essen
        sampled(
            make_eat_template(human, object_to_eat, background),
            max_to_sample=num_samples if num_samples else 5,
            ontology=GAILA_PHASE_1_ONTOLOGY,
            chooser=PHASE1_CHOOSER_FACTORY(),
        ),
        language_generator=language_generator,
    )


def make_animal_eat_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    object_to_eat = standard_object("object_0", required_properties=[EDIBLE])
    animal = standard_object("eater_0", NONHUMAN_ANIMAL)
    background = make_noise_objects(noise_objects)

    return phase1_instances(
        "Animal-Eat-Curriculum",
        # Fressen
        sampled(
            make_eat_template(animal, object_to_eat, background),
            max_to_sample=num_samples if num_samples else 5,
            ontology=GAILA_PHASE_1_ONTOLOGY,
            chooser=PHASE1_CHOOSER_FACTORY(),
        ),
        language_generator=language_generator,
    )


def make_german_eat_test_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:

    object_to_eat = standard_object("object_0", required_properties=[EDIBLE])
    eater = standard_object(
        "eater_0",
        THING,
        required_properties=[ANIMATE],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )
    background = make_noise_objects(noise_objects)

    return phase1_instances(
        "german-eating",
        chain(
            *[
                sampled(
                    make_eat_template(eater, object_to_eat, background),
                    max_to_sample=num_samples if num_samples else 5,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                )
            ]
        ),
        language_generator=language_generator,
    )


def make_german_eat_train(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Sequence[Phase1InstanceGroup]:
    return [
        make_human_eat_curriculum(num_samples, num_noise_objects, language_generator),
        make_animal_eat_curriculum(num_samples, num_noise_objects, language_generator),
    ]


def make_german_complete(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Sequence[Phase1InstanceGroup]:
    rtrnr = list(
        make_german_eat_train(num_samples, num_noise_objects, language_generator)
    )
    rtrnr.append(
        make_german_eat_test_curriculum(
            num_samples, num_noise_objects, language_generator
        )
    )
    return rtrnr
