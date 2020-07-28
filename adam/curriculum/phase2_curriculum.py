"""
Additions for the Curricula for DARPA GAILA Phase 2
"""

from adam.ontology import IS_SPEAKER, IS_ADDRESSEE
import random

from itertools import chain
from typing import Sequence, Optional

from more_itertools import flatten

from adam.language.language_generator import LanguageGenerator
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.language.dependency import LinearizedDependencyTree
from adam.curriculum.curriculum_utils import (
    PHASE1_CHOOSER_FACTORY,
    Phase1InstanceGroup,
    standard_object,
    phase2_instances,
    phase1_instances,
    make_noise_objects,
)
from adam.curriculum.imprecise_descriptions_curriculum import (
    make_imprecise_temporal_descriptions,
    make_subtle_verb_distinctions_curriculum,
    make_spin_tall_short_curriculum,
    make_eat_big_small_curriculum,
)
from adam.curriculum.phase1_curriculum import (
    _make_plural_objects_curriculum,
    _make_pass_curriculum,
    _make_generic_statements_curriculum,
    _make_part_whole_curriculum,
    _make_transitive_roll_curriculum,
    build_gaila_phase1_object_curriculum,
    build_gaila_plurals_curriculum,
    build_gaila_phase1_attribute_curriculum,
    build_gaila_generics_curriculum,
    build_gaila_phase1_verb_curriculum,
    make_sit_transitive,
    make_sit_template_intransitive,
)
from adam.curriculum.preposition_curriculum import make_prepositions_curriculum
from adam.curriculum.verbs_with_dynamic_prepositions_curriculum import (
    make_verb_with_dynamic_prepositions_curriculum,
)
from adam.language_specific.english.english_language_generator import IGNORE_COLORS
from adam.ontology import THING
from adam.ontology.phase1_ontology import (
    CHAIR,
    CUP,
    ANIMATE,
    INANIMATE_OBJECT,
    HOLLOW,
    GAILA_PHASE_1_ONTOLOGY,
    SIT,
    AGENT,
    SIT_GOAL,
    SIT_THING_SAT_ON,
    GOAL,
    DRINK,
    LIQUID,
    PERSON,
    THEME,
    DRINK_CONTAINER_AUX,
    inside,
)
from adam.ontology.phase1_spatial_relations import (
    Region,
    GRAVITATIONAL_UP,
    EXTERIOR_BUT_IN_CONTACT,
)
from adam.ontology.phase2_ontology import (
    CHAIR_2,
    CHAIR_3,
    CHAIR_4,
    CHAIR_5,
    CUP_2,
    CUP_3,
    CUP_4,
    GAILA_PHASE_2_ONTOLOGY,
)
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    GAILA_PHASE_2_PERCEPTION_GENERATOR,
)
from adam.situation import Action
from adam.situation.templates.phase1_situation_templates import _put_in_template
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    all_possible,
    sampled,
    object_variable,
)

# TODO: fix https://github.com/isi-vista/adam/issues/917 which causes us to have to specify that we don't wish to include ME_HACK and YOU_HACK in our curriculum design


def _make_sit_on_chair_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:

    templates = []
    for chair_type in [CHAIR, CHAIR_2, CHAIR_3, CHAIR_4, CHAIR_5]:
        sitter = standard_object(
            "sitter_0",
            THING,
            required_properties=[ANIMATE],
            banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
        )
        seat = standard_object("chair", chair_type)
        templates.append(
            make_sit_transitive(
                sitter, seat, noise_objects, surface=False, syntax_hints=False
            )
        )
        templates.append(
            make_sit_template_intransitive(
                sitter, seat, noise_objects, surface=False, syntax_hints=False
            )
        )

    return phase2_instances(
        "sit on chair",
        chain(
            *[
                sampled(
                    template,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_2_ONTOLOGY,
                    max_to_sample=num_samples,
                )
                if num_samples
                else all_possible(
                    template,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_2_ONTOLOGY,
                )
                for template in templates
            ]
        ),
        perception_generator=GAILA_PHASE_2_PERCEPTION_GENERATOR,
        language_generator=language_generator,
    )


def _make_drink_cups_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:

    templates = []
    for cup in [CUP, CUP_2, CUP_3, CUP_4]:
        cup_obj = standard_object("cup", cup)
        liquid_0 = object_variable("liquid_0", required_properties=[LIQUID])
        person_0 = standard_object(
            "person_0", PERSON, banned_properties=[IS_SPEAKER, IS_ADDRESSEE]
        )

        templates.append(
            Phase1SituationTemplate(
                "drink-cup",
                salient_object_variables=[liquid_0, person_0, cup_obj],
                background_object_variables=make_noise_objects(noise_objects),
                actions=[
                    Action(
                        DRINK,
                        argument_roles_to_fillers=[(AGENT, person_0), (THEME, liquid_0)],
                        auxiliary_variable_bindings=[(DRINK_CONTAINER_AUX, cup_obj)],
                    )
                ],
                asserted_always_relations=[inside(liquid_0, cup_obj)],
            )
        )

    return phase2_instances(
        "drink - cup",
        chain(
            *[
                sampled(
                    cup_template,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_2_ONTOLOGY,
                    max_to_sample=num_samples,
                )
                if num_samples
                else all_possible(
                    cup_template,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_2_ONTOLOGY,
                )
                for cup_template in templates
            ]
        ),
        perception_generator=GAILA_PHASE_2_PERCEPTION_GENERATOR,
        language_generator=language_generator,
    )


def _make_put_in_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    agent = standard_object(
        "agent",
        THING,
        required_properties=[ANIMATE],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )
    theme = standard_object("theme", INANIMATE_OBJECT)
    goal_in = standard_object("goal_in", INANIMATE_OBJECT, required_properties=[HOLLOW])

    return phase1_instances(
        "Capabilities - Put in",
        sampled(
            _put_in_template(agent, theme, goal_in, make_noise_objects(noise_objects)),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            chooser=PHASE1_CHOOSER_FACTORY(),
            max_to_sample=num_samples if num_samples else 20,
        ),
        language_generator=language_generator,
    )


def build_functionally_defined_objects_curriculum(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Sequence[Phase1InstanceGroup]:
    return [
        _make_sit_on_chair_curriculum(
            num_samples, num_noise_objects, language_generator
        ),  # functionally defined objects
        _make_drink_cups_curriculum(num_samples, num_noise_objects, language_generator),
    ]


def build_object_restrictions_curriculum(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Sequence[Phase1InstanceGroup]:
    return [
        _make_transitive_roll_curriculum(
            num_samples, num_noise_objects, language_generator
        ),
        _make_put_in_curriculum(num_samples, num_noise_objects, language_generator),
    ]


def build_gaila_m8_curriculum(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Sequence[Phase1InstanceGroup]:
    return list(
        chain(
            [
                _make_plural_objects_curriculum(
                    num_samples, num_noise_objects, language_generator
                ),  # plurals
                _make_sit_on_chair_curriculum(
                    num_samples, num_noise_objects, language_generator
                ),  # functionally defined objects
                _make_drink_cups_curriculum(
                    num_samples, num_noise_objects, language_generator
                ),
                _make_pass_curriculum(
                    num_samples, num_noise_objects, language_generator
                ),  # Subtle verb distinctions
                _make_generic_statements_curriculum(
                    num_samples, num_noise_objects, language_generator
                ),  # Generics
                _make_part_whole_curriculum(
                    num_samples, num_noise_objects, language_generator
                ),  # Part whole
                _make_transitive_roll_curriculum(
                    num_samples, num_noise_objects, language_generator
                ),  # External Limitations
                make_eat_big_small_curriculum(
                    num_samples, num_noise_objects, language_generator
                ),
                make_spin_tall_short_curriculum(
                    num_samples, num_noise_objects, language_generator
                ),
            ],
            list(
                make_imprecise_temporal_descriptions(
                    num_samples, num_noise_objects, language_generator
                )
            ),  # Imprecise descriptions
            make_verb_with_dynamic_prepositions_curriculum(
                num_samples, num_noise_objects, language_generator
            ),  # Dynamic prepositions
            make_prepositions_curriculum(
                num_samples, num_noise_objects, language_generator
            ),  # Relative prepositions
            list(
                make_subtle_verb_distinctions_curriculum(
                    num_samples, num_noise_objects, language_generator
                )
            ),  # Subtle verb distinctions
        )
    )


def build_gaila_m13_curriculum(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Sequence[Phase1InstanceGroup]:
    return list(
        chain(
            build_gaila_phase1_object_curriculum(
                num_samples, num_noise_objects, language_generator
            ),
            build_gaila_plurals_curriculum(
                num_samples, num_noise_objects, language_generator
            ),
            build_gaila_phase1_attribute_curriculum(
                num_samples, num_noise_objects, language_generator
            ),
            build_gaila_generics_curriculum(
                num_samples, num_noise_objects, language_generator
            ),
            make_prepositions_curriculum(
                num_samples, num_noise_objects, language_generator
            ),
            build_gaila_phase1_verb_curriculum(
                num_samples, num_noise_objects, language_generator
            ),
            list(
                make_imprecise_temporal_descriptions(
                    num_samples, num_noise_objects, language_generator
                )
            ),
            make_verb_with_dynamic_prepositions_curriculum(
                num_samples, num_noise_objects, language_generator
            ),
            list(
                make_subtle_verb_distinctions_curriculum(
                    num_samples, num_noise_objects, language_generator
                )
            ),
            build_functionally_defined_objects_curriculum(
                num_samples, num_noise_objects, language_generator
            ),
        )
    )


def build_m13_shuffled_curriculum(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Sequence[Phase1InstanceGroup]:

    random.seed(0)
    situations = flatten(
        build_gaila_m13_curriculum(num_samples, num_noise_objects, language_generator)
    )
    random.shuffle(situations)

    return situations
