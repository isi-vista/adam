"""
Additions for the Curricula for DARPA GAILA Phase 2
"""
import math


from adam.curriculum import (
    AblatedLanguageSituationsInstanceGroup,
    ExplicitWithSituationInstanceGroup,
)
from adam.curriculum.m6_curriculum import M6_CURRICULUM_ALL_OBJECTS

from immutablecollections import immutableset, ImmutableSet

from adam.language_specific.english.english_language_generator import IGNORE_COLORS
from adam.ontology import IS_SPEAKER, IS_ADDRESSEE
import random

from itertools import chain
from typing import Sequence, Optional, Iterable, Any, MutableSequence

from more_itertools import flatten, only

from adam.language.language_generator import LanguageGenerator
from adam.ontology.integrated_learner_experiement_ontology import (
    INTEGRATED_EXPERIMENT_CURRICULUM_OBJECTS,
    INTEGRATED_EXPERIMENT_ONTOLOGY,
    ZUP,
    SPAD,
    DAYGIN,
    MAWG,
    TOMBUR,
    GLIM,
)
from adam.random_utils import RandomChooser
from adam.relation import Relation
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.language.dependency import LinearizedDependencyTree
from adam.curriculum.curriculum_utils import (
    CHOOSER_FACTORY,
    Phase1InstanceGroup,
    standard_object,
    phase2_instances,
    phase1_instances,
    make_noise_objects,
    shuffle_curriculum,
    background_relations_builder,
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
    _make_colour_predicates_curriculum,
    build_gaila_phase1_object_curriculum,
    build_gaila_plurals_curriculum,
    build_gaila_phase1_attribute_curriculum,
    build_gaila_generics_curriculum,
    build_gaila_phase1_verb_curriculum,
    make_sit_transitive,
    make_sit_template_intransitive,
    build_gaila_phase1_relation_curriculum,
)
from adam.curriculum.attribute_constraining_action_curriculum import make_german_complete
from adam.curriculum.pursuit_curriculum import (
    make_pursuit_curriculum,
    make_simple_pursuit_curriculum,
)
from adam.curriculum.preposition_curriculum import (
    make_prepositions_curriculum,
    _on_template,
    _beside_template,
    _behind_template,
    _in_front_template,
)
from adam.curriculum.verbs_with_dynamic_prepositions_curriculum import (
    make_verb_with_dynamic_prepositions_curriculum,
)

from adam.ontology import THING
from adam.ontology.phase1_ontology import (
    CHAIR,
    CUP,
    ANIMATE,
    INANIMATE_OBJECT,
    HOLLOW,
    GAILA_PHASE_1_ONTOLOGY,
    AGENT,
    DRINK,
    LIQUID,
    PERSON,
    THEME,
    DRINK_CONTAINER_AUX,
    inside,
    WHITE,
    LIGHT_BROWN,
    DARK_BROWN,
    BLACK,
    INTEGRATED_EXPERIMENT_PROP,
    CAN_HAVE_THINGS_RESTING_ON_THEM,
    PHASE_1_CURRICULUM_OBJECTS,
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
    INTEGRATED_EXPERIMENT_PERCEPTION_GENERATOR,
    GazePerceivedNoisily,
    HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator,
)
from adam.situation import Action, SituationObject
from adam.situation.templates.phase1_situation_templates import _put_in_template
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    all_possible,
    sampled,
    object_variable,
    TemplateObjectVariable,
)
from vistautils.parameters import Parameters

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
                    chooser=CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_2_ONTOLOGY,
                    max_to_sample=num_samples,
                    block_multiple_of_the_same_type=True,
                )
                if num_samples
                else all_possible(
                    template,
                    chooser=CHOOSER_FACTORY(),
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
                    chooser=CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_2_ONTOLOGY,
                    max_to_sample=num_samples,
                    block_multiple_of_the_same_type=True,
                )
                if num_samples
                else all_possible(
                    cup_template,
                    chooser=CHOOSER_FACTORY(),
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
            chooser=CHOOSER_FACTORY(),
            max_to_sample=num_samples if num_samples else 20,
            block_multiple_of_the_same_type=True,
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
    use_path_instead_of_goal: bool = False,
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
                num_samples,
                num_noise_objects,
                language_generator,
                use_path_instead_of_goal,
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
    use_path_instead_of_goal: bool = False,
) -> Sequence[Phase1InstanceGroup]:
    return list(
        chain(
            build_gaila_phase1_object_curriculum(
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
                num_samples,
                num_noise_objects,
                language_generator,
                use_path_instead_of_goal,
            ),
            list(
                make_subtle_verb_distinctions_curriculum(
                    num_samples, num_noise_objects, language_generator
                )
            ),
            build_functionally_defined_objects_curriculum(
                num_samples, num_noise_objects, language_generator
            ),
            build_gaila_plurals_curriculum(
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
    use_path_instead_of_goal: bool = False,
) -> Sequence[Phase1InstanceGroup]:

    random.seed(0)
    situations: MutableSequence[Phase1InstanceGroup] = flatten(  # type: ignore
        build_gaila_m13_curriculum(  # type: ignore
            num_samples, num_noise_objects, language_generator, use_path_instead_of_goal
        )
    )
    random.shuffle(situations)

    return situations


def _make_multiple_object_template(
    target: TemplateObjectVariable, background: Iterable[TemplateObjectVariable]
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        "object-falls",
        salient_object_variables=[target],
        background_object_variables=background,
    )


def make_multiple_object_situation(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:

    target_object = standard_object("target_object")
    noise_object_variables = [
        standard_object("obj-" + str(idx), banned_properties=[IS_SPEAKER, IS_ADDRESSEE])
        for idx in range(num_noise_objects if num_noise_objects else 0)
    ]

    return phase1_instances(
        "Multiple Objects",
        sampled(
            _make_multiple_object_template(target_object, noise_object_variables),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            chooser=CHOOSER_FACTORY(),
            max_to_sample=num_samples if num_samples else 20,
            block_multiple_of_the_same_type=True,
        ),
        language_generator=language_generator,
    )


def _single_object_described_curriculum(
    max_to_sample: int,
    target_objects: Iterable[TemplateObjectVariable],
    noise_objects_sets: Iterable[Iterable[TemplateObjectVariable]],
    *,
    min_noise_relations: int = 0,
    max_noise_relations: int = 0,
    add_noise: bool,
    chooser: RandomChooser,
    samples_to_template_den: int = 1,
    block_multiple_of_same_type: bool = True,
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
    include_targets_in_noise: bool = False,
    min_samples: int = 6,
) -> Phase1InstanceGroup:
    def single_object_described_template(
        target: TemplateObjectVariable,
        *,
        background_objects: Iterable[TemplateObjectVariable] = immutableset(),
        background_relations: Iterable[Relation[Any]] = immutableset(),
    ) -> Phase1SituationTemplate:
        return Phase1SituationTemplate(
            name=f"single-object-{target.handle}",
            salient_object_variables=[target],
            background_object_variables=background_objects,
            asserted_always_relations=background_relations,
            syntax_hints=[IGNORE_COLORS],
        )

    templates = (
        [
            single_object_described_template(
                target_object,
                background_objects=background_objects,
                background_relations=background_relations_builder(
                    background_objects,
                    num_relations,
                    target=target_object,
                    include_targets_in_noise=include_targets_in_noise,
                ),
            )
            for target_object in target_objects
            for background_objects in noise_objects_sets
            for num_relations in range(min_noise_relations, max_noise_relations)
        ]
        if add_noise
        else [
            single_object_described_template(target_object)
            for target_object in target_objects
        ]
    )

    return phase2_instances(
        "Single Object",
        flatten(
            [
                sampled(
                    template,
                    ontology=INTEGRATED_EXPERIMENT_ONTOLOGY,
                    chooser=chooser,
                    max_to_sample=max(
                        math.ceil(max_to_sample / samples_to_template_den), min_samples
                    ),
                    block_multiple_of_the_same_type=block_multiple_of_same_type,
                )
                for template in templates
            ]
        ),
        language_generator=language_generator,
        perception_generator=INTEGRATED_EXPERIMENT_PERCEPTION_GENERATOR,
    )


def _single_attribute_described_curriculum(
    max_to_sample: int,
    target_color_objects: Iterable[TemplateObjectVariable],
    noise_objects_sets: Iterable[Iterable[TemplateObjectVariable]],
    *,
    min_noise_relations: int = 0,
    max_noise_relations: int = 0,
    add_noise: bool,
    chooser: RandomChooser,
    samples_to_template_den: int = 1,
    block_multiple_of_same_type: bool,
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
    include_targets_in_noise: bool = False,
    min_samples: int = 6,
) -> Phase1InstanceGroup:
    def object_with_color(
        target_with_color: TemplateObjectVariable,
        *,
        background_objects: Iterable[TemplateObjectVariable] = immutableset(),
        background_relations: Iterable[Relation[Any]] = immutableset(),
    ) -> Phase1SituationTemplate:
        return Phase1SituationTemplate(
            name=f"single-attribute-color-{target_with_color.handle}",
            salient_object_variables=[target_with_color],
            background_object_variables=background_objects
            if add_noise
            else immutableset(),
            asserted_always_relations=background_relations
            if add_noise
            else immutableset(),
        )

    templates = (
        [
            object_with_color(
                target_object,
                background_objects=background_objects,
                background_relations=background_relations_builder(
                    background_objects,
                    num_relations,
                    target=target_object,
                    include_targets_in_noise=include_targets_in_noise,
                ),
            )
            for target_object in target_color_objects
            for background_objects in noise_objects_sets
            for num_relations in range(min_noise_relations, max_noise_relations)
        ]
        if add_noise
        else [object_with_color(target_object) for target_object in target_color_objects]
    )
    return phase2_instances(
        "Single Attribute",
        flatten(
            [
                sampled(
                    template,
                    ontology=INTEGRATED_EXPERIMENT_ONTOLOGY,
                    chooser=chooser,
                    max_to_sample=max(
                        math.ceil(max_to_sample / samples_to_template_den), min_samples
                    ),
                    block_multiple_of_the_same_type=block_multiple_of_same_type,
                )
                for template in templates
            ]
        ),
        language_generator=language_generator,
        perception_generator=INTEGRATED_EXPERIMENT_PERCEPTION_GENERATOR,
    )


def _prepositional_relation_described_curriculum(
    max_to_sample: int,
    noise_objects_sets: Iterable[Iterable[TemplateObjectVariable]],
    *,
    min_noise_relations: int = 0,
    max_noise_relations: int = 0,
    add_noise: bool,
    chooser: RandomChooser,
    samples_to_template_den: int = 1,
    block_multiple_of_same_type: bool,
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
    include_targets_in_noise: bool = False,
    min_samples: int = 6,
) -> Phase1InstanceGroup:
    target_1 = standard_object(
        "target_1", THING, required_properties=[INTEGRATED_EXPERIMENT_PROP]
    )
    target_2 = standard_object(
        "target_2", THING, required_properties=[INTEGRATED_EXPERIMENT_PROP]
    )
    target_with_object_on = standard_object(
        "target with object on",
        INANIMATE_OBJECT,
        required_properties=[INTEGRATED_EXPERIMENT_PROP, CAN_HAVE_THINGS_RESTING_ON_THEM],
    )
    templates = (
        [
            _on_template(
                target_1,
                target_with_object_on,
                background_objects,
                is_training=True,
                background_relations=background_relations_builder(
                    background_objects,
                    num_relations,
                    target=target_1,
                    target_2=target_with_object_on,
                    include_targets_in_noise=include_targets_in_noise,
                ),
            )
            for background_objects in noise_objects_sets
            for num_relations in range(min_noise_relations, max_noise_relations)
        ]
        if add_noise
        else [
            _on_template(
                target_1, target_with_object_on, immutableset(), is_training=True
            )
        ]
    )
    templates.extend(
        [
            _beside_template(
                target_1,
                target_2,
                background_objects,
                is_right=is_right,
                is_training=True,
                background_relations=background_relations_builder(
                    background_objects,
                    num_relations,
                    target=target_1,
                    target_2=target_2,
                    include_targets_in_noise=include_targets_in_noise,
                ),
            )
            for is_right in BOOL_SET
            for background_objects in noise_objects_sets
            for num_relations in range(min_noise_relations, max_noise_relations)
        ]
        if add_noise
        else [
            _beside_template(
                target_1, target_2, immutableset(), is_right=is_right, is_training=True
            )
            for is_right in BOOL_SET
        ]
    )
    templates.extend(
        [
            _behind_template(
                target_1,
                target_2,
                background_objects,
                is_near=is_near,
                is_training=True,
                background_relations=background_relations_builder(
                    background_objects,
                    num_relations,
                    target=target_1,
                    target_2=target_2,
                    include_targets_in_noise=include_targets_in_noise,
                ),
            )
            for is_near in BOOL_SET
            for background_objects in noise_objects_sets
            for num_relations in range(min_noise_relations, max_noise_relations)
        ]
        if add_noise
        else [
            _behind_template(
                target_1, target_2, immutableset(), is_near=is_near, is_training=True
            )
            for is_near in BOOL_SET
        ]
    )
    templates.extend(
        [
            _in_front_template(
                target_1,
                target_2,
                background_objects,
                is_near=is_near,
                is_training=True,
                background_relations=background_relations_builder(
                    background_objects,
                    num_relations,
                    target=target_1,
                    target_2=target_2,
                    include_targets_in_noise=include_targets_in_noise,
                ),
            )
            for is_near in BOOL_SET
            for background_objects in noise_objects_sets
            for num_relations in range(min_noise_relations, max_noise_relations)
        ]
        if add_noise
        else [
            _in_front_template(
                target_1, target_2, immutableset(), is_near=is_near, is_training=True
            )
            for is_near in BOOL_SET
        ]
    )

    return phase2_instances(
        "Prepositional Relation",
        flatten(
            [
                sampled(
                    template,
                    ontology=INTEGRATED_EXPERIMENT_ONTOLOGY,
                    chooser=chooser,
                    max_to_sample=max(
                        math.ceil(max_to_sample / samples_to_template_den), min_samples
                    ),
                    block_multiple_of_the_same_type=block_multiple_of_same_type,
                )
                for template in templates
            ]
        ),
        language_generator=language_generator,
        perception_generator=INTEGRATED_EXPERIMENT_PERCEPTION_GENERATOR,
    )


INTEGRATED_EXPERIMENT_COLORS = immutableset([BLACK, WHITE, LIGHT_BROWN, DARK_BROWN])
BOOL_SET = (True, False)


def integrated_pursuit_learner_experiment_curriculum(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
    *,
    params: Parameters = Parameters.empty(),
) -> Sequence[Phase1InstanceGroup]:

    # Load Parameters
    add_noise = params.boolean("add_noise", default=False)
    block_multiple_of_same_type = params.boolean(
        "block_multiple_of_same_type", default=True
    )
    include_targets_in_noise = params.boolean("include_targets_in_noise", default=False)

    min_noise_objects = params.integer("min_noise_objects", default=0)
    max_noise_objects = params.integer(
        "max_noise_objects", default=num_noise_objects if num_noise_objects else 10
    )
    min_noise_relations = params.integer("min_noise_relations", default=0)
    max_noise_relations = params.integer("max_noise_relations", default=5)

    # This value ensure that pursuit gets at least 6 instances of any example
    # As otherwise the lexicalization system might not lexicalize it
    # But if there's lots of variants for noise we don't want to have thousands of examples
    # As could happen combinatorially
    min_samples_per_noise_object_relation_pair = (
        max(
            6
            // (
                max_noise_relations
                - min_noise_relations
                + min_noise_objects
                - max_noise_objects
            ),
            1,
        )
        if add_noise
        else 6
    )

    if num_samples is None:
        num_samples = 50

    # Random Number Generator for Curriculum Use
    rng = random.Random()
    rng.seed(params.integer("random_seed", default=0))

    # Random Chooser for Curriculum Generation
    chooser = RandomChooser.for_seed(params.integer("chooser_seed", default=0))

    # Noise Elements
    noise_objects_sets: ImmutableSet[ImmutableSet[TemplateObjectVariable]] = immutableset(
        [
            immutableset(
                [
                    standard_object(
                        f"{i}_noise_object_{num}",
                        THING,
                        required_properties=[INTEGRATED_EXPERIMENT_PROP],
                    )
                    for num in range(i)
                ]
            )
            for i in range(min_noise_objects, max_noise_objects)
        ]
    )
    if noise_objects_sets.empty() or not add_noise:
        noise_objects_sets = immutableset(immutableset())

    target_objects = [
        standard_object(node.handle, node)
        for node in INTEGRATED_EXPERIMENT_CURRICULUM_OBJECTS
    ]

    target_color_objects = [
        standard_object(f"{node.handle}_{color.handle}", node, added_properties=[color])
        for node in INTEGRATED_EXPERIMENT_CURRICULUM_OBJECTS
        for color in INTEGRATED_EXPERIMENT_COLORS
        if node not in [ZUP, SPAD, DAYGIN, MAWG, TOMBUR, GLIM]
    ]

    # We use a max of 1 here to account for when noise values are not used as otherwise
    # We'd be multiplying by 0 and cause div by 0 errors
    samples_to_template_den = (
        len(target_objects)
        * max(len(noise_objects_sets), 1)
        * max((max_noise_relations - min_noise_relations), 1)
    )

    ordered_curriculum = [
        _single_object_described_curriculum(
            num_samples,
            target_objects,
            noise_objects_sets,
            min_noise_relations=min_noise_relations,
            max_noise_relations=max_noise_relations,
            add_noise=add_noise,
            chooser=chooser,
            samples_to_template_den=samples_to_template_den,
            block_multiple_of_same_type=block_multiple_of_same_type,
            language_generator=language_generator,
            include_targets_in_noise=include_targets_in_noise,
            min_samples=min_samples_per_noise_object_relation_pair,
        )
    ]
    if params.boolean("include_attributes", default=True):
        ordered_curriculum.append(
            _single_attribute_described_curriculum(
                num_samples,
                target_color_objects,
                noise_objects_sets,
                min_noise_relations=min_noise_relations,
                max_noise_relations=max_noise_relations,
                add_noise=add_noise,
                chooser=chooser,
                samples_to_template_den=samples_to_template_den,
                block_multiple_of_same_type=block_multiple_of_same_type,
                language_generator=language_generator,
                include_targets_in_noise=include_targets_in_noise,
                min_samples=min_samples_per_noise_object_relation_pair,
            )
        )
    if params.boolean("include_relations", default=True):
        ordered_curriculum.append(
            _prepositional_relation_described_curriculum(
                num_samples,
                noise_objects_sets,
                min_noise_relations=min_noise_relations,
                max_noise_relations=max_noise_relations,
                add_noise=add_noise,
                chooser=chooser,
                samples_to_template_den=samples_to_template_den,
                block_multiple_of_same_type=block_multiple_of_same_type,
                language_generator=language_generator,
                include_targets_in_noise=include_targets_in_noise,
                min_samples=min_samples_per_noise_object_relation_pair,
            )
        )

    # Convert the 'from situation instances' into explicit instances this allows for
    # 1) Less computation time on the learner experiment to generate the perception graphs
    # 2) Allows us to shuffle the output order which we otherwise can't do

    explicit_instances = [
        instance for sit in ordered_curriculum for instance in sit.instances()
    ]

    return [
        ExplicitWithSituationInstanceGroup(
            name="m18-integrated-learners-experiment",
            instances=tuple(shuffle_curriculum(explicit_instances, rng=rng))
            if params.boolean("shuffled", default=False)
            else tuple(explicit_instances),
        )
    ]


def integrated_pursuit_learner_experiment_test(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
    *,
    params: Parameters = Parameters.empty(),
) -> Sequence[Phase1InstanceGroup]:
    # pylint: disable=unused-argument

    # Random Number Generator for Curriculum Use
    rng = random.Random()
    rng.seed(params.integer("random_seed", default=1))

    # Random Chooser for Curriculum Generation
    chooser = RandomChooser.for_seed(params.integer("chooser_seed", default=1))

    if num_samples is None:
        num_samples = 5

    target_objects = [
        standard_object(node.handle, node)
        for node in INTEGRATED_EXPERIMENT_CURRICULUM_OBJECTS
    ]

    target_color_objects = [
        standard_object(f"{node.handle}_{color.handle}", node, added_properties=[color])
        for node in INTEGRATED_EXPERIMENT_CURRICULUM_OBJECTS
        for color in INTEGRATED_EXPERIMENT_COLORS
        if node not in [ZUP, SPAD, DAYGIN, MAWG, TOMBUR, GLIM]
    ]

    ordered_curriculum = [
        _single_object_described_curriculum(
            num_samples,
            target_objects,
            immutableset(immutableset()),
            add_noise=False,
            chooser=chooser,
            block_multiple_of_same_type=True,
            language_generator=language_generator,
            min_samples=num_samples,
        )
    ]
    if params.boolean("include_attributes", default=True):
        ordered_curriculum.append(
            _single_attribute_described_curriculum(
                num_samples,
                target_color_objects,
                immutableset(immutableset()),
                add_noise=False,
                chooser=chooser,
                block_multiple_of_same_type=True,
                language_generator=language_generator,
                min_samples=num_samples,
            )
        )
    if params.boolean("include_relations", default=True):
        ordered_curriculum.append(
            _prepositional_relation_described_curriculum(
                num_samples,
                immutableset(immutableset()),
                add_noise=False,
                chooser=chooser,
                block_multiple_of_same_type=True,
                language_generator=language_generator,
            )
        )
    # Convert the 'from situation instances' into explicit instances this allows for
    # 1) Less computation time on the learner experiment to generate the perception graphs
    # 2) Allows us to shuffle the output order which we otherwise can't do

    explicit_instances = [
        instance for sit in ordered_curriculum for instance in sit.instances()
    ]

    return [
        ExplicitWithSituationInstanceGroup(
            name="m18-integrated-learners-experiment-test",
            instances=tuple(shuffle_curriculum(explicit_instances, rng=rng))
            if params.boolean("shuffled", default=False)
            else tuple(explicit_instances),
        )
    ]


def build_gaila_phase_2_curriculum(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
    use_path_instead_of_goal: bool = False,
) -> Sequence[Phase1InstanceGroup]:
    """
    One particular instantiation of the curriculum for GAILA Phase 1.
    """
    return list(
        chain(
            # Objects
            build_gaila_phase1_object_curriculum(
                num_samples, num_noise_objects, language_generator
            ),
            # Attributes
            build_gaila_phase1_attribute_curriculum(
                num_samples, num_noise_objects, language_generator
            ),
            # Color predicates
            [
                _make_colour_predicates_curriculum(
                    num_samples, num_noise_objects, language_generator
                )
            ],
            # Generics
            build_gaila_generics_curriculum(
                num_samples, num_noise_objects, language_generator
            ),
            # Relations
            make_prepositions_curriculum(
                num_samples, num_noise_objects, language_generator
            ),
            build_gaila_phase1_relation_curriculum(
                num_samples, num_noise_objects, language_generator
            ),
            # Verbs
            build_gaila_phase1_verb_curriculum(
                num_samples, num_noise_objects, language_generator
            ),
            # Imprecise temporal
            list(
                make_imprecise_temporal_descriptions(
                    num_samples, num_noise_objects, language_generator
                )
            ),
            # Events with dynamic prepositions
            make_verb_with_dynamic_prepositions_curriculum(
                num_samples,
                num_noise_objects,
                language_generator,
                use_path_instead_of_goal,
            ),
            # Subtle verb distinctions
            list(
                make_subtle_verb_distinctions_curriculum(
                    num_samples, num_noise_objects, language_generator
                )
            ),
            # Functionally-defined objects
            build_functionally_defined_objects_curriculum(
                num_samples, num_noise_objects, language_generator
            ),
            # Plurals
            build_gaila_plurals_curriculum(
                num_samples, num_noise_objects, language_generator
            ),
            # Attribute constraining action
            make_german_complete(num_samples, num_noise_objects, language_generator),
            # Part-whole
            [
                _make_part_whole_curriculum(
                    num_samples, num_noise_objects, language_generator
                )
            ],
            # Pursuit
            make_pursuit_curriculum(num_samples, num_noise_objects, language_generator),
            build_pursuit_curriculum(num_samples, num_noise_objects, language_generator),
            # Object learner experiment
            build_object_learner_experiment_curriculum_train(
                num_samples, num_noise_objects, language_generator
            ),
            # Integrated learner experiment
            integrated_pursuit_learner_experiment_curriculum(
                num_samples, num_noise_objects, language_generator
            ),
        )
    )


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
                if fake_language:
                    output_language = LinearizedDependencyTree(
                        dependency_tree=fake_language.dependency_tree,
                        surface_token_order=fake_language.surface_token_order,
                        accurate=False,
                    )
                else:
                    raise RuntimeError("No fake language successfully generated")
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
    perception_generator = (
        HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator(
            ontology=GAILA_PHASE_2_ONTOLOGY, gaze_strategy=gaze_perciever
        )
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
