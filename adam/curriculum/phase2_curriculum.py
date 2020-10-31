"""
Additions for the Curricula for DARPA GAILA Phase 2
"""
import math
from adam.axes import HorizontalAxisOfObject, FacingAddresseeAxis
from adam.ontology.phase1_spatial_relations import Direction, PROXIMAL, DISTAL

from immutablecollections import immutableset, ImmutableSet

from adam.language_specific.english.english_language_generator import IGNORE_COLORS
from adam.ontology import IS_SPEAKER, IS_ADDRESSEE
import random

from itertools import chain
from typing import Sequence, Optional, Iterable, Any, Tuple

from more_itertools import flatten

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
from adam.relation import Relation, flatten_relations
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.language.dependency import LinearizedDependencyTree
from adam.curriculum.curriculum_utils import (
    PHASE1_CHOOSER_FACTORY,
    Phase1InstanceGroup,
    standard_object,
    phase2_instances,
    phase1_instances,
    make_noise_objects,
    shuffle_curriculum,
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
    on,
    near,
    strictly_under,
    far,
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
)
from adam.situation import Action
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
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_2_ONTOLOGY,
                    max_to_sample=num_samples,
                    block_multiple_of_the_same_type=True,
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
                    block_multiple_of_the_same_type=True,
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
    situations = flatten(
        build_gaila_m13_curriculum(
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
            chooser=PHASE1_CHOOSER_FACTORY(),
            max_to_sample=num_samples if num_samples else 20,
            block_multiple_of_the_same_type=True,
        ),
        language_generator=language_generator,
    )


INTEGRATED_EXPERIMENT_COLORS = immutableset([BLACK, WHITE, LIGHT_BROWN, DARK_BROWN])
BOOL_SET = (True, False)
NOISE_RELATION_DSL_OPTIONS = immutableset(["on", "beside", "under", "in_front"])


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

    def background_relations_builder(
        background_objects: Iterable[TemplateObjectVariable],
        num_relations: int,
        *,
        target: Optional[TemplateObjectVariable] = None,
        target_2: Optional[TemplateObjectVariable] = None,
    ) -> Iterable[Relation[Any]]:
        if add_noise:
            potential_objects = list(background_objects)
            if target and include_targets_in_noise:
                potential_objects.append(target)
            if target_2 and include_targets_in_noise:
                potential_objects.append(target_2)

            if len(potential_objects) < 2:
                return immutableset()

            relations = []
            for _ in range(num_relations):
                choice = chooser.choice(NOISE_RELATION_DSL_OPTIONS)
                if choice == "on":
                    relations.append(
                        on(
                            chooser.choice(potential_objects),
                            chooser.choice(potential_objects),
                        )
                    )
                elif choice == "beside":
                    obj_choice_2 = chooser.choice(potential_objects)
                    relations.append(
                        near(
                            chooser.choice(potential_objects),
                            obj_choice_2,
                            direction=Direction(
                                positive=chooser.choice(BOOL_SET),
                                relative_to_axis=HorizontalAxisOfObject(
                                    obj_choice_2, index=0
                                ),
                            ),
                        )
                    )
                elif choice == "under":
                    relations.append(
                        strictly_under(
                            chooser.choice(potential_objects),
                            chooser.choice(potential_objects),
                            dist=DISTAL if chooser.choice(BOOL_SET) else PROXIMAL,
                        )
                    )
                elif choice == "in_front":
                    obj_choice_2 = chooser.choice(potential_objects)
                    direction = Direction(
                        positive=chooser.choice(BOOL_SET),
                        relative_to_axis=FacingAddresseeAxis(obj_choice_2),
                    )
                    relations.append(
                        near(
                            chooser.choice(potential_objects),
                            obj_choice_2,
                            direction=direction,
                        )
                        if chooser.choice(BOOL_SET)
                        else far(
                            chooser.choice(potential_objects),
                            obj_choice_2,
                            direction=direction,
                        )
                    )
                else:
                    raise RuntimeError("Invalid relation type in background relations")

            return flatten_relations(relations)
        else:
            return immutableset()

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

    samples_to_template_den = (
        len(target_objects)
        * len(noise_objects_sets)
        * (max_noise_relations - min_noise_relations)
    )

    # Sub-Curriculums
    def single_object_described_curriculum(max_to_sample: int) -> Phase1InstanceGroup:
        def single_object_described_template(
            target: TemplateObjectVariable,
            *,
            background_objects: Iterable[TemplateObjectVariable] = immutableset(),
            relations: Iterable[Tuple[Relation[Any], ...]] = immutableset(),
        ) -> Phase1SituationTemplate:
            return Phase1SituationTemplate(
                name=f"single-object-{target.handle}",
                salient_object_variables=[target],
                background_object_variables=background_objects,
                asserted_always_relations=relations,
                syntax_hints=[IGNORE_COLORS],
            )

        templates = [
            single_object_described_template(
                target_object, background_objects=background_objects
            )
            for target_object in target_objects
            for background_objects in noise_objects_sets
        ]

        return phase2_instances(
            "Single Object",
            flatten(
                [
                    sampled(
                        template,
                        ontology=INTEGRATED_EXPERIMENT_ONTOLOGY,
                        chooser=chooser,
                        max_to_sample=max(
                            math.ceil(max_to_sample / samples_to_template_den), 5
                        ),
                        block_multiple_of_the_same_type=block_multiple_of_same_type,
                    )
                    for template in templates
                ]
            ),
            language_generator=language_generator,
            perception_generator=INTEGRATED_EXPERIMENT_PERCEPTION_GENERATOR,
        )

    def single_attribute_described_curriculum(max_to_sample: int) -> Phase1InstanceGroup:
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

        templates = [
            object_with_color(
                target_object,
                background_objects=background_objects,
                background_relations=background_relations_builder(
                    background_objects, num_relations, target=target_object
                ),
            )
            for target_object in target_color_objects
            for background_objects in noise_objects_sets
            for num_relations in range(min_noise_relations, max_noise_relations)
        ]
        return phase2_instances(
            "Single Attribute",
            flatten(
                [
                    sampled(
                        template,
                        ontology=INTEGRATED_EXPERIMENT_ONTOLOGY,
                        chooser=chooser,
                        max_to_sample=max(
                            math.ceil(max_to_sample / samples_to_template_den), 5
                        ),
                        block_multiple_of_the_same_type=block_multiple_of_same_type,
                    )
                    for template in templates
                ]
            ),
            language_generator=language_generator,
            perception_generator=INTEGRATED_EXPERIMENT_PERCEPTION_GENERATOR,
        )

    def prepositional_relation_described_curriculum(
        max_to_sample: int
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
            required_properties=[
                INTEGRATED_EXPERIMENT_PROP,
                CAN_HAVE_THINGS_RESTING_ON_THEM,
            ],
        )
        templates = [
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
                ),
            )
            for background_objects in noise_objects_sets
            for num_relations in range(min_noise_relations, max_noise_relations)
        ]
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
                    ),
                )
                for is_right in BOOL_SET
                for background_objects in noise_objects_sets
                for num_relations in range(min_noise_relations, max_noise_relations)
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
                    ),
                )
                for is_near in BOOL_SET
                for background_objects in noise_objects_sets
                for num_relations in range(min_noise_relations, max_noise_relations)
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
                    ),
                )
                for is_near in BOOL_SET
                for background_objects in noise_objects_sets
                for num_relations in range(min_noise_relations, max_noise_relations)
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
                            math.ceil(max_to_sample / samples_to_template_den), 5
                        ),
                        block_multiple_of_the_same_type=block_multiple_of_same_type,
                    )
                    for template in templates
                ]
            ),
            language_generator=language_generator,
            perception_generator=INTEGRATED_EXPERIMENT_PERCEPTION_GENERATOR,
        )

    ordered_curriculum = [single_object_described_curriculum(num_samples)]
    if params.boolean("include_attributes", default=True):
        ordered_curriculum.append(single_attribute_described_curriculum(num_samples))
    if params.boolean("include_relations", default=True):
        prepositional_relation_described_curriculum(num_samples)

    return (
        ordered_curriculum
        if not params.boolean("random_order", default=False)
        else shuffle_curriculum(ordered_curriculum, rng=rng)
    )
