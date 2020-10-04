from itertools import chain
from typing import Sequence, Optional, Iterable
from immutablecollections import immutableset
from more_itertools import flatten
from adam.language.language_generator import LanguageGenerator
from adam.language.dependency import LinearizedDependencyTree
from adam.ontology import OntologyNode
from adam.curriculum.curriculum_utils import (
    Phase1InstanceGroup,
    PHASE1_CHOOSER_FACTORY,
    phase1_instances,
    standard_object,
    learner_template_factory,
    make_noise_objects,
)
from adam.language_specific import MASS_NOUN
from adam.language.dependency.universal_dependencies import NOUN
from adam.ontology.phase2_ontology import gravitationally_aligned_axis_is_largest
from adam.ontology import IS_SPEAKER, IS_ADDRESSEE
from adam.curriculum.phase1_curriculum import (
    make_pass_template,
    throw_on_ground_template,
    throw_template,
    throw_up_down_template,
    throw_to_template,
    bare_move_template,
    transitive_move_template,
    make_jump_template,
    intransitive_roll,
    transitive_roll_with_surface,
    transitive_roll,
    bare_fly,
    fall_on_ground_template,
    falling_template,
    make_take_template,
    make_push_templates,
    make_walk_run_template,
)
from adam.language_specific.english.english_language_generator import (
    USE_ADVERBIAL_PATH_MODIFIER,
)
from adam.ontology import THING
from adam.ontology.phase1_ontology import (
    GAILA_PHASE_1_ONTOLOGY,
    ANIMATE,
    INANIMATE,
    BOX,
    FAST,
    HARD_FORCE,
    SOFT_FORCE,
    SLOW,
    SELF_MOVING,
    CAN_JUMP,
    ROLLABLE,
    CAN_HAVE_THINGS_RESTING_ON_THEM,
    BIRD,
    bigger_than,
    EAT,
    AGENT,
    PATIENT,
    COOKIE,
    WATERMELON,
    TOWARD,
    AWAY_FROM,
    MOM,
    LEARNER,
    DOG,
    BABY,
    DAD,
    CHAIR,
    TABLE,
    THEME,
    SPIN,
    HEAD,
    HAND,
    GROUND,
    NONHUMAN_ANIMAL,
)
from adam.situation import Action, SituationObject
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.situation.templates.phase1_templates import (
    sampled,
    TemplateObjectVariable,
    Phase1SituationTemplate,
)
from adam.language_specific.english.english_phase_1_lexicon import (
    GAILA_PHASE_1_ENGLISH_LEXICON,
)

BOOL_SET = immutableset([True, False])

# easy hack to get all nouns that aren't recognized particulars, body parts, or mass nouns -- i.e. the ones that can be big or small
NODES_TO_CHOOSE_FROM = [
    x[0]
    for x in GAILA_PHASE_1_ENGLISH_LEXICON._ontology_node_to_word.items()  # pylint:disable=protected-access
    if x[1].part_of_speech in [NOUN]
    and MASS_NOUN not in x[1].properties
    and x[0] not in [BABY, HEAD, HAND, GROUND, NONHUMAN_ANIMAL]
]
# differentiate between the nodes that can be modified with tall and those that can't
TALL_ELIGIBLE_NODES = [
    node
    for node in NODES_TO_CHOOSE_FROM
    if gravitationally_aligned_axis_is_largest(node, GAILA_PHASE_1_ONTOLOGY)
]
BIG_ELIGIBLE_NODES = [
    node for node in NODES_TO_CHOOSE_FROM if node not in TALL_ELIGIBLE_NODES
]
CHOOSER = PHASE1_CHOOSER_FACTORY()


def make_eat_big_small_curriculum(  # pylint: disable=unused-argument
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    # "Mom eats a big cookie"
    # We generate situations directly since templates fail to generate plurals.

    learner = SituationObject.instantiate_ontology_node(
        ontology_node=LEARNER,
        debug_handle=LEARNER.handle,
        ontology=GAILA_PHASE_1_ONTOLOGY,
    )
    situations = []

    for eater_ontology_node in [MOM, DAD, BABY, DOG]:
        eater = SituationObject.instantiate_ontology_node(
            ontology_node=eater_ontology_node,
            debug_handle=eater_ontology_node.handle,
            ontology=GAILA_PHASE_1_ONTOLOGY,
        )
        for _object in [COOKIE, WATERMELON]:
            object_to_eat = SituationObject.instantiate_ontology_node(
                ontology_node=_object,
                debug_handle=_object.handle + "_salient",
                ontology=GAILA_PHASE_1_ONTOLOGY,
            )
            object_to_eat2 = SituationObject.instantiate_ontology_node(
                ontology_node=_object,
                debug_handle=_object.handle + "_non_salient",
                ontology=GAILA_PHASE_1_ONTOLOGY,
            )
            other_edibles = [
                SituationObject.instantiate_ontology_node(
                    ontology_node=_object,
                    debug_handle=_object.handle + f"_{i}",
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                )
                for i in range(3)
            ]
            computed_background = [learner]
            computed_background.extend(other_edibles)
            computed_background.extend([object_to_eat2])

            # Big
            for relation_list in [
                [
                    bigger_than(object_to_eat, object_to_eat2),
                    bigger_than(object_to_eat, other_edibles),
                ],
                [
                    bigger_than(object_to_eat2, object_to_eat),
                    bigger_than(other_edibles, object_to_eat),
                ],
            ]:
                situations.append(
                    HighLevelSemanticsSituation(
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        salient_objects=[eater, object_to_eat],
                        other_objects=computed_background,
                        actions=[
                            Action(
                                EAT,
                                argument_roles_to_fillers=[
                                    (AGENT, eater),
                                    (PATIENT, object_to_eat),
                                ],
                            )
                        ],
                        always_relations=relation_list,
                    )
                )

    return phase1_instances(
        "Big - Small Curriculum", situations, language_generator=language_generator
    )


def _tall_x_template(
    background: Iterable[TemplateObjectVariable],
    random_node: OntologyNode = CHOOSER.choice(TALL_ELIGIBLE_NODES),
) -> Phase1SituationTemplate:
    # hack to pick a random node that will yield "tall"
    theme1 = standard_object("theme1", random_node)
    theme2 = standard_object("theme2", random_node)
    computed_background = [theme2]
    computed_background.extend(background)
    return Phase1SituationTemplate(
        f"tall-{theme1.handle}",
        salient_object_variables=[theme1],
        background_object_variables=computed_background,
        asserted_always_relations=[bigger_than(theme1, theme2)],
        gazed_objects=[theme1],
    )


def _big_x_template(
    background: Iterable[TemplateObjectVariable],
    random_node: OntologyNode = CHOOSER.choice(BIG_ELIGIBLE_NODES),
) -> Phase1SituationTemplate:
    # hack to pick a random node that will yield "big"
    theme1 = standard_object("theme1", random_node)
    theme2 = standard_object("theme2", random_node)
    computed_background = [theme2, learner_template_factory()]
    computed_background.extend(background)
    return Phase1SituationTemplate(
        f"big-{theme1.handle}",
        salient_object_variables=[theme1],
        background_object_variables=computed_background,
        asserted_always_relations=[bigger_than(theme1, theme2)],
        gazed_objects=[theme1],
    )


def _little_x_template(
    background: Iterable[TemplateObjectVariable],
    random_node: OntologyNode = CHOOSER.choice(BIG_ELIGIBLE_NODES),
) -> Phase1SituationTemplate:
    # hack to pick a random node that will yield "little"
    theme1 = standard_object("theme1", random_node)
    theme2 = standard_object("theme2", random_node)
    computed_background = [theme2]
    computed_background.extend(background)
    return Phase1SituationTemplate(
        f"little-{theme1.handle}",
        salient_object_variables=[theme1],
        background_object_variables=computed_background,
        asserted_always_relations=[bigger_than(theme2, theme1)],
        gazed_objects=[theme1],
    )


def _short_x_template(
    background: Iterable[TemplateObjectVariable],
    random_node: OntologyNode = CHOOSER.choice(TALL_ELIGIBLE_NODES),
) -> Phase1SituationTemplate:
    # hack to pick a random node that will yield "short"
    theme1 = standard_object("theme1", random_node)
    theme2 = standard_object("theme2", random_node)
    computed_background = [theme2]
    computed_background.extend(background)
    return Phase1SituationTemplate(
        f"short-{theme1.handle}",
        salient_object_variables=[theme1],
        background_object_variables=computed_background,
        asserted_always_relations=[bigger_than(theme2, theme1)],
        gazed_objects=[theme1],
    )


def make_spin_tall_short_curriculum(
    # TODO: Refactor this curriculum
    # See: https://github.com/isi-vista/adam/issues/898
    # pylint: disable=unused-argument
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    # "Mom spins a tall chair"
    # We generate situations directly since templates fail to generate plurals.

    learner = SituationObject.instantiate_ontology_node(
        ontology_node=LEARNER,
        debug_handle=LEARNER.handle,
        ontology=GAILA_PHASE_1_ONTOLOGY,
    )
    situations = []
    for agent_ontology_node in [MOM, DAD, BABY, DOG]:
        agent = SituationObject.instantiate_ontology_node(
            ontology_node=agent_ontology_node,
            debug_handle=agent_ontology_node.handle,
            ontology=GAILA_PHASE_1_ONTOLOGY,
        )
        for _object in [CHAIR, TABLE]:
            theme = SituationObject.instantiate_ontology_node(
                ontology_node=_object,
                debug_handle=_object.handle + "_salient",
                ontology=GAILA_PHASE_1_ONTOLOGY,
            )
            theme2 = SituationObject.instantiate_ontology_node(
                ontology_node=_object,
                debug_handle=_object.handle + "_non_salient",
                ontology=GAILA_PHASE_1_ONTOLOGY,
            )
            other_objs = [
                SituationObject.instantiate_ontology_node(
                    ontology_node=_object,
                    debug_handle=_object.handle + f"_{i}",
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                )
                for i in range(3)
            ]
            computed_background = [learner]
            computed_background.extend(other_objs)
            computed_background.extend([theme2])

            # Tall and short
            for relation_list in [
                [bigger_than(theme2, theme), bigger_than(other_objs, theme)],
                [bigger_than(theme, theme2), bigger_than(theme, other_objs)],
            ]:
                situations.append(
                    HighLevelSemanticsSituation(
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        salient_objects=[agent, theme],
                        other_objects=computed_background,
                        actions=[
                            Action(
                                SPIN,
                                argument_roles_to_fillers=[
                                    (AGENT, agent),
                                    (THEME, theme),
                                ],
                            )
                        ],
                        always_relations=relation_list,
                    )
                )

    return phase1_instances(
        "Tall - Short Curriculum", situations, language_generator=language_generator
    )


def make_imprecise_size_descriptions(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    # we choose random tall and short nodes here
    random_tall_nodes = (
        [CHOOSER.choice(TALL_ELIGIBLE_NODES) for i in range(num_samples)]
        if num_samples
        else [CHOOSER.choice(TALL_ELIGIBLE_NODES) for i in range(5)]
    )
    random_big_nodes = (
        [CHOOSER.choice(BIG_ELIGIBLE_NODES) for i in range(num_samples)]
        if num_samples
        else [CHOOSER.choice(BIG_ELIGIBLE_NODES) for i in range(5)]
    )

    background = make_noise_objects(noise_objects)

    return phase1_instances(
        "Imprecise Size",
        chain(
            flatten(
                # generate big and small for all eligible nodes
                [
                    sampled(
                        template(random_node=node, background=background),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        block_multiple_of_the_same_type=False,
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for node in random_big_nodes
                    for template in [_big_x_template, _little_x_template]
                ]
            ),
            flatten(
                # generate tall and short for all eligible nodes
                [
                    sampled(
                        template(random_node=node, background=background),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=1,
                        block_multiple_of_the_same_type=False,
                    )
                    for node in random_tall_nodes
                    for template in [_tall_x_template, _short_x_template]
                ]
            ),
        ),
        language_generator=language_generator,
    )


def make_throw_imprecise_temporal_descriptions(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    thrower = standard_object(
        "thrower_0",
        THING,
        required_properties=[ANIMATE],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )
    catcher = standard_object(
        "catcher_0",
        THING,
        required_properties=[ANIMATE],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )
    object_thrown = standard_object("object_0", required_properties=[INANIMATE])
    implicit_goal_reference = standard_object("implicit_throw_goal_object", BOX)
    background = make_noise_objects(noise_objects)

    return phase1_instances(
        "throwing-with-temporal-descriptions",
        chain(
            # Throw on Ground
            flatten(
                sampled(
                    throw_on_ground_template(
                        thrower,
                        object_thrown,
                        spatial_properties=[FAST] if is_fast else [SLOW],
                        background=background,
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples if num_samples else 5,
                    block_multiple_of_the_same_type=True,
                )
                for is_fast in BOOL_SET
            ),
            # Throw
            flatten(
                sampled(
                    throw_template(
                        thrower,
                        object_thrown,
                        implicit_goal_reference,
                        spatial_properties=[FAST] if is_fast else [SLOW],
                        background=background,
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples if num_samples else 5,
                    block_multiple_of_the_same_type=True,
                )
                for is_fast in BOOL_SET
            ),
            # Throw up, down
            flatten(
                sampled(
                    throw_up_down_template(
                        thrower,
                        object_thrown,
                        implicit_goal_reference,
                        is_up=is_up,
                        spatial_properties=[FAST] if is_fast else [SLOW],
                        background=background,
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples if num_samples else 5,
                    block_multiple_of_the_same_type=True,
                )
                for is_fast in BOOL_SET
                for is_up in BOOL_SET
            ),
            # Throw To
            flatten(
                sampled(
                    throw_to_template(
                        thrower,
                        object_thrown,
                        catcher,
                        spatial_properties=[FAST] if is_fast else [SLOW],
                        background=background,
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples if num_samples else 5,
                    block_multiple_of_the_same_type=True,
                )
                for is_fast in BOOL_SET
            ),
        ),
        language_generator=language_generator,
    )


def make_move_imprecise_temporal_descriptions(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    self_mover_0 = standard_object(
        "self-mover_0",
        THING,
        required_properties=[SELF_MOVING],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )

    other_mover_0 = standard_object("mover_0", THING, required_properties=[ANIMATE])
    movee_0 = standard_object("movee_0", THING, required_properties=[INANIMATE])
    move_goal_reference = standard_object(
        "move-goal-reference", THING, required_properties=[INANIMATE]
    )
    background = make_noise_objects(noise_objects)

    return phase1_instances(
        "move-with-temporal-descriptions",
        chain(
            # bare move (e.g. "a box moves") is about half of uses in child speed
            flatten(
                sampled(
                    bare_move_template(
                        self_mover_0,
                        move_goal_reference,
                        spatial_properties=[FAST] if is_fast else [SLOW],
                        background=background,
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples if num_samples else 5,
                    block_multiple_of_the_same_type=True,
                )
                for is_fast in BOOL_SET
            ),
            # Transitive Move
            flatten(
                sampled(
                    transitive_move_template(
                        other_mover_0,
                        movee_0,
                        move_goal_reference,
                        spatial_properties=[FAST] if is_fast else [SLOW],
                        background=background,
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples if num_samples else 5,
                    block_multiple_of_the_same_type=True,
                )
                for is_fast in BOOL_SET
            ),
        ),
        language_generator=language_generator,
    )


def make_jump_imprecise_temporal_descriptions(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:

    jumper = standard_object(
        "jumper_0",
        THING,
        required_properties=[CAN_JUMP],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )

    background = make_noise_objects(noise_objects)

    return phase1_instances(
        "jumping",
        chain(
            flatten(
                [
                    sampled(
                        # "A person jumps"
                        make_jump_template(
                            jumper,
                            use_adverbial_path_modifier=use_adverbial_path_modifier,
                            spatial_properties=[FAST] if is_fast else [SLOW],
                            background=background,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                        block_multiple_of_the_same_type=True,
                    )
                    for use_adverbial_path_modifier in (True, False)
                    for is_fast in BOOL_SET
                ]
            )
        ),
        language_generator=language_generator,
    )


def make_take_grab_subtle_verb_distinction(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    taker = standard_object(
        "tosser_passer_0",
        THING,
        required_properties=[ANIMATE],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )
    takee = standard_object("tossee_passee_0", THING, required_properties=[INANIMATE])
    background = make_noise_objects(noise_objects)
    return phase1_instances(
        "taking-grabbing",
        chain(
            flatten(
                [
                    sampled(
                        make_take_template(
                            taker,
                            takee,
                            use_adverbial_path_modifier=use_adverbial_path_modifier,
                            operator=operator,
                            spatial_properties=[HARD_FORCE]
                            if hard_force
                            else [SOFT_FORCE],
                            background=background,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                        block_multiple_of_the_same_type=True,
                    )
                    for use_adverbial_path_modifier in BOOL_SET
                    for hard_force in BOOL_SET
                    for operator in [TOWARD, AWAY_FROM]
                ]
            )
        ),
        language_generator=language_generator,
    )


def make_push_shove_subtle_verb_distinctions(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    pusher = standard_object(
        "pusher_0",
        THING,
        required_properties=[ANIMATE],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )
    pushee = standard_object("pushee_0", THING, required_properties=[INANIMATE])
    push_surface = standard_object(
        "push_surface_0", THING, required_properties=[INANIMATE]
    )
    push_goal = standard_object("push_goal_0", THING, required_properties=[INANIMATE])
    background = make_noise_objects(noise_objects)
    # get all possible templates
    templates = flatten(
        [
            make_push_templates(
                pusher,
                pushee,
                push_surface,
                push_goal,
                use_adverbial_path_modifier=use_adverbial_path_modifier,
                operator=operator,
                spatial_properties=[HARD_FORCE] if hard_force else [SOFT_FORCE],
                background=background,
            )
            for hard_force in BOOL_SET
            for use_adverbial_path_modifier in BOOL_SET
            for operator in [TOWARD, AWAY_FROM]
        ]
    )
    return phase1_instances(
        "pushing-shoving",
        chain(
            flatten(
                [
                    sampled(
                        template,
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                        block_multiple_of_the_same_type=True,
                    )
                    for template in templates
                ]
            )
        ),
        language_generator=language_generator,
    )


def make_walk_run_subtle_verb_distinction(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:

    agent = standard_object(
        "walker_0",
        THING,
        required_properties=[ANIMATE],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )
    background = make_noise_objects(noise_objects)

    return phase1_instances(
        "walking-running",
        chain(
            flatten(
                [
                    sampled(
                        make_walk_run_template(
                            agent,
                            use_adverbial_path_modifier=use_adverbial_path_modifier,
                            operator=operator,
                            spatial_properties=[HARD_FORCE]
                            if hard_force
                            else [SOFT_FORCE],
                            background=background,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                        block_multiple_of_the_same_type=True,
                    )
                    for use_adverbial_path_modifier in BOOL_SET
                    for hard_force in BOOL_SET
                    for operator in [AWAY_FROM, TOWARD]
                ]
            )
        ),
        language_generator=language_generator,
    )


def make_pass_toss_subtle_verb_distinction(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    tosser = standard_object("tosser_passer_0", THING, required_properties=[ANIMATE])
    tossee = standard_object("tossee_passee_0", THING, required_properties=[INANIMATE])
    goal = standard_object("move-goal-reference", THING, required_properties=[INANIMATE])
    background = make_noise_objects(noise_objects)

    return phase1_instances(
        "tossing_passing",
        chain(
            flatten(
                [
                    sampled(
                        make_pass_template(
                            tosser,
                            tossee,
                            goal,
                            use_adverbial_path_modifier=use_adverbial_path_modifier,
                            operator=operator,
                            spatial_properties=[HARD_FORCE]
                            if hard_force
                            else [SOFT_FORCE],
                            background=background,
                        ),
                        block_multiple_of_the_same_type=True,
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 5,
                    )
                    for use_adverbial_path_modifier in BOOL_SET
                    for hard_force in BOOL_SET
                    for operator in [TOWARD, AWAY_FROM]
                ]
            )
        ),
        language_generator=language_generator,
    )


def make_roll_imprecise_temporal_descriptions(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    animate_0 = standard_object("object_0", THING, required_properties=[ANIMATE])
    rollable_0 = standard_object("object_1", required_properties=[ROLLABLE])
    rolling_surface = standard_object(
        "surface", THING, required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
    )
    background = make_noise_objects(noise_objects)

    return phase1_instances(
        "roll-imprecise-temporal-descriptions",
        chain(
            # rolls intransitively
            flatten(
                sampled(
                    intransitive_roll(
                        animate_0,
                        rolling_surface,
                        spatial_properties=[FAST] if is_fast else [SLOW],
                        background=background,
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples if num_samples else 5,
                    block_multiple_of_the_same_type=True,
                )
                for is_fast in BOOL_SET
            ),
            # rolls transitively
            flatten(
                sampled(
                    transitive_roll(
                        animate_0,
                        rollable_0,
                        rolling_surface,
                        spatial_properties=[FAST] if is_fast else [SLOW],
                        background=background,
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples if num_samples else 5,
                    block_multiple_of_the_same_type=True,
                )
                for is_fast in BOOL_SET
            ),
            # rolls on a surface
            flatten(
                sampled(
                    transitive_roll_with_surface(
                        animate_0,
                        rollable_0,
                        rolling_surface,
                        spatial_properties=[FAST] if is_fast else [SLOW],
                        background=background,
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples if num_samples else 5,
                    block_multiple_of_the_same_type=True,
                )
                for is_fast in BOOL_SET
            ),
        ),
        language_generator=language_generator,
    )


def make_fly_imprecise_temporal_descriptions(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    bird = standard_object("bird_0", BIRD)
    syntax_hints_options = ([], [USE_ADVERBIAL_PATH_MODIFIER])  # type: ignore
    background = make_noise_objects(noise_objects)

    return phase1_instances(
        "fly-imprecise-temporal-descripttions",
        chain(
            # Bare Fly
            flatten(
                sampled(
                    bare_fly(
                        bird,
                        up=is_up,
                        syntax_hints=syntax_hints,
                        spatial_properties=[FAST] if is_fast else [SLOW],
                        background=background,
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples if num_samples else 5,
                    block_multiple_of_the_same_type=True,
                )
                for is_up in BOOL_SET
                for syntax_hints in syntax_hints_options
                for is_fast in BOOL_SET
            )
        ),
        language_generator=language_generator,
    )


def make_fall_imprecise_temporal_descriptions(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    arbitary_object = standard_object(
        "object_0", THING, banned_properties=[IS_SPEAKER, IS_ADDRESSEE]
    )
    syntax_hints_options = ([], [USE_ADVERBIAL_PATH_MODIFIER])  # type: ignore
    background = make_noise_objects(noise_objects)

    return phase1_instances(
        f"fall-imprecise-temporal-description",
        chain(
            # Any Object Falling
            flatten(
                sampled(
                    falling_template(
                        arbitary_object,
                        lands_on_ground=object_ends_up_on_ground,
                        syntax_hints=syntax_hints,
                        spatial_properties=[FAST] if is_fast else [SLOW],
                        background=background,
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples if num_samples else 5,
                    block_multiple_of_the_same_type=True,
                )
                for object_ends_up_on_ground in BOOL_SET
                for syntax_hints in syntax_hints_options
                for is_fast in BOOL_SET
            ),
            # Fall on Ground
            flatten(
                sampled(
                    fall_on_ground_template(
                        arbitary_object,
                        spatial_properties=[FAST] if is_fast else [SLOW],
                        background=background,
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples if num_samples else 5,
                    block_multiple_of_the_same_type=True,
                )
                for is_fast in BOOL_SET
            ),
        ),
        language_generator=language_generator,
    )


def make_imprecise_size_curriculum(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Sequence[Phase1InstanceGroup]:
    """
    One particular instantiation of the Imprecise Size Descriptions Curriculum
    """
    return [
        make_imprecise_size_descriptions(
            num_samples, num_noise_objects, language_generator
        )
    ]


def make_imprecise_temporal_descriptions(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Sequence[Phase1InstanceGroup]:
    """
    One particular instantiation of the Imprecise Temporal Descriptions Curriculum
    """
    return [
        make_throw_imprecise_temporal_descriptions(
            num_samples, num_noise_objects, language_generator
        ),
        make_move_imprecise_temporal_descriptions(
            num_samples, num_noise_objects, language_generator
        ),
        make_jump_imprecise_temporal_descriptions(
            num_samples, num_noise_objects, language_generator
        ),
        make_roll_imprecise_temporal_descriptions(
            num_samples, num_noise_objects, language_generator
        ),
        make_fly_imprecise_temporal_descriptions(
            num_samples, num_noise_objects, language_generator
        ),
        make_fall_imprecise_temporal_descriptions(
            num_samples, num_noise_objects, language_generator
        ),
    ]


def make_subtle_verb_distinctions_curriculum(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Sequence[Phase1InstanceGroup]:
    """One particular instanatiation of the Subtle Verb Distinction Curriculum"""
    return [
        make_push_shove_subtle_verb_distinctions(
            num_samples, num_noise_objects, language_generator
        ),
        make_walk_run_subtle_verb_distinction(
            num_samples, num_noise_objects, language_generator
        ),
        make_pass_toss_subtle_verb_distinction(
            num_samples, num_noise_objects, language_generator
        ),
        make_take_grab_subtle_verb_distinction(
            num_samples, num_noise_objects, language_generator
        ),
    ]
