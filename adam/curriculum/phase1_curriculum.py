"""
Curricula for DARPA GAILA Phase 1
"""
from math import ceil

from adam.language_specific.english import DETERMINERS
from adam.ontology.phase1_ontology import DISTAL
from adam.language.language_generator import LanguageGenerator
from adam.language.dependency import LinearizedDependencyTree
from itertools import chain
from typing import Iterable, Sequence, List, Dict, Optional
from adam.language_specific.english.english_language_generator import (
    GAILA_PHASE_1_LANGUAGE_GENERATOR,
    GAILA_PHASE_2_LANGUAGE_GENERATOR,
    INTEGRATED_EXPERIMENT_LANGUAGE_GENERATOR,
)
from adam.language_specific.chinese.chinese_language_generator import (
    GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR,
    GAILA_PHASE_2_CHINESE_LANGUAGE_GENERATOR,
)
from immutablecollections import immutableset
from more_itertools import flatten, first

from adam.axes import AxesInfo, FacingAddresseeAxis, HorizontalAxisOfObject
from adam.curriculum import ExplicitWithSituationInstanceGroup
from adam.curriculum.curriculum_utils import (
    GROUND_OBJECT_TEMPLATE,
    PHASE1_CHOOSER_FACTORY,
    Phase1InstanceGroup,
    phase1_instances,
    standard_object,
    make_noise_objects,
)
from adam.language import TokenSequenceLinguisticDescription
from adam.language_specific.english.english_language_generator import (
    IGNORE_HAS_AS_VERB,
    PREFER_DITRANSITIVE,
    USE_ADVERBIAL_PATH_MODIFIER,
    ATTRIBUTES_AS_X_IS_Y,
    IGNORE_COLORS,
)
from adam.ontology import IS_ADDRESSEE, IS_SPEAKER, THING, OntologyNode
from adam.ontology.during import DuringAction
from adam.ontology.ontology import Ontology
from adam.ontology.phase1_spatial_relations import PathOperator
from adam.ontology.phase1_ontology import (
    AGENT,
    ANIMATE,
    BIRD,
    CAN_BE_SAT_ON_BY_PEOPLE,
    CAN_HAVE_THINGS_RESTING_ON_THEM,
    CAN_JUMP,
    COME,
    HARD_FORCE,
    DRINK,
    DRINK_CONTAINER_AUX,
    EAT,
    EDIBLE,
    FALL,
    FLY,
    GAILA_PHASE_1_ONTOLOGY,
    WALK_SURFACE_AUXILIARY,
    GIVE,
    GOAL,
    GROUND,
    HAS_SPACE_UNDER,
    HOLLOW,
    INANIMATE,
    INANIMATE_OBJECT,
    IS_BODY_PART,
    JUMP,
    JUMP_INITIAL_SUPPORTER_AUX,
    LEARNER,
    LIQUID,
    MOVE,
    MOVE_GOAL,
    PATIENT,
    PERSON,
    PERSON_CAN_HAVE,
    PHASE_1_CURRICULUM_OBJECTS,
    PUSH,
    PUSH_GOAL,
    PUSH_SURFACE_AUX,
    ROLL,
    ROLLABLE,
    ROLL_SURFACE_AUXILIARY,
    SELF_MOVING,
    SIT,
    SIT_GOAL,
    SIT_THING_SAT_ON,
    SPIN,
    TAKE,
    THEME,
    THROW,
    THROW_GOAL,
    TRANSFER_OF_POSSESSION,
    bigger_than,
    contacts,
    far,
    has,
    inside,
    is_recognized_particular,
    near,
    on,
    PASS,
    BABY,
    TRUCK,
    CAR,
    DOG,
    MOM,
    DAD,
    HOUSE,
    BALL,
    WALK,
    strictly_over,
    strictly_under,
)
from adam.ontology.phase1_spatial_relations import (
    AWAY_FROM,
    Direction,
    EXTERIOR_BUT_IN_CONTACT,
    GRAVITATIONAL_UP,
    PROXIMAL,
    Region,
    SpatialPath,
    TO,
    TOWARD,
)
from adam.random_utils import RandomChooser
from adam.relation import flatten_relations
from adam.relation_dsl import negate
from adam.situation import Action, SituationObject
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.situation.templates.phase1_situation_templates import (
    _fly_over_template,
    _go_in_template,
    _go_to_template,
    _go_under_template,
    _jump_over_template,
    _put_in_template,
    _put_on_body_part_template,
    _put_on_template,
    _fly_under_template,
)
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    TemplateObjectVariable,
    action_variable,
    all_possible,
    color_variable,
    object_variable,
    sampled,
)


# TODO: fix https://github.com/isi-vista/adam/issues/917 which causes us to have to specify that we don't wish to include ME_HACK and YOU_HACK in our curriculum design

# given an ontology node, make a template with just it as the addressee
def _make_single_addressee_template(addressee: OntologyNode):
    return Phase1SituationTemplate(
        "single-addressee",
        salient_object_variables=[
            standard_object("addressee", addressee, added_properties=[IS_ADDRESSEE])
        ],
    )


# given an ontology node, make a template with just it as a speaker
def _make_single_speaker_template(speaker: OntologyNode):
    return Phase1SituationTemplate(
        "single-addressee",
        salient_object_variables=[
            standard_object("addressee", speaker, added_properties=[IS_SPEAKER])
        ],
    )


# this is used for cases where we want to learn colour since the single
# object curriculum just assigns random colours
def _make_each_object_without_colour_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],  # pylint: disable=unused-argument
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
    banned_ontology_types: Iterable[OntologyNode] = immutableset(),
) -> Phase1InstanceGroup:
    single_object_template = Phase1SituationTemplate(
        "single-object",
        salient_object_variables=[
            object_variable(
                "object",
                banned_properties=[LIQUID, IS_SPEAKER, IS_ADDRESSEE],
                banned_ontology_types=banned_ontology_types,
            )
        ],
        syntax_hints=[IGNORE_COLORS],
    )

    return phase1_instances(
        "each object by itself",
        chain(
            *[
                sampled(
                    single_object_template,
                    max_to_sample=num_samples,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    block_multiple_of_the_same_type=True,
                )
                if num_samples
                else all_possible(
                    single_object_template,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                ),
                flatten(
                    sampled(
                        _make_single_addressee_template(addressee=object),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=5,
                        block_multiple_of_the_same_type=True,
                    )
                    for object in [MOM, DAD, BABY]
                ),
                flatten(
                    sampled(
                        _make_single_speaker_template(speaker=object),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=5,
                        block_multiple_of_the_same_type=True,
                    )
                    for object in [MOM, DAD, BABY]
                ),
            ]
        ),
        language_generator=language_generator,
    )


# Show each object once by itself
# We ignore noise objects here as this curriculum is
# explicitly noiseless
def _make_each_object_by_itself_curriculum(  # pylint: disable=unused-argument
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
    banned_ontology_types: Iterable[OntologyNode] = immutableset(),
) -> Phase1InstanceGroup:
    color = color_variable("color")
    single_object_template = Phase1SituationTemplate(
        "single-object",
        salient_object_variables=[
            object_variable(
                "object",
                added_properties=[color],
                banned_properties=[LIQUID, IS_SPEAKER, IS_ADDRESSEE],
                banned_ontology_types=banned_ontology_types,
            )
        ],
        syntax_hints=[IGNORE_COLORS],
    )
    single_liquid_template = Phase1SituationTemplate(
        "single-liquids",
        salient_object_variables=[
            object_variable(
                "liquid",
                required_properties=[LIQUID],
                banned_ontology_types=banned_ontology_types,
            )
        ],
        syntax_hints=[IGNORE_COLORS],
    )
    return phase1_instances(
        "each object by itself",
        chain(
            *[
                sampled(
                    single_object_template,
                    max_to_sample=num_samples,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    block_multiple_of_the_same_type=True,
                )
                if num_samples
                else all_possible(
                    single_object_template,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                ),
                sampled(
                    single_liquid_template,
                    max_to_sample=num_samples,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    block_multiple_of_the_same_type=True,
                )
                if num_samples
                else all_possible(
                    single_liquid_template,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                ),
                flatten(
                    sampled(
                        _make_single_addressee_template(addressee=object),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=5,
                        block_multiple_of_the_same_type=True,
                    )
                    for object in [MOM, DAD, BABY]
                ),
                flatten(
                    sampled(
                        _make_single_speaker_template(speaker=object),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=5,
                        block_multiple_of_the_same_type=True,
                    )
                    for object in [MOM, DAD, BABY]
                ),
                flatten(
                    sampled(
                        Phase1SituationTemplate(
                            "only-object",
                            salient_object_variables=[
                                standard_object("addressee", object)
                            ],
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=5,
                        block_multiple_of_the_same_type=True,
                    )
                    for object in [MOM, DAD, BABY]
                ),
            ]
        ),
        language_generator=language_generator,
    )


def _object_with_color_template(
    object_with_color: TemplateObjectVariable, noise_objects: Optional[int]
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        "object-with-color",
        salient_object_variables=[object_with_color],
        background_object_variables=make_noise_objects(noise_objects),
    )


def _make_objects_with_colors_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
    banned_ontology_types: Iterable[OntologyNode] = immutableset(),
) -> Phase1InstanceGroup:
    color = color_variable("color")
    object_with_color = standard_object(
        "object", added_properties=[color], banned_ontology_types=banned_ontology_types
    )

    return phase1_instances(
        "objects with colors",
        chain(
            *[
                sampled(
                    _object_with_color_template(object_with_color, noise_objects),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples if num_samples else 80,
                    block_multiple_of_the_same_type=True,
                )
            ]
        ),
        language_generator=language_generator,
    )


def _object_with_color_is_template(
    object_with_color: TemplateObjectVariable, noise_objects: Optional[int]
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        "object-with-color-is",
        salient_object_variables=[object_with_color],
        syntax_hints=[ATTRIBUTES_AS_X_IS_Y],
        background_object_variables=make_noise_objects(noise_objects),
    )


def _make_objects_with_colors_is_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    color = color_variable("color")
    object_with_color = standard_object("object", added_properties=[color])

    return phase1_instances(
        "objects with colors-is",
        chain(
            *[
                sampled(
                    _object_with_color_is_template(object_with_color, noise_objects),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples if num_samples else 80,
                    block_multiple_of_the_same_type=True,
                )
            ]
        ),
        language_generator=language_generator,
    )


# We ignore any noise objects in this curriculum as pursuit
# has its own implementation method
def _make_plural_objects_curriculum(  # pylint: disable=unused-argument
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
    banned_ontology_types: Iterable[OntologyNode] = immutableset(),
) -> Phase1InstanceGroup:
    def build_object_multiples_situations(
        ontology: Ontology, *, samples_per_object: int = 3, chooser: RandomChooser
    ) -> Iterable[HighLevelSemanticsSituation]:
        for object_type in PHASE_1_CURRICULUM_OBJECTS:
            if object_type in banned_ontology_types:
                continue
            is_liquid = ontology.has_all_properties(object_type, [LIQUID])
            # don't want multiples of named people
            if not is_recognized_particular(ontology, object_type) and not is_liquid:
                for _ in range(samples_per_object):
                    num_objects = chooser.choice(range(2, 4))
                    yield HighLevelSemanticsSituation(
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        salient_objects=[
                            SituationObject.instantiate_ontology_node(
                                ontology_node=object_type,
                                debug_handle=object_type.handle + f"_{idx}",
                                ontology=GAILA_PHASE_1_ONTOLOGY,
                            )
                            for idx in range(num_objects)
                        ],
                        axis_info=AxesInfo(),
                    )

    return phase1_instances(
        "multiples of the same object",
        build_object_multiples_situations(
            ontology=GAILA_PHASE_1_ONTOLOGY,
            chooser=PHASE1_CHOOSER_FACTORY(),
            samples_per_object=max(ceil(num_samples / len(PHASE_1_CURRICULUM_OBJECTS)), 3)
            if num_samples
            else 3,
        ),
        language_generator=language_generator,
    )


def _make_kind_predicates_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
    banned_ontology_types: Iterable[OntologyNode] = immutableset(),
) -> Phase1InstanceGroup:
    """Creates situations and descriptions such as `dogs are animals` with
    a perception of a dog"""
    all_instances = []
    # chinese mapping of objects to their kinds
    chinese_kind_dictionary = {
        "syi1 gwa1": ["shr2 wu4"],
        "chyu1 chi2 bing3": ["shr2 wu4"],
        "syung2": ["dung4 wu4"],
        "gou3": ["dung4 wu4"],
        "mau1": ["dung4 wu4"],
        "nyau3": ["dung4 wu4"],
        "ma1 ma1": ["ren2"],
        "ba4 ba4": ["ren2"],
        "bau3 bau3": ["ren2"],
    }
    # english mapping of objects to their kinds
    english_kind_dictionary = {
        "bear": [["animal", "s"]],
        "dog": [["animal", "s"]],
        "cat": [["animal", "s"]],
        "bird": [["animal", "s"]],
        "Mom": [["people"]],
        "Dad": [["people"]],
        "baby": [["people"]],
        "watermelon": [["food"]],
        "cookie": [["food"]],
        "cow": [["animal", "s"]],
        "beef": [["food"]],
        "chicken": [["animal", "s"], ["food"]],
    }
    # we keep track of the subjects so we only generate one predicate for each subject
    all_subjects: List[str] = []
    for (instance, description, perception) in _make_each_object_by_itself_curriculum(
        num_samples, noise_objects, language_generator, banned_ontology_types
    ).instances():
        linguistic_tokens = description.as_token_sequence()
        if language_generator in [
            GAILA_PHASE_2_CHINESE_LANGUAGE_GENERATOR,
            GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR,
        ]:
            if (
                linguistic_tokens[0] not in all_subjects
                and linguistic_tokens[0] in chinese_kind_dictionary
            ):
                all_subjects.append(linguistic_tokens[0])
                all_instances.append(
                    (
                        instance,
                        TokenSequenceLinguisticDescription(
                            (
                                linguistic_tokens[0],
                                "shr4",
                                chinese_kind_dictionary[linguistic_tokens[0]][0],
                            )
                        ),
                        perception,
                    )
                )
        else:
            if (
                linguistic_tokens[-1] not in all_subjects
                and linguistic_tokens[-1] in english_kind_dictionary
            ):
                for kind in english_kind_dictionary[linguistic_tokens[-1]]:
                    all_subjects.append(linguistic_tokens[-1])
                    all_instances.append(
                        (
                            instance,
                            TokenSequenceLinguisticDescription(
                                (linguistic_tokens[-1], "s", "are", *kind)
                            ),
                            perception,
                        )
                    )
    return ExplicitWithSituationInstanceGroup("kind predicates", all_instances)


def _make_chinese_classifier_single_object_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    """Creates situations and descriptions such as yi1_jr1 shou3 (one CLF hand)"""
    all_instances = []
    classifiers_of_interest = ["yi1_jang1", "yi1_jr1", "yi1_ge4"]
    # accumulate all instances of this object with the desired classifiers
    for (instance, description, perception) in _make_each_object_by_itself_curriculum(
        num_samples, noise_objects, language_generator
    ).instances():
        linguistic_tokens = description.as_token_sequence()
        if linguistic_tokens[0] in classifiers_of_interest:
            all_instances.append((instance, description, perception))
    return ExplicitWithSituationInstanceGroup("single object classifiers", all_instances)


def _make_colour_predicates_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
    banned_ontology_types: Iterable[OntologyNode] = immutableset(),
) -> Phase1InstanceGroup:
    """Creates situations and descriptions such as `cookies are brown' with a single
    cookie that is brown"""
    all_instances = []
    # chinese mapping of objects to the colours they usually are
    chinese_colour_dictionary = {
        "syi1 gwa1": "lyu4 se4",
        "chyu1 chi2 bing3": "chyan3 he2 se4",
        "jr3": "bai2 se4",
        "syung2": "shen1 dzung1 se4",
    }
    # english mapping of objects to the colours they usually are
    english_colour_dictionary = {
        "watermelon": "green",
        "cookie": "light brown",
        "paper": "white",
        "bear": "dark brown",
    }
    # we keep track of the subjects so we only generate one predicate colour for each subject
    all_subjects: List[str] = []
    for (
        instance,
        description,
        perception,
    ) in _make_each_object_without_colour_curriculum(
        num_samples, noise_objects, language_generator, banned_ontology_types
    ).instances():
        linguistic_tokens = description.as_token_sequence()
        if language_generator in [
            GAILA_PHASE_2_CHINESE_LANGUAGE_GENERATOR,
            GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR,
        ]:
            if (
                linguistic_tokens[0] not in all_subjects
                and linguistic_tokens[0] in chinese_colour_dictionary
            ):
                all_subjects.append(linguistic_tokens[0])
                all_instances.append(
                    (
                        instance,
                        TokenSequenceLinguisticDescription(
                            (
                                linguistic_tokens[0],
                                "shr4",
                                chinese_colour_dictionary[linguistic_tokens[0]],
                            )
                        ),
                        perception,
                    )
                )
        else:
            if (
                linguistic_tokens[-1] not in all_subjects
                and linguistic_tokens[-1] in english_colour_dictionary
            ):
                all_subjects.append(linguistic_tokens[-1])
                all_instances.append(
                    (
                        instance,
                        TokenSequenceLinguisticDescription(
                            (
                                linguistic_tokens[-1],
                                "s",
                                "are",
                                english_colour_dictionary[linguistic_tokens[-1]],
                            )
                        ),
                        perception,
                    )
                )
    return ExplicitWithSituationInstanceGroup("colour predicates", all_instances)


# TODO: Refactor this curriculum
def _make_generic_statements_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
    banned_ontology_types: Iterable[OntologyNode] = immutableset(),
) -> Phase1InstanceGroup:
    # Hard-coded examples: we create dynamic instances and replace the linguistic description
    # The way we do this is explained here: https://github.com/isi-vista/adam/issues/771
    all_instances = []
    verbs_to_instances = {
        "eat": _make_eat_curriculum(
            num_samples,
            noise_objects,
            language_generator,
            banned_ontology_types=banned_ontology_types,
        ).instances(),  # E.g babies eat
        "drink": _make_drink_curriculum(
            num_samples,
            noise_objects,
            language_generator,
            banned_ontology_types=banned_ontology_types,
        ).instances(),
        "sit": _make_sit_curriculum(
            num_samples,
            noise_objects,
            language_generator,
            banned_ontology_types=banned_ontology_types,
        ).instances(),
        "jump": _make_jump_curriculum(
            num_samples,
            noise_objects,
            language_generator,
            banned_ontology_types=banned_ontology_types,
        ).instances(),
        "fly": _make_fly_curriculum(
            num_samples,
            noise_objects,
            language_generator,
            banned_ontology_types=banned_ontology_types,
        ).instances(),
    }
    # hack for chinese generics
    verbs_to_ch = {
        "eat": "chr1",
        "drink": "he1",
        "sit": "dzwo4",
        "jump": "tyau4",
        "fly": "fei1",
    }
    for verb, instances in verbs_to_instances.items():
        for (situation, description, perception) in instances:
            subject = [
                token
                for token in description.as_token_sequence()
                if token not in DETERMINERS
            ][0]
            all_instances.append(
                (
                    situation,
                    # the token sequence needs pluralization for English but this isn't morphologically salient for Chinese
                    TokenSequenceLinguisticDescription((subject, "s", verb))
                    if language_generator
                    in [
                        GAILA_PHASE_1_LANGUAGE_GENERATOR,
                        GAILA_PHASE_2_LANGUAGE_GENERATOR,
                    ]
                    else TokenSequenceLinguisticDescription(
                        (subject, "hwei4", verbs_to_ch[verb])
                    ),
                    perception,
                )
            )
    return ExplicitWithSituationInstanceGroup("generics instances", all_instances)


def _make_object_on_ground_curriculum(  # pylint: disable=unused-argument
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    object_0 = standard_object("object_0")
    liquid_0 = object_variable("liquid_0", THING, required_properties=[LIQUID])
    ground = GROUND_OBJECT_TEMPLATE
    computed_background = [ground]
    computed_background.extend(make_noise_objects(noise_objects))

    object_on_ground_template = Phase1SituationTemplate(
        "object-on-ground",
        salient_object_variables=[object_0],
        background_object_variables=computed_background,
        asserted_always_relations=[on(object_0, ground)],
    )

    liquid_on_ground_template = Phase1SituationTemplate(
        "liquid-on-ground",
        salient_object_variables=[liquid_0],
        background_object_variables=computed_background,
        asserted_always_relations=[on(liquid_0, ground)],
    )

    return phase1_instances(
        "object on ground",
        chain(
            *[
                sampled(
                    object_on_ground_template,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples,
                    block_multiple_of_the_same_type=True,
                )
                if num_samples
                else all_possible(
                    object_on_ground_template,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                ),
                sampled(
                    liquid_on_ground_template,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples,
                    block_multiple_of_the_same_type=True,
                )
                if num_samples
                else all_possible(
                    liquid_on_ground_template,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                ),
            ]
        ),
        language_generator=language_generator,
    )


def _x_has_y_template(
    person: TemplateObjectVariable,
    has_object: TemplateObjectVariable,
    *,
    background: Iterable[TemplateObjectVariable] = immutableset(),
    syntax_hints: Iterable[str] = immutableset(),
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{person.handle}-has-{has_object.handle}",
        salient_object_variables=[person, has_object],
        asserted_always_relations=flatten_relations(has(person, has_object)),
        background_object_variables=background,
        syntax_hints=syntax_hints,
    )


def _make_person_has_object_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    inanimate_object_0 = standard_object(
        "inanimate-object", INANIMATE_OBJECT, required_properties=[PERSON_CAN_HAVE]
    )
    background = make_noise_objects(noise_objects)

    return phase1_instances(
        "person has object",
        chain(
            *[
                flatten(
                    sampled(
                        _x_has_y_template(
                            object_variable("person", person),
                            inanimate_object_0,
                            background=background,
                        ),
                        block_multiple_of_the_same_type=True,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        max_to_sample=num_samples if num_samples else 35,
                    )
                    for person in [MOM, DAD, BABY]
                )
            ]
        ),
        language_generator=language_generator,
    )


# TODO: Refactor this curriculum
# See: https://github.com/isi-vista/adam/issues/899
def _make_part_whole_curriculum(  # pylint: disable=unused-argument
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    whole_object_to_parts = {
        BABY: ["head", "hand", "arm"],
        BIRD: ["head", "wing"],
        TRUCK: ["tire"],
        CAR: ["tire", "trailer"],
        DAD: ["head", "hand", "arm"],
        MOM: ["head", "hand", "arm"],
        DOG: ["head", "leg"],
        HOUSE: ["wall", "roof"],
    }
    whole_object_to_parts_ch = {
        BABY: ["tou2", "shou3", "bi4"],
        BIRD: ["tou2", "chr4bang3"],
        TRUCK: ["tai1"],
        CAR: ["tai1", "yu4 gau4 pyan4"],
        DAD: ["tou2", "shou3", "bi4"],
        MOM: ["tou2", "shou3", "bi4"],
        DOG: ["tou2", "twei3"],
        HOUSE: ["bi4", "wu1 ding3"],
    }

    all_instances = []
    currdict: Dict[OntologyNode, List[str]]
    if (
        language_generator == GAILA_PHASE_1_LANGUAGE_GENERATOR
        or language_generator == GAILA_PHASE_2_LANGUAGE_GENERATOR
        or language_generator == INTEGRATED_EXPERIMENT_LANGUAGE_GENERATOR
    ):
        currdict = whole_object_to_parts
    elif (
        language_generator == GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR
        or language_generator == GAILA_PHASE_2_CHINESE_LANGUAGE_GENERATOR
    ):
        currdict = whole_object_to_parts_ch
    else:
        raise RuntimeError("Invalid language generator")
    for whole_object, parts in currdict.items():
        whole = object_variable("whole", whole_object)

        # Get the description sequence for "[whole] has a [part]" Using a part directly causes issues.
        seq = first(
            phase1_instances(
                "desc",
                situations=sampled(
                    _x_has_y_template(whole, object_variable("filler", BALL)),
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    max_to_sample=1,
                    block_multiple_of_the_same_type=True,
                ),
                language_generator=language_generator,
            ).instances()
        )[1].as_token_sequence()

        for part in parts:
            # Replace the filler object with the part object description
            description = TokenSequenceLinguisticDescription(
                tuple([w if (w != "ball" and w != "chyou2") else part for w in seq])
            )

            # Get the situation and perception from just the [whole] object
            instances = phase1_instances(
                "desc",
                situations=sampled(
                    Phase1SituationTemplate(
                        f"{whole.handle}", salient_object_variables=[whole]
                    ),
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    max_to_sample=3,
                    block_multiple_of_the_same_type=True,
                ),
                language_generator=language_generator,
            ).instances()
            for situation, _, perception in instances:
                all_instances.append((situation, description, perception))

    return ExplicitWithSituationInstanceGroup("part of instances", all_instances)


def _make_my_your_object_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    person_0 = standard_object(
        "speaker",
        PERSON,
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
        added_properties=[IS_SPEAKER],
    )
    person_1 = standard_object(
        "addressee",
        PERSON,
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
        added_properties=[IS_ADDRESSEE],
    )
    inanimate_object = standard_object(
        "object", INANIMATE_OBJECT, required_properties=[PERSON_CAN_HAVE]
    )

    owners = (person_0, person_1)

    background_noise = make_noise_objects(noise_objects)

    return phase1_instances(
        "my-your-object",
        chain(
            *[
                sampled(
                    _x_has_y_template(
                        person,
                        inanimate_object,
                        background=chain([person_0], background_noise)
                        if person == person_1
                        else background_noise,
                        syntax_hints=[IGNORE_HAS_AS_VERB],
                    ),
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    max_to_sample=num_samples if num_samples else 20,
                    block_multiple_of_the_same_type=True,
                )
                for person in owners
            ]
        ),
        language_generator=language_generator,
    )


def falling_template(
    theme: TemplateObjectVariable,
    *,
    lands_on_ground: bool,
    syntax_hints: Iterable[str],
    spatial_properties: Iterable[OntologyNode] = immutableset(),
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    ground = GROUND_OBJECT_TEMPLATE
    computed_background = [ground]
    computed_background.extend(background)
    return Phase1SituationTemplate(
        "object-falls",
        salient_object_variables=[theme],
        background_object_variables=computed_background,
        actions=[
            Action(
                action_type=FALL,
                argument_roles_to_fillers=[(THEME, theme)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            theme,
                            SpatialPath(
                                operator=TOWARD,
                                reference_source_object=Region(ground, distance=DISTAL),
                                reference_destination_object=ground,
                                properties=spatial_properties,
                            ),
                        )
                    ]
                ),
            )
        ],
        syntax_hints=syntax_hints,
        before_action_relations=[negate(contacts(theme, ground))],
        after_action_relations=flatten_relations([on(theme, ground)])
        if lands_on_ground
        else flatten_relations([negate(contacts(theme, ground))]),
    )


def fall_on_ground_template(
    theme: TemplateObjectVariable,
    *,
    spatial_properties: Iterable[OntologyNode] = immutableset(),
    background: Iterable[TemplateObjectVariable] = immutableset(),
) -> Phase1SituationTemplate:
    ground = GROUND_OBJECT_TEMPLATE
    return Phase1SituationTemplate(
        "falls-to-ground",
        salient_object_variables=[theme, ground],
        background_object_variables=background,
        actions=[
            Action(
                action_type=FALL,
                argument_roles_to_fillers=[(THEME, theme)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            theme,
                            SpatialPath(
                                None,
                                reference_source_object=Region(ground, distance=DISTAL),
                                reference_destination_object=ground,
                                properties=spatial_properties,
                            ),
                        )
                    ]
                ),
            )
        ],
        before_action_relations=[negate(on(theme, ground))],
        after_action_relations=[on(theme, ground)],
    )


def make_fall_templates(
    background: Iterable[TemplateObjectVariable]
) -> Iterable[Phase1SituationTemplate]:
    arbitary_object = standard_object(
        "object_0", THING, banned_properties=[IS_SPEAKER, IS_ADDRESSEE]
    )
    syntax_hints_options = ([], [USE_ADVERBIAL_PATH_MODIFIER])  # type: ignore

    # Any Object Falling
    object_falling = [
        falling_template(
            arbitary_object,
            lands_on_ground=object_ends_up_on_ground,
            syntax_hints=syntax_hints,
            background=background,
        )
        for object_ends_up_on_ground in (True, False)
        for syntax_hints in syntax_hints_options
    ]

    object_falling.extend(
        [fall_on_ground_template(arbitary_object, background=background)]
    )

    # "ball fell on the ground"
    return object_falling


def _make_fall_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    background = make_noise_objects(noise_objects)
    return phase1_instances(
        "falling objects",
        chain(
            *[
                sampled(
                    template,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples,
                    block_multiple_of_the_same_type=True,
                )
                if num_samples
                else all_possible(
                    template,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                )
                for template in make_fall_templates(background)
            ]
        ),
        language_generator=language_generator,
    )


def make_give_templates(
    background: Iterable[TemplateObjectVariable]
) -> Iterable[Phase1SituationTemplate]:
    action_variable("transfer-verb", with_properties=[TRANSFER_OF_POSSESSION])
    # banning being the speaker or addressee keeps us from trying to instantiate "you" or "me" hack nodes
    giver = object_variable(
        "person_0", PERSON, banned_properties=[IS_SPEAKER, IS_ADDRESSEE]
    )
    recipient = object_variable(
        "person_1", PERSON, banned_properties=[IS_SPEAKER, IS_ADDRESSEE]
    )
    given_object = standard_object("give_object_0")

    for prefer_ditransitive in (True, False):
        yield Phase1SituationTemplate(
            "transfer-of-possession",
            salient_object_variables=[giver, recipient, given_object],
            background_object_variables=background,
            actions=[
                Action(
                    GIVE,
                    argument_roles_to_fillers=[
                        (AGENT, giver),
                        (GOAL, recipient),
                        (THEME, given_object),
                    ],
                )
            ],
            syntax_hints=[PREFER_DITRANSITIVE] if prefer_ditransitive else [],
        )


def _make_transfer_of_possession_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    background = make_noise_objects(noise_objects)
    return phase1_instances(
        "transfer-of-possession",
        chain(
            *[
                sampled(
                    template,
                    max_to_sample=num_samples if num_samples else 100,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    block_multiple_of_the_same_type=True,
                )
                for template in make_give_templates(background)
            ]
        ),
        language_generator=language_generator,
    )


def _make_object_on_object_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    object_ = object_variable("object_0", INANIMATE_OBJECT)
    object_with_surface = object_variable(
        "object_1",
        INANIMATE_OBJECT,
        required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM],
    )
    situation_template = Phase1SituationTemplate(
        "object-on-surface",
        salient_object_variables=[object_, object_with_surface],
        background_object_variables=make_noise_objects(noise_objects),
        constraining_relations=[bigger_than(object_with_surface, object_)],
        asserted_always_relations=[on(object_, object_with_surface)],
    )

    return phase1_instances(
        "objects-on-surfaces",
        sampled(
            situation_template,
            max_to_sample=num_samples if num_samples else 100,
            chooser=PHASE1_CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            block_multiple_of_the_same_type=True,
        ),
        language_generator=language_generator,
    )


def _make_object_beside_object_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    smaller_beside_object = standard_object("object")
    larger_beside_object = standard_object("larger_beside_object")

    situation_template = Phase1SituationTemplate(
        "object-beside-object",
        salient_object_variables=[smaller_beside_object, larger_beside_object],
        background_object_variables=make_noise_objects(noise_objects),
        constraining_relations=[bigger_than(larger_beside_object, smaller_beside_object)],
        asserted_always_relations=[
            near(
                smaller_beside_object,
                larger_beside_object,
                direction=Direction(
                    positive=True,
                    relative_to_axis=HorizontalAxisOfObject(
                        larger_beside_object, index=0
                    ),
                ),
            )
        ],
    )

    return phase1_instances(
        "objects-beside-objects",
        sampled(
            situation_template,
            max_to_sample=num_samples if num_samples else 50,
            chooser=PHASE1_CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            block_multiple_of_the_same_type=True,
        ),
        language_generator=language_generator,
    )


def _make_object_under_or_over_object_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    object_under = standard_object("object_0")
    object_above = standard_object("object_1", required_properties=[HAS_SPACE_UNDER])

    templates = [
        Phase1SituationTemplate(
            f"object-under-object",
            salient_object_variables=[object_above, object_under],
            constraining_relations=[bigger_than(object_above, object_under)],
            asserted_always_relations=[strictly_under(object_under, object_above)],
            background_object_variables=make_noise_objects(noise_objects),
        ),
        Phase1SituationTemplate(
            f"object-over-object",
            salient_object_variables=[object_under, object_above],
            asserted_always_relations=[strictly_over(object_above, object_under)],
            background_object_variables=make_noise_objects(noise_objects),
        ),
    ]

    return phase1_instances(
        "objects-under-over-objects",
        chain(
            *[
                sampled(
                    template,
                    max_to_sample=num_samples if num_samples else 100,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    block_multiple_of_the_same_type=True,
                )
                for template in templates
            ]
        ),
        language_generator=language_generator,
    )


def _make_object_in_other_object_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    object_ = standard_object("object_0")
    liquid = object_variable(
        "liquid_0", required_properties=[LIQUID], banned_properties=[IS_BODY_PART]
    )
    containing_object = standard_object("object_1", required_properties=[HOLLOW])
    liquid_containing_object = standard_object(
        "object_2", required_properties=[HOLLOW, PERSON_CAN_HAVE]
    )
    solid_template = Phase1SituationTemplate(
        "solid-containment",
        salient_object_variables=[object_, containing_object],
        constraining_relations=[bigger_than(containing_object, object_)],
        asserted_always_relations=[inside(object_, containing_object)],
        background_object_variables=make_noise_objects(noise_objects),
    )
    liquid_template = Phase1SituationTemplate(
        "liquid-containment",
        salient_object_variables=[liquid, liquid_containing_object],
        asserted_always_relations=[inside(liquid, liquid_containing_object)],
        background_object_variables=make_noise_objects(noise_objects),
    )

    return phase1_instances(
        "objects-in-other-objects",
        chain(
            *[
                sampled(
                    liquid_template,
                    max_to_sample=num_samples if num_samples else 25,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    block_multiple_of_the_same_type=True,
                ),
                sampled(
                    solid_template,
                    max_to_sample=num_samples * 3 if num_samples else 75,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    block_multiple_of_the_same_type=True,
                ),
            ]
        ),
        language_generator=language_generator,
    )


def bare_fly(
    agent: TemplateObjectVariable,
    *,
    up: bool,
    syntax_hints: Iterable[str],
    spatial_properties: Iterable[OntologyNode] = immutableset(),
    background: Iterable[TemplateObjectVariable] = immutableset(),
) -> Phase1SituationTemplate:
    ground = GROUND_OBJECT_TEMPLATE
    return Phase1SituationTemplate(
        "fly",
        salient_object_variables=[agent],
        background_object_variables=background,
        actions=[
            Action(
                FLY,
                argument_roles_to_fillers=[(AGENT, agent)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            agent,
                            SpatialPath(
                                AWAY_FROM if up else TOWARD,
                                reference_source_object=Region(ground, distance=DISTAL)
                                if (not up) and syntax_hints
                                else ground,
                                reference_destination_object=Region(
                                    ground, distance=DISTAL
                                )
                                if up and syntax_hints
                                else ground,
                                properties=spatial_properties,
                            ),
                        )
                    ]
                ),
            )
        ],
        syntax_hints=syntax_hints,
    )


def make_fly_templates(
    background: Iterable[TemplateObjectVariable],
    banned_ontology_types: Iterable[OntologyNode] = immutableset(),
) -> Iterable[Phase1SituationTemplate]:
    bird = standard_object("bird_0", BIRD, banned_ontology_types=banned_ontology_types)
    object_0 = standard_object(
        "object_0",
        THING,
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
        banned_ontology_types=banned_ontology_types,
    )
    object_with_space_under = standard_object(
        "object_with_space_under",
        THING,
        required_properties=[HAS_SPACE_UNDER],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
        banned_ontology_types=banned_ontology_types,
    )
    syntax_hints_options = ([], [USE_ADVERBIAL_PATH_MODIFIER])  # type: ignore

    fly_templates = [
        bare_fly(bird, up=up, syntax_hints=syntax_hints, background=background)
        for up in (True, False)
        for syntax_hints in syntax_hints_options
    ]
    # We have fly under disabled due to long run times
    # See https://github.com/isi-vista/adam/issues/672
    return fly_templates + [
        _fly_under_template(bird, object_with_space_under, background),
        _fly_over_template(bird, object_0, background),
    ]


def _make_fly_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
    banned_ontology_types: Iterable[OntologyNode] = immutableset(),
) -> Phase1InstanceGroup:
    return phase1_instances(
        "flying",
        chain(
            flatten(
                [
                    sampled(
                        template,
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples,
                        block_multiple_of_the_same_type=True,
                    )
                    if num_samples
                    else all_possible(
                        template,
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                    )
                    for template in make_fly_templates(
                        background=make_noise_objects(
                            noise_objects, banned_ontology_types=banned_ontology_types
                        ),
                        banned_ontology_types=banned_ontology_types,
                    )
                ]
            )
        ),
        language_generator=language_generator,
    )


def intransitive_roll(
    agent: TemplateObjectVariable,
    surface: TemplateObjectVariable,
    *,
    spatial_properties: Iterable[OntologyNode] = immutableset(),
    background: Iterable[TemplateObjectVariable] = immutableset(),
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        "roll-intransitive",
        salient_object_variables=[agent],
        background_object_variables=background,
        actions=[
            Action(
                ROLL,
                argument_roles_to_fillers=[(AGENT, agent)],
                auxiliary_variable_bindings=[(ROLL_SURFACE_AUXILIARY, surface)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            agent,
                            SpatialPath(
                                None,
                                reference_source_object=surface,
                                reference_destination_object=surface,
                                properties=spatial_properties,
                            ),
                        )
                    ]
                ),
            )
        ],
        constraining_relations=[bigger_than(surface, agent)],
    )


def transitive_roll(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    surface: TemplateObjectVariable,
    *,
    spatial_properties: Iterable[OntologyNode] = immutableset(),
    background: Iterable[TemplateObjectVariable] = immutableset(),
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        "roll-transitive",
        salient_object_variables=[agent, theme],
        background_object_variables=background,
        actions=[
            Action(
                ROLL,
                argument_roles_to_fillers=[(AGENT, agent), (THEME, theme)],
                auxiliary_variable_bindings=[(ROLL_SURFACE_AUXILIARY, surface)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            theme,
                            SpatialPath(
                                None,
                                reference_source_object=surface,
                                reference_destination_object=surface,
                                properties=spatial_properties,
                            ),
                        )
                    ]
                ),
            )
        ],
        constraining_relations=[bigger_than(agent, theme)],
    )


def transitive_roll_with_surface(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    surface: TemplateObjectVariable,
    *,
    spatial_properties: Iterable[OntologyNode] = immutableset(),
    background: Iterable[TemplateObjectVariable] = immutableset(),
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        "roll-transitive-with-salient-surface",
        salient_object_variables=[agent, theme, surface],
        background_object_variables=background,
        actions=[
            Action(
                ROLL,
                argument_roles_to_fillers=[(AGENT, agent), (THEME, theme)],
                auxiliary_variable_bindings=[(ROLL_SURFACE_AUXILIARY, surface)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            theme,
                            SpatialPath(
                                None,
                                reference_source_object=surface,
                                reference_destination_object=surface,
                                properties=spatial_properties,
                            ),
                        )
                    ]
                ),
            )
        ],
        asserted_always_relations=[on(theme, surface)],
        constraining_relations=[bigger_than([surface, agent], theme)],
    )


def make_roll_templates(
    noise_objects: Optional[int]
) -> Sequence[Phase1SituationTemplate]:
    animate_0 = standard_object(
        "object_0",
        THING,
        required_properties=[ANIMATE],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )
    rollable_0 = standard_object(
        "object_1", INANIMATE_OBJECT, required_properties=[ROLLABLE]
    )
    rolling_surface = standard_object(
        "surface", THING, required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
    )
    background = make_noise_objects(noise_objects)

    return [
        # rolls intransitively
        intransitive_roll(animate_0, rolling_surface, background=background),
        # # rolls transitively
        transitive_roll(animate_0, rollable_0, rolling_surface, background=background),
        # rolls on a surface
        transitive_roll_with_surface(
            animate_0, rollable_0, rolling_surface, background=background
        ),
    ]


def _make_roll_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    return phase1_instances(
        "rolling",
        chain(
            *[
                sampled(
                    situation,
                    max_to_sample=num_samples if num_samples else 25,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    block_multiple_of_the_same_type=True,
                )
                for situation in make_roll_templates(noise_objects)
            ]
        ),
        language_generator=language_generator,
    )


def make_transitive_roll_templates(
    noise_objects: Optional[int]
) -> Iterable[Phase1SituationTemplate]:
    animate_0 = standard_object("object_0", THING, required_properties=[ANIMATE])
    rollable_0 = standard_object(
        "object_1", INANIMATE_OBJECT, required_properties=[ROLLABLE]
    )
    rolling_surface = standard_object(
        "surface", THING, required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
    )
    background = make_noise_objects(noise_objects)

    return [
        # rolls transitively
        transitive_roll(animate_0, rollable_0, rolling_surface, background=background),
        # rolls on a surface
        transitive_roll_with_surface(
            animate_0, rollable_0, rolling_surface, background=background
        ),
    ]


def _make_transitive_roll_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    return phase1_instances(
        "rolling",
        chain(
            *[
                sampled(
                    situation,
                    max_to_sample=num_samples if num_samples else 25,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    block_multiple_of_the_same_type=True,
                )
                for situation in make_transitive_roll_templates(noise_objects)
            ]
        ),
        language_generator=language_generator,
    )


def _make_speaker_addressee_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    speaker = standard_object(
        "speaker_0",
        PERSON,
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
        added_properties=[IS_SPEAKER],
    )
    addressee = standard_object(
        "addressee_0",
        PERSON,
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
        added_properties=[IS_ADDRESSEE],
    )
    given_object = standard_object("given_object", INANIMATE_OBJECT)

    def _make_templates() -> Iterable[Phase1SituationTemplate]:
        for prefer_ditransitive in (True, False):
            # "you give Mom the cookie"
            yield Phase1SituationTemplate(
                "addressee-agent",
                salient_object_variables=[speaker, addressee, given_object],
                background_object_variables=make_noise_objects(noise_objects),
                actions=[
                    Action(
                        GIVE,
                        argument_roles_to_fillers=[
                            (AGENT, addressee),
                            (GOAL, speaker),
                            (THEME, given_object),
                        ],
                    )
                ],
                syntax_hints=[PREFER_DITRANSITIVE] if prefer_ditransitive else [],
            )

            # "Mom gives you the cookie"
            yield Phase1SituationTemplate(
                "addressee-goal",
                salient_object_variables=[speaker, addressee, given_object],
                background_object_variables=make_noise_objects(noise_objects),
                actions=[
                    Action(
                        GIVE,
                        argument_roles_to_fillers=[
                            (AGENT, speaker),
                            (GOAL, addressee),
                            (THEME, given_object),
                        ],
                    )
                ],
                syntax_hints=[PREFER_DITRANSITIVE] if prefer_ditransitive else [],
            )

    return phase1_instances(
        "addressee_curriculum",
        chain(
            *[
                flatten(
                    sampled(
                        template,
                        max_to_sample=num_samples if num_samples else 25,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        block_multiple_of_the_same_type=True,
                    )
                    for template in _make_templates()
                )
            ]
        ),
        language_generator=language_generator,
    )


def make_jump_template(
    agent: TemplateObjectVariable,
    *,
    use_adverbial_path_modifier: bool,
    spatial_properties: Iterable[OntologyNode] = immutableset(),
    background: Iterable[TemplateObjectVariable] = immutableset(),
) -> Phase1SituationTemplate:
    ground = GROUND_OBJECT_TEMPLATE
    return Phase1SituationTemplate(
        "jump-on-ground",
        salient_object_variables=[agent],
        background_object_variables=background,
        actions=[
            Action(
                JUMP,
                argument_roles_to_fillers=[(AGENT, agent)],
                auxiliary_variable_bindings=[(JUMP_INITIAL_SUPPORTER_AUX, ground)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            agent,
                            SpatialPath(
                                operator=AWAY_FROM
                                if use_adverbial_path_modifier
                                else None,
                                reference_source_object=ground,
                                reference_destination_object=ground,
                                properties=spatial_properties,
                            ),
                        )
                    ]
                ),
            )
        ],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER] if use_adverbial_path_modifier else [],
    )


def make_pass_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal: TemplateObjectVariable,
    *,
    use_adverbial_path_modifier: bool,
    operator: PathOperator = None,
    spatial_properties: Iterable[OntologyNode] = immutableset(),
    background: Iterable[TemplateObjectVariable] = immutableset(),
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle} tosses {theme.handle}",
        salient_object_variables=[agent, theme, goal],
        background_object_variables=background,
        actions=[
            Action(
                PASS,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (THEME, theme),
                    (GOAL, Region(goal, distance=PROXIMAL)),
                ],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            agent,
                            SpatialPath(
                                None,
                                reference_source_object=Region(goal, distance=DISTAL),
                                reference_destination_object=goal,
                                properties=spatial_properties,
                            ),
                        ),
                        (
                            agent,
                            SpatialPath(
                                operator=operator,
                                reference_source_object=Region(
                                    GROUND_OBJECT_TEMPLATE, distance=DISTAL
                                )
                                if (operator and operator == TOWARD)
                                else GROUND_OBJECT_TEMPLATE,
                                reference_destination_object=Region(
                                    GROUND_OBJECT_TEMPLATE, distance=DISTAL
                                )
                                if (operator and operator == AWAY_FROM)
                                else GROUND_OBJECT_TEMPLATE,
                                properties=spatial_properties,
                            ),
                        ),
                    ]
                ),
            )
        ],
        constraining_relations=[bigger_than(agent, theme)],
        after_action_relations=[near(theme, goal)],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER] if use_adverbial_path_modifier else [],
    )


def make_jump_templates(noise_objects: Optional[int]):
    jumper = standard_object(
        "jumper_0",
        THING,
        required_properties=[CAN_JUMP],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )
    background = make_noise_objects(noise_objects)
    for use_adverbial_path_modifier in (True, False):
        yield make_jump_template(
            jumper,
            use_adverbial_path_modifier=use_adverbial_path_modifier,
            background=background,
        )


def _make_jump_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
    banned_ontology_types: Iterable[OntologyNode] = immutableset(),
) -> Phase1InstanceGroup:

    jumper = standard_object(
        "jumper_0",
        THING,
        required_properties=[CAN_JUMP],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
        banned_ontology_types=banned_ontology_types,
    )
    jumped_over = standard_object(
        "jumped_over",
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
        banned_ontology_types=banned_ontology_types,
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
                            background=background,
                        ),
                        block_multiple_of_the_same_type=True,
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 25,
                    )
                    for use_adverbial_path_modifier in (True, False)
                ]
            ),
            flatten(
                [
                    sampled(
                        _jump_over_template(jumper, jumped_over, background),
                        max_to_sample=num_samples if num_samples else 25,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        block_multiple_of_the_same_type=True,
                    )
                ]
            ),
        ),
        language_generator=language_generator,
    )


def make_put_templates(noise_objects: Optional[int]) -> Iterable[Phase1SituationTemplate]:
    putter = standard_object(
        "putter_0",
        THING,
        required_properties=[ANIMATE],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )
    object_put = standard_object("object_0", required_properties=[INANIMATE])
    on_region_object = standard_object(
        "on_region_object", required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
    )

    in_region_object = standard_object(
        "in_region_object", required_properties=[HOLLOW], banned_properties=[ANIMATE]
    )

    background = make_noise_objects(noise_objects)

    return [
        _put_on_template(putter, object_put, on_region_object, background),
        _put_in_template(putter, object_put, in_region_object, background),
    ]


def _make_put_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    return phase1_instances(
        "putting",
        chain(
            flatten(
                [
                    sampled(
                        template,
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 25,
                        block_multiple_of_the_same_type=True,
                    )
                    for template in make_put_templates(noise_objects)
                ]
            )
        ),
        language_generator=language_generator,
    )


def _make_put_on_speaker_addressee_body_part_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    speaker_putter = standard_object(
        "speaker_putter_0",
        THING,
        required_properties=[ANIMATE],
        added_properties=[IS_SPEAKER],
    )
    addressee_putter = standard_object(
        "addressee_putter_0",
        THING,
        required_properties=[ANIMATE],
        added_properties=[IS_ADDRESSEE],
    )
    object_put = standard_object("object_put_0", required_properties=[INANIMATE])

    body_part_of_putter = object_variable(
        "body_part_of_putter",
        required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM, IS_BODY_PART],
    )

    return phase1_instances(
        "putting-on-body-part-addressee-speaker",
        chain(
            flatten(
                [
                    sampled(
                        _put_on_body_part_template(
                            putter,
                            object_put,
                            body_part_of_putter,
                            background=make_noise_objects(noise_objects),
                        ),
                        block_multiple_of_the_same_type=True,
                        max_to_sample=num_samples if num_samples else 25,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                    )
                    for putter in [speaker_putter, addressee_putter]
                ]
            )
        ),
        language_generator=language_generator,
    )


def make_drink_template(
    agent: TemplateObjectVariable,
    liquid: TemplateObjectVariable,
    container: TemplateObjectVariable,
    noise_objects: Optional[int],
) -> Phase1SituationTemplate:
    background = make_noise_objects(noise_objects)

    return Phase1SituationTemplate(
        "drink",
        salient_object_variables=[agent, liquid],
        background_object_variables=background,
        actions=[
            Action(
                DRINK,
                argument_roles_to_fillers=[(AGENT, agent), (THEME, liquid)],
                auxiliary_variable_bindings=[(DRINK_CONTAINER_AUX, container)],
            )
        ],
        asserted_always_relations=[inside(liquid, container)],
    )


def make_drink_from_template(
    agent: TemplateObjectVariable,
    liquid: TemplateObjectVariable,
    container: TemplateObjectVariable,
    noise_objects: Optional[int],
) -> Phase1SituationTemplate:
    background = make_noise_objects(noise_objects)

    return Phase1SituationTemplate(
        "drink-from",
        salient_object_variables=[agent, liquid, container],
        background_object_variables=background,
        actions=[
            Action(
                DRINK,
                argument_roles_to_fillers=[(AGENT, agent), (THEME, liquid)],
                auxiliary_variable_bindings=[(DRINK_CONTAINER_AUX, container)],
            )
        ],
        asserted_always_relations=[inside(liquid, container)],
    )


def _make_drink_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
    banned_ontology_types: Iterable[OntologyNode] = immutableset(),
) -> Phase1InstanceGroup:
    object_0 = standard_object(
        "object_0",
        required_properties=[HOLLOW, PERSON_CAN_HAVE],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
        banned_ontology_types=banned_ontology_types,
    )
    liquid_0 = object_variable(
        "liquid_0",
        required_properties=[LIQUID],
        banned_ontology_types=banned_ontology_types,
    )
    person_0 = standard_object(
        "person_0",
        PERSON,
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
        banned_ontology_types=banned_ontology_types,
    )
    return phase1_instances(
        "drinking",
        chain(
            *[
                sampled(
                    make_drink_template(person_0, liquid_0, object_0, noise_objects),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples,
                    block_multiple_of_the_same_type=True,
                )
                if num_samples
                else all_possible(
                    make_drink_template(person_0, liquid_0, object_0, noise_objects),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                ),
                sampled(
                    make_drink_from_template(
                        person_0, liquid_0, object_0, noise_objects=noise_objects
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples if num_samples else 5,
                    block_multiple_of_the_same_type=True,
                ),
            ]
        ),
        language_generator=language_generator,
    )


def make_eat_template(
    agent: TemplateObjectVariable,
    patient: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable] = immutableset(),
) -> Phase1SituationTemplate:

    # "Mom eats a cookie"
    return Phase1SituationTemplate(
        "eat-object",
        salient_object_variables=[patient, agent],
        background_object_variables=background,
        actions=[
            Action(EAT, argument_roles_to_fillers=[(AGENT, agent), (PATIENT, patient)])
        ],
    )


def _make_eat_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
    banned_ontology_types: Iterable[OntologyNode] = immutableset(),
) -> Phase1InstanceGroup:
    # TODO: "eat it up"
    # https://github.com/isi-vista/adam/issues/267

    object_to_eat = standard_object(
        "object_0",
        required_properties=[EDIBLE],
        banned_ontology_types=banned_ontology_types,
    )
    eater = standard_object(
        "eater_0",
        THING,
        required_properties=[ANIMATE],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
        banned_ontology_types=banned_ontology_types,
    )
    background = make_noise_objects(noise_objects)

    return phase1_instances(
        "eating",
        chain(
            *[
                sampled(
                    make_eat_template(eater, object_to_eat, background),
                    max_to_sample=num_samples if num_samples else 25,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    block_multiple_of_the_same_type=True,
                )
            ]
        ),
        language_generator=language_generator,
    )


def make_sit_template_intransitive(
    agent: TemplateObjectVariable,
    sit_surface: TemplateObjectVariable,
    noise_objects: Optional[int],
    *,
    syntax_hints: bool,
    surface: bool,
) -> Phase1SituationTemplate:
    computed_background = [sit_surface]
    computed_background.extend(make_noise_objects(noise_objects))
    return Phase1SituationTemplate(
        f"{agent.handle}-sit-intransitive",
        salient_object_variables=[agent],
        background_object_variables=computed_background,
        actions=[
            Action(
                SIT,
                argument_roles_to_fillers=[(AGENT, agent)],
                auxiliary_variable_bindings=[
                    (
                        SIT_GOAL,
                        Region(
                            sit_surface,
                            direction=GRAVITATIONAL_UP,
                            distance=EXTERIOR_BUT_IN_CONTACT,
                        ),
                    ),
                    (SIT_THING_SAT_ON, sit_surface),
                ],
            )
        ],
        constraining_relations=[bigger_than(sit_surface, agent)] if surface else [],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER] if syntax_hints else [],
    )


def make_sit_transitive(
    agent: TemplateObjectVariable,
    sit_surface: TemplateObjectVariable,
    noise_objects: Optional[int],
    *,
    syntax_hints: bool,
    surface: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-sit-on-{sit_surface.handle}-transitive",
        salient_object_variables=[agent, sit_surface],
        background_object_variables=make_noise_objects(noise_objects),
        actions=[
            Action(
                SIT,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (
                        GOAL,
                        Region(
                            sit_surface,
                            direction=GRAVITATIONAL_UP,
                            distance=EXTERIOR_BUT_IN_CONTACT,
                        ),
                    ),
                ],
                auxiliary_variable_bindings=[(SIT_THING_SAT_ON, sit_surface)],
            )
        ],
        constraining_relations=[bigger_than(sit_surface, agent)] if surface else [],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER] if syntax_hints else [],
    )


def make_sit_templates(
    noise_objects: Optional[int],
    banned_ontology_types: Iterable[OntologyNode] = immutableset(),
) -> Iterable[Phase1SituationTemplate]:
    sitter = standard_object(
        "sitter_0",
        THING,
        required_properties=[ANIMATE],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
        banned_ontology_types=banned_ontology_types,
    )

    sit_surface = standard_object(
        "surface",
        THING,
        required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM],
        banned_properties=[GROUND],
        banned_ontology_types=banned_ontology_types,
    )
    seat = standard_object(
        "sitting-surface",
        INANIMATE_OBJECT,
        required_properties=[CAN_BE_SAT_ON_BY_PEOPLE],
        banned_ontology_types=banned_ontology_types,
    )

    # we need two groups of templates because in general something can sit on
    # anything bigger than itself which has a surface,
    # but people also sit in chairs, etc., which are smaller than them.
    for adverbial_mod in [True, False]:
        for surface in [True, False]:
            yield make_sit_template_intransitive(
                sitter,
                sit_surface if surface else seat,
                noise_objects,
                syntax_hints=adverbial_mod,
                surface=surface,
            )
            yield make_sit_transitive(
                sitter,
                sit_surface if surface else seat,
                noise_objects,
                syntax_hints=adverbial_mod,
                surface=surface,
            )


def _make_sit_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
    banned_ontology_types: Iterable[OntologyNode] = immutableset(),
) -> Phase1InstanceGroup:
    return phase1_instances(
        "sitting",
        chain(
            *[
                sampled(
                    situation_templates,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    max_to_sample=num_samples,
                    block_multiple_of_the_same_type=True,
                )
                if num_samples
                else all_possible(
                    situation_templates,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                )
                for situation_templates in make_sit_templates(
                    noise_objects, banned_ontology_types=banned_ontology_types
                )
            ]
        ),
        language_generator=language_generator,
    )


def make_take_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    *,
    use_adverbial_path_modifier: bool,
    spatial_properties: Iterable[OntologyNode] = None,
    operator: PathOperator = None,
    background: Iterable[TemplateObjectVariable] = immutableset(),
) -> Phase1SituationTemplate:
    # X grabs Y
    ground = GROUND_OBJECT_TEMPLATE
    return Phase1SituationTemplate(
        f"{agent.handle}-take-{theme.handle}",
        salient_object_variables=[agent, theme],
        background_object_variables=background,
        actions=[
            Action(
                TAKE,
                argument_roles_to_fillers=[(AGENT, agent), (THEME, theme)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            agent,
                            SpatialPath(
                                # this is a hack since "grab" with an adverb doesn't really work in English
                                operator=operator
                                if (
                                    not spatial_properties
                                    or HARD_FORCE not in spatial_properties
                                )
                                else None,
                                reference_source_object=Region(ground, distance=DISTAL)
                                if (operator and operator == TOWARD)
                                else ground,
                                reference_destination_object=Region(
                                    ground, distance=DISTAL
                                )
                                if (operator and operator == AWAY_FROM)
                                else ground,
                                properties=spatial_properties,
                            ),
                        )
                    ]
                ),
            )
        ],
        # this is a hack since "grab" with an adverb doesn't really work in English
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER]
        if (
            use_adverbial_path_modifier
            and (not spatial_properties or HARD_FORCE not in spatial_properties)
        )
        else [],
    )


def make_walk_run_template(
    agent: TemplateObjectVariable,
    *,
    use_adverbial_path_modifier: bool,
    operator: PathOperator = None,
    spatial_properties: Iterable[OntologyNode] = None,
    background: Iterable[TemplateObjectVariable] = immutableset(),
) -> Phase1SituationTemplate:
    # X walks
    goal = standard_object("goal", THING, required_properties=[INANIMATE])
    ground = GROUND_OBJECT_TEMPLATE
    computed_background = [ground]
    computed_background.extend(background)
    return Phase1SituationTemplate(
        f"{agent.handle} walk",
        salient_object_variables=[agent],
        background_object_variables=computed_background,
        actions=[
            Action(
                WALK,
                auxiliary_variable_bindings=[(WALK_SURFACE_AUXILIARY, ground)],
                argument_roles_to_fillers=[(AGENT, agent)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            agent,
                            SpatialPath(
                                operator=TOWARD,
                                reference_source_object=Region(goal, distance=DISTAL),
                                reference_destination_object=goal,
                                properties=spatial_properties,
                            ),
                        ),
                        (
                            agent,
                            SpatialPath(
                                operator=operator
                                if use_adverbial_path_modifier
                                else None,
                                reference_source_object=Region(ground, distance=DISTAL)
                                if (operator and operator == TOWARD)
                                else ground,
                                reference_destination_object=Region(
                                    ground, distance=DISTAL
                                )
                                if (operator and operator == AWAY_FROM)
                                else ground,
                                properties=spatial_properties,
                            ),
                        ),
                    ]
                ),
            )
        ],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER] if use_adverbial_path_modifier else [],
    )


def _make_take_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    return phase1_instances(
        "taking",
        chain(
            flatten(
                [
                    sampled(
                        make_take_template(
                            agent=standard_object(
                                "taker_0",
                                THING,
                                required_properties=[ANIMATE],
                                banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
                            ),
                            theme=standard_object(
                                "object_taken_0", required_properties=[INANIMATE]
                            ),
                            use_adverbial_path_modifier=use_adverbial_path_modifier,
                            operator=operator if use_adverbial_path_modifier else None,
                            background=make_noise_objects(noise_objects),
                        ),
                        block_multiple_of_the_same_type=True,
                        max_to_sample=num_samples if num_samples else 25,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                    )
                    for use_adverbial_path_modifier in [True, False]
                    for operator in [TOWARD, AWAY_FROM]
                ]
            )
        ),
        language_generator=language_generator,
    )


def bare_move_template(
    agent: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    *,
    spatial_properties: Iterable[OntologyNode] = immutableset(),
    background: Iterable[TemplateObjectVariable] = immutableset(),
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        "bare-move",
        salient_object_variables=[agent],
        background_object_variables=background,
        actions=[
            Action(
                MOVE,
                argument_roles_to_fillers=[(AGENT, agent)],
                auxiliary_variable_bindings=[(MOVE_GOAL, goal_reference)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            agent,
                            SpatialPath(
                                TO,
                                reference_source_object=Region(
                                    goal_reference, distance=DISTAL
                                ),
                                reference_destination_object=goal_reference,
                                properties=spatial_properties,
                            ),
                        )
                    ]
                ),
            )
        ],
    )


def transitive_move_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    *,
    spatial_properties: Iterable[OntologyNode] = immutableset(),
    background: Iterable[TemplateObjectVariable] = immutableset(),
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        "transitive-move",
        salient_object_variables=[agent, theme],
        background_object_variables=background,
        actions=[
            Action(
                MOVE,
                argument_roles_to_fillers=[(AGENT, agent), (THEME, theme)],
                auxiliary_variable_bindings=[(MOVE_GOAL, goal_reference)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            theme,
                            SpatialPath(
                                TO,
                                reference_source_object=Region(
                                    goal_reference, distance=DISTAL
                                ),
                                reference_destination_object=goal_reference,
                                properties=spatial_properties,
                            ),
                        )
                    ]
                ),
            )
        ],
        constraining_relations=[bigger_than(agent, theme)],
    )


def make_move_templates(
    noise_objects: Optional[int]
) -> Iterable[Phase1SituationTemplate]:
    self_mover_0 = standard_object(
        "self-mover_0",
        THING,
        required_properties=[SELF_MOVING],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )

    other_mover_0 = standard_object(
        "mover_0",
        THING,
        required_properties=[ANIMATE],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )
    movee_0 = standard_object("movee_0", THING, required_properties=[INANIMATE])
    move_goal_reference = standard_object(
        "move-goal-reference", THING, required_properties=[INANIMATE]
    )
    background = make_noise_objects(noise_objects)

    return [
        # bare move (e.g. "a box moves") is about half of uses in child speed
        bare_move_template(self_mover_0, move_goal_reference, background=background),
        # Transitive Move
        transitive_move_template(
            other_mover_0, movee_0, move_goal_reference, background=background
        ),
    ]


def _make_move_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    return phase1_instances(
        "move",
        chain(
            *[
                sampled(
                    situation,
                    max_to_sample=num_samples if num_samples else 25,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    block_multiple_of_the_same_type=True,
                )
                for situation in make_move_templates(noise_objects)
            ]
        ),
        language_generator=language_generator,
    )


def make_spin_templates(
    noise_objects: Optional[int]
) -> Iterable[Phase1SituationTemplate]:
    self_turner = standard_object(
        "self-spinner_0",
        THING,
        required_properties=[ANIMATE],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )

    other_spinner = standard_object(
        "spinner_0",
        THING,
        required_properties=[ANIMATE],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )
    spinee = standard_object("spinee_0", THING, required_properties=[INANIMATE])
    background = make_noise_objects(noise_objects)

    bare_spin_template = Phase1SituationTemplate(
        "bare-spin",
        salient_object_variables=[self_turner],
        background_object_variables=background,
        actions=[Action(SPIN, argument_roles_to_fillers=[(AGENT, self_turner)])],
    )

    transitive_spin_template = Phase1SituationTemplate(
        "transitive-spin",
        salient_object_variables=[other_spinner, spinee],
        background_object_variables=background,
        actions=[
            Action(
                SPIN, argument_roles_to_fillers=[(AGENT, other_spinner), (THEME, spinee)]
            )
        ],
        constraining_relations=[bigger_than(other_spinner, spinee)],
    )
    return [bare_spin_template, transitive_spin_template]


def _make_spin_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    return phase1_instances(
        "spin",
        chain(
            *[
                sampled(
                    situation,
                    max_to_sample=num_samples if num_samples else 25,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    block_multiple_of_the_same_type=True,
                )
                for situation in make_spin_templates(noise_objects)
            ]
        ),
        language_generator=language_generator,
    )


def make_go_templates(noise_objects: Optional[int]) -> Iterable[Phase1SituationTemplate]:
    goer = standard_object(
        "goer",
        THING,
        required_properties=[ANIMATE],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )
    goal_reference = standard_object(
        "go-goal", THING, banned_properties=[IS_SPEAKER, IS_ADDRESSEE]
    )
    in_goal_reference = standard_object(
        "go-in-goal", THING, required_properties=[HOLLOW], banned_properties=[ANIMATE]
    )
    background = make_noise_objects(noise_objects)

    go_to = _go_to_template(goer, goal_reference, background)
    go_in = _go_in_template(goer, in_goal_reference, background)
    return [go_to, go_in]


def _make_go_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    goer = standard_object(
        "goer",
        THING,
        required_properties=[ANIMATE],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )
    under_goal_reference = standard_object(
        "go-under-goal",
        THING,
        required_properties=[HAS_SPACE_UNDER],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )

    return phase1_instances(
        "go",
        chain(
            flatten(
                [
                    sampled(
                        situation,
                        max_to_sample=num_samples if num_samples else 25,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        block_multiple_of_the_same_type=True,
                    )
                    for situation in make_go_templates(noise_objects)
                ]
            ),
            flatten(
                [
                    sampled(
                        _go_under_template(
                            goer,
                            under_goal_reference,
                            make_noise_objects(noise_objects),
                            is_distal=is_distal,
                        ),
                        max_to_sample=num_samples if num_samples else 25,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        block_multiple_of_the_same_type=True,
                    )
                    for is_distal in (True, False)
                ]
            ),
        ),
        language_generator=language_generator,
    )


def make_push_templates(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    push_surface: TemplateObjectVariable,
    push_goal: TemplateObjectVariable,
    *,
    operator: PathOperator = None,
    use_adverbial_path_modifier: bool,
    spatial_properties: Iterable[OntologyNode] = immutableset(),
    background: Iterable[TemplateObjectVariable] = immutableset(),
) -> List[Phase1SituationTemplate]:
    # push with implicit goal
    aux_bindings = [(PUSH_SURFACE_AUX, push_surface), (PUSH_GOAL, push_goal)]
    push_unexpressed_goal = Phase1SituationTemplate(
        "push-unexpressed-surface-goal",
        salient_object_variables=[agent, theme],
        background_object_variables=background,
        actions=[
            Action(
                PUSH,
                argument_roles_to_fillers=[(AGENT, agent), (THEME, theme)],
                auxiliary_variable_bindings=aux_bindings,
                during=DuringAction(
                    continuously=[on(theme, push_surface)],
                    objects_to_paths=[
                        (
                            agent,
                            SpatialPath(
                                TO,
                                reference_source_object=Region(
                                    push_goal, distance=DISTAL
                                ),
                                reference_destination_object=push_goal,
                                properties=spatial_properties,
                            ),
                        ),
                        (
                            agent,
                            SpatialPath(
                                operator=operator,
                                reference_source_object=Region(
                                    GROUND_OBJECT_TEMPLATE, distance=DISTAL
                                )
                                if (operator and operator == TOWARD)
                                else GROUND_OBJECT_TEMPLATE,
                                reference_destination_object=Region(
                                    GROUND_OBJECT_TEMPLATE, distance=DISTAL
                                )
                                if (operator and operator == AWAY_FROM)
                                else GROUND_OBJECT_TEMPLATE,
                                properties=spatial_properties,
                            ),
                        ),
                    ],
                ),
            )
        ],
        constraining_relations=[bigger_than(push_surface, theme)],
        asserted_always_relations=[on(theme, push_surface)],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER] if use_adverbial_path_modifier else [],
    )

    # push with implicit goal
    push_unexpressed_goal_expressed_surface = Phase1SituationTemplate(
        "push-unexpressed-goal",
        salient_object_variables=[agent, theme, push_surface],
        background_object_variables=background,
        actions=[
            Action(
                PUSH,
                argument_roles_to_fillers=[(AGENT, agent), (THEME, theme)],
                auxiliary_variable_bindings=aux_bindings,
                during=DuringAction(
                    continuously=[on(theme, push_surface)],
                    objects_to_paths=[
                        (
                            agent,
                            SpatialPath(
                                TO,
                                reference_source_object=Region(
                                    push_goal, distance=DISTAL
                                ),
                                reference_destination_object=push_goal,
                                properties=spatial_properties,
                            ),
                        ),
                        (
                            agent,
                            SpatialPath(
                                operator=operator,
                                reference_source_object=Region(
                                    GROUND_OBJECT_TEMPLATE, distance=DISTAL
                                )
                                if (operator and operator == TOWARD)
                                else GROUND_OBJECT_TEMPLATE,
                                reference_destination_object=Region(
                                    GROUND_OBJECT_TEMPLATE, distance=DISTAL
                                )
                                if (operator and operator == AWAY_FROM)
                                else GROUND_OBJECT_TEMPLATE,
                                properties=spatial_properties,
                            ),
                        ),
                    ],
                ),
            )
        ],
        constraining_relations=[bigger_than(push_surface, theme)],
        asserted_always_relations=[on(theme, push_surface)],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER] if use_adverbial_path_modifier else [],
    )
    return [push_unexpressed_goal, push_unexpressed_goal_expressed_surface]


def _make_push_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    return phase1_instances(
        "pushing",
        chain(
            *[
                sampled(
                    situation,
                    max_to_sample=num_samples if num_samples else 25,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    block_multiple_of_the_same_type=True,
                )
                for adverbial_path_modifier in [True, False]
                for operator in [TOWARD, AWAY_FROM]
                for situation in make_push_templates(
                    agent=standard_object(
                        "pusher",
                        THING,
                        required_properties=[ANIMATE],
                        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
                    ),
                    theme=standard_object("pushee", INANIMATE_OBJECT),
                    push_surface=standard_object(
                        "push_surface",
                        THING,
                        required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM],
                    ),
                    push_goal=standard_object("push_goal", INANIMATE_OBJECT),
                    use_adverbial_path_modifier=adverbial_path_modifier,
                    operator=operator if adverbial_path_modifier else None,
                    background=make_noise_objects(noise_objects),
                )
            ]
        ),
        language_generator=language_generator,
    )


def throw_on_ground_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    *,
    spatial_properties: Iterable[OntologyNode] = immutableset(),
    background: Iterable[TemplateObjectVariable] = immutableset(),
) -> Phase1SituationTemplate:
    ground = GROUND_OBJECT_TEMPLATE
    return Phase1SituationTemplate(
        "throw-on-ground",
        salient_object_variables=[agent, theme, ground],
        background_object_variables=background,
        actions=[
            Action(
                THROW,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (THEME, theme),
                    (
                        GOAL,
                        Region(
                            ground,
                            distance=EXTERIOR_BUT_IN_CONTACT,
                            direction=GRAVITATIONAL_UP,
                        ),
                    ),
                ],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            theme,
                            SpatialPath(
                                None,
                                reference_source_object=agent,
                                reference_destination_object=Region(
                                    ground,
                                    distance=EXTERIOR_BUT_IN_CONTACT,
                                    direction=GRAVITATIONAL_UP,
                                ),
                                properties=spatial_properties,
                            ),
                        )
                    ]
                ),
            )
        ],
        after_action_relations=[on(theme, ground)],
    )


def throw_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal: TemplateObjectVariable,
    *,
    spatial_properties: Iterable[OntologyNode] = immutableset(),
    background: Iterable[TemplateObjectVariable] = immutableset(),
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        "throw",
        salient_object_variables=[agent, theme],
        background_object_variables=background,
        actions=[
            Action(
                THROW,
                argument_roles_to_fillers=[(AGENT, agent), (THEME, theme)],
                auxiliary_variable_bindings=[
                    (THROW_GOAL, Region(goal, distance=PROXIMAL))
                ],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            theme,
                            SpatialPath(
                                None,
                                reference_source_object=agent,
                                reference_destination_object=goal,
                                properties=spatial_properties,
                            ),
                        )
                    ]
                ),
            )
        ],
        constraining_relations=[bigger_than(agent, theme)],
    )


def throw_up_down_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal: TemplateObjectVariable,
    *,
    is_up: bool,
    spatial_properties: Iterable[OntologyNode] = immutableset(),
    background: Iterable[TemplateObjectVariable] = immutableset(),
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-throws-{theme.handle}-up-down",
        salient_object_variables=[agent, theme],
        background_object_variables=background,
        actions=[
            Action(
                THROW,
                argument_roles_to_fillers=[(AGENT, agent), (THEME, theme)],
                auxiliary_variable_bindings=[
                    (THROW_GOAL, Region(goal, distance=PROXIMAL))
                ],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            theme,
                            SpatialPath(
                                AWAY_FROM,
                                reference_source_object=GROUND_OBJECT_TEMPLATE,
                                reference_destination_object=Region(
                                    GROUND_OBJECT_TEMPLATE, distance=DISTAL
                                ),
                                properties=spatial_properties,
                            ),
                        )
                    ]
                    if is_up
                    else [
                        (
                            theme,
                            SpatialPath(
                                TOWARD,
                                reference_source_object=Region(
                                    GROUND_OBJECT_TEMPLATE, distance=DISTAL
                                ),
                                reference_destination_object=GROUND_OBJECT_TEMPLATE,
                                properties=spatial_properties,
                            ),
                        )
                    ]
                ),
            )
        ],
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER],
    )


def throw_to_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal: TemplateObjectVariable,
    *,
    spatial_properties: Iterable[OntologyNode] = None,
    background: Iterable[TemplateObjectVariable] = immutableset(),
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        "throw-to",
        salient_object_variables=[agent, theme, goal],
        background_object_variables=background,
        actions=[
            Action(
                THROW,
                argument_roles_to_fillers=[(AGENT, agent), (THEME, theme), (GOAL, goal)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            theme,
                            SpatialPath(
                                None,
                                reference_source_object=agent,
                                reference_destination_object=goal,
                                properties=spatial_properties,
                            ),
                        )
                    ]
                ),
            )
        ],
        after_action_relations=[near(theme, goal)],
    )


def throw_to_region_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal: TemplateObjectVariable,
    *,
    spatial_properties: Iterable[OntologyNode] = None,
    background: Iterable[TemplateObjectVariable] = immutableset(),
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        "throw-to",
        salient_object_variables=[agent, theme, goal],
        background_object_variables=background,
        actions=[
            Action(
                THROW,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (THEME, theme),
                    (GOAL, Region(goal, distance=PROXIMAL)),
                ],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            theme,
                            SpatialPath(
                                None,
                                reference_source_object=agent,
                                reference_destination_object=theme,
                                properties=spatial_properties,
                            ),
                        )
                    ]
                ),
            )
        ],
        after_action_relations=[near(theme, goal)],
        constraining_relations=[bigger_than(agent, theme)],
    )


# for testing gei vs dao X shang in Chinese
def make_throw_animacy_templates(
    noise_objects: Optional[int]
) -> Iterable[Phase1SituationTemplate]:
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
    goal_reference = standard_object("object_1", required_properties=[INANIMATE])
    background = make_noise_objects(noise_objects)

    return [
        # Throw to an animate catcher
        throw_to_template(thrower, object_thrown, catcher, background=background),
        # Throw to an inanimate object
        throw_to_region_template(
            thrower, object_thrown, goal_reference, background=background
        ),
    ]


def make_throw_templates(
    noise_objects: Optional[int]
) -> Iterable[Phase1SituationTemplate]:
    thrower = standard_object(
        "thrower_0",
        THING,
        required_properties=[ANIMATE],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )

    object_thrown = standard_object("object_0", required_properties=[INANIMATE])
    implicit_goal_reference = standard_object(
        "implicit_throw_goal_object", required_properties=[INANIMATE]
    )
    background = make_noise_objects(noise_objects)

    return [
        # # Dad throws a cookie on the ground -- disabled because we learn this in verbs with dynamic prepositions
        # throw_on_ground_template(thrower, object_thrown, background=background),
        # A baby throws a truck
        throw_template(
            thrower, object_thrown, implicit_goal_reference, background=background
        ),
        # Throw up
        throw_up_down_template(
            thrower,
            object_thrown,
            implicit_goal_reference,
            is_up=True,
            background=background,
        ),
        # Throw down
        throw_up_down_template(
            thrower,
            object_thrown,
            implicit_goal_reference,
            is_up=False,
            background=background,
        ),
        # Throw To -- disabled because we learn this in verbs with dynamic prepositions
        # throw_to_template(thrower, object_thrown, catcher, background=background),
    ]


def _make_throw_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    return phase1_instances(
        "throwing",
        chain(
            flatten(
                [
                    sampled(
                        template,
                        max_to_sample=num_samples if num_samples else 25,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        block_multiple_of_the_same_type=True,
                    )
                    for template in make_throw_templates(noise_objects)
                ]
            )
        ),
        language_generator=language_generator,
    )


def _make_pass_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    return phase1_instances(
        "passing",
        chain(
            flatten(
                [
                    sampled(
                        make_pass_template(
                            agent=standard_object(
                                "thrower_0",
                                THING,
                                required_properties=[ANIMATE],
                                banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
                            ),
                            theme=standard_object(
                                "object_0", required_properties=[INANIMATE]
                            ),
                            goal=standard_object(
                                "catcher_0",
                                THING,
                                required_properties=[ANIMATE],
                                banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
                            ),
                            use_adverbial_path_modifier=use_adverbial_path_modifier,
                            operator=operator if use_adverbial_path_modifier else None,
                            background=make_noise_objects(noise_objects),
                        ),
                        block_multiple_of_the_same_type=True,
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        max_to_sample=num_samples if num_samples else 25,
                    )
                    for use_adverbial_path_modifier in (True, False)
                    for operator in [TOWARD, AWAY_FROM]
                ]
            )
        ),
        language_generator=language_generator,
    )


def _make_come_down_template(
    agent: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    speaker: TemplateObjectVariable,
    ground: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    background_objects_mutable = [speaker, ground]
    background_objects_mutable.extend(background)
    background_objects = immutableset(background_objects_mutable)
    return Phase1SituationTemplate(
        f"{agent.handle}-come-to-{goal_reference.handle}",
        salient_object_variables=[agent, goal_reference],
        background_object_variables=background_objects,
        actions=[
            Action(
                COME,
                argument_roles_to_fillers=[(AGENT, agent), (GOAL, goal_reference)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            agent,
                            SpatialPath(
                                TOWARD,
                                reference_source_object=Region(ground, distance=DISTAL),
                                reference_destination_object=ground,
                            ),
                        ),
                        (
                            agent,
                            SpatialPath(
                                TO,
                                reference_source_object=Region(
                                    goal_reference, distance=DISTAL
                                ),
                                reference_destination_object=Region(
                                    goal_reference, distance=PROXIMAL
                                ),
                            ),
                        ),
                    ]
                ),
            )
        ],
        before_action_relations=[negate(contacts(agent, ground))],
        asserted_always_relations=flatten_relations(near(speaker, goal_reference)),
        syntax_hints=[USE_ADVERBIAL_PATH_MODIFIER],
    )


def _make_come_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    movee = standard_object(
        "movee",
        THING,
        required_properties=[ANIMATE],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )
    learner = standard_object("leaner_0", LEARNER)
    speaker = standard_object(
        "speaker",
        PERSON,
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
        added_properties=[IS_SPEAKER],
    )
    object_ = standard_object(
        "object_0", THING, banned_properties=[IS_SPEAKER, IS_ADDRESSEE]
    )
    ground = standard_object("ground", root_node=GROUND)
    background = make_noise_objects(noise_objects)

    come_to_speaker = Phase1SituationTemplate(
        "come-to-speaker",
        salient_object_variables=[movee, speaker],
        background_object_variables=background,
        actions=[
            Action(COME, argument_roles_to_fillers=[(AGENT, movee), (GOAL, speaker)])
        ],
    )

    come_to_learner = Phase1SituationTemplate(
        "come-to-leaner",
        salient_object_variables=[movee],
        background_object_variables=background,
        actions=[
            Action(COME, argument_roles_to_fillers=[(AGENT, movee), (GOAL, learner)])
        ],
    )

    come_to_object = Phase1SituationTemplate(
        "come-to-object",
        salient_object_variables=[movee, object_],
        background_object_variables=background,
        actions=[
            Action(COME, argument_roles_to_fillers=[(AGENT, movee), (GOAL, object_)])
        ],
    )

    return phase1_instances(
        "come",
        chain(
            *[
                sampled(
                    come_to_speaker,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples,
                    block_multiple_of_the_same_type=True,
                )
                if num_samples
                else all_possible(
                    come_to_speaker,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                ),
                sampled(
                    come_to_learner,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples,
                    block_multiple_of_the_same_type=True,
                )
                if num_samples
                else all_possible(
                    come_to_learner,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                ),
                sampled(
                    come_to_object,
                    max_to_sample=num_samples if num_samples else 25,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    block_multiple_of_the_same_type=True,
                ),
                sampled(
                    _make_come_down_template(movee, object_, speaker, ground, background),
                    max_to_sample=num_samples if num_samples else 25,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    block_multiple_of_the_same_type=True,
                ),
            ]
        ),
        language_generator=language_generator,
    )


def _make_behind_in_front_curriculum(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    front_behind_ground_object = standard_object("ground_object")
    front_behind_figure_object = standard_object("figure_object")
    front_behind_speaker = standard_object(
        "speaker_0", PERSON, added_properties=[IS_SPEAKER]
    )
    front_behind_addressee = standard_object(
        "addressee_0", PERSON, added_properties=[IS_ADDRESSEE]
    )
    background = make_noise_objects(noise_objects)

    def make_behind_in_front_templates() -> Iterable[Phase1SituationTemplate]:
        for in_front_of in (True, False):
            for distal in (True, False):
                suffix = "-in-front" if in_front_of else "-behind"
                direction = Direction(
                    positive=in_front_of,
                    relative_to_axis=FacingAddresseeAxis(front_behind_ground_object),
                )
                yield Phase1SituationTemplate(
                    f"front_behind_addressee-relative-{suffix}",
                    salient_object_variables=[
                        front_behind_figure_object,
                        front_behind_ground_object,
                    ],
                    background_object_variables=chain(
                        [front_behind_speaker, front_behind_addressee], background
                    ),
                    asserted_always_relations=[
                        near(
                            front_behind_figure_object,
                            front_behind_ground_object,
                            direction=direction,
                        )
                        if distal
                        else far(
                            front_behind_figure_object,
                            front_behind_ground_object,
                            direction=direction,
                        )
                    ],
                    constraining_relations=[
                        bigger_than(
                            front_behind_ground_object, front_behind_figure_object
                        )
                    ],
                )

    return phase1_instances(
        "behind_in_front_curriculum",
        chain(
            *[
                flatten(
                    sampled(
                        template,
                        max_to_sample=num_samples if num_samples else 25,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        block_multiple_of_the_same_type=True,
                    )
                    for template in make_behind_in_front_templates()
                )
            ]
        ),
        language_generator=language_generator,
    )


def build_gaila_phase1_object_curriculum(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Sequence[Phase1InstanceGroup]:
    """
    One particular instantiation of the object-learning parts of the curriculum for GAILA Phase 1.
    """
    return [
        _make_each_object_by_itself_curriculum(
            num_samples, num_noise_objects, language_generator
        ),
        #     We are deferring handling numeric quantifiers until Phase 2,
        #     so this curriculum is not actually executed in Phase 1.
        # _make_multiple_objects_curriculum(num_sampled, num_noise_objects, language_generator),
        _make_object_on_ground_curriculum(
            num_samples, num_noise_objects, language_generator
        ),
    ]


def build_gaila_plurals_curriculum(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Sequence[Phase1InstanceGroup]:
    return [
        _make_plural_objects_curriculum(
            num_samples, num_noise_objects, language_generator
        )
    ]


def build_gaila_generics_curriculum(
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


def build_gaila_phase1_attribute_curriculum(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Sequence[Phase1InstanceGroup]:
    """
    One particular instantiation of the object-learning parts of the curriculum for GAILA Phase 1.
    """
    return [
        _make_objects_with_colors_curriculum(
            num_samples, num_noise_objects, language_generator
        ),
        # TODO: Enable this curriculum once we handle it better
        # See: https://github.com/isi-vista/adam/issues/830
        # _make_objects_with_colors_is_curriculum(num_samples, num_noise_objects, language_generator),
        _make_my_your_object_curriculum(
            num_samples, num_noise_objects, language_generator
        ),
    ]


def build_classifier_curriculum(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Sequence[Phase1InstanceGroup]:
    """One particular instantiation of the Chinese classifier learning curriculum"""
    return [
        _make_chinese_classifier_single_object_curriculum(
            num_samples, num_noise_objects, language_generator
        )
    ]


def build_gaila_phase1_relation_curriculum(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Sequence[Phase1InstanceGroup]:
    """
    One particular instantiation of the object-learning parts of the curriculum for GAILA Phase 1.
    """
    return [
        _make_person_has_object_curriculum(
            num_samples, num_noise_objects, language_generator
        ),
        _make_object_on_object_curriculum(
            num_samples, num_noise_objects, language_generator
        ),
        _make_object_beside_object_curriculum(
            num_samples, num_noise_objects, language_generator
        ),
        _make_object_under_or_over_object_curriculum(
            num_samples, num_noise_objects, language_generator
        ),
        _make_object_in_other_object_curriculum(
            num_samples, num_noise_objects, language_generator
        ),
        _make_behind_in_front_curriculum(
            num_samples, num_noise_objects, language_generator
        ),
    ]


def build_gaila_phase1_verb_curriculum(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Sequence[Phase1InstanceGroup]:
    """
    One particular instantiation of the object-learning parts of the curriculum for GAILA Phase 1.
    """
    return [
        _make_fall_curriculum(num_samples, num_noise_objects, language_generator),
        _make_transfer_of_possession_curriculum(
            num_samples, num_noise_objects, language_generator
        ),
        _make_fly_curriculum(num_samples, num_noise_objects, language_generator),
        _make_roll_curriculum(num_samples, num_noise_objects, language_generator),
        # TODO: see https://github.com/isi-vista/adam/issues/937
        # _make_speaker_addressee_curriculum(
        #     num_samples, num_noise_objects, language_generator
        # ),
        _make_jump_curriculum(num_samples, num_noise_objects, language_generator),
        _make_drink_curriculum(num_samples, num_noise_objects, language_generator),
        _make_sit_curriculum(num_samples, num_noise_objects, language_generator),
        _make_put_curriculum(num_samples, num_noise_objects, language_generator),
        _make_eat_curriculum(num_samples, num_noise_objects, language_generator),
        _make_take_curriculum(num_samples, num_noise_objects, language_generator),
        _make_move_curriculum(num_samples, num_noise_objects, language_generator),
        _make_spin_curriculum(num_samples, num_noise_objects, language_generator),
        _make_go_curriculum(num_samples, num_noise_objects, language_generator),
        _make_push_curriculum(num_samples, num_noise_objects, language_generator),
        # TODO: fix this based on Deniz's thoughts
        _make_throw_curriculum(num_samples, num_noise_objects, language_generator),
        _make_pass_curriculum(num_samples, num_noise_objects, language_generator),
        # _make_put_on_speaker_addressee_body_part_curriculum(num_samples, num_noise_objects, language_generator),
        _make_come_curriculum(num_samples, num_noise_objects, language_generator),
    ]


def build_gaila_phase_1_curriculum(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Sequence[Phase1InstanceGroup]:
    """
    One particular instantiation of the curriculum for GAILA Phase 1.
    """
    return list(
        chain(
            build_gaila_phase1_object_curriculum(
                num_samples, num_noise_objects, language_generator
            ),
            build_gaila_phase1_attribute_curriculum(
                num_samples, num_noise_objects, language_generator
            ),
            build_gaila_phase1_relation_curriculum(
                num_samples, num_noise_objects, language_generator
            ),
            build_gaila_phase1_verb_curriculum(
                num_samples, num_noise_objects, language_generator
            ),
        )
    )
