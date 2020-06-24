"""
Additions for the Curricula for DARPA GAILA Phase 2
"""

from itertools import chain
from typing import Sequence

from immutablecollections import immutableset

from adam.curriculum.curriculum_utils import (
    PHASE1_CHOOSER_FACTORY,
    Phase1InstanceGroup,
    standard_object,
    phase2_instances,
    phase1_instances)
from adam.curriculum.imprecise_descriptions_curriculum import make_imprecise_temporal_descriptions, \
    make_imprecise_size_curriculum
from adam.curriculum.phase1_curriculum import _make_plural_objects_curriculum, _make_pass_curriculum, \
    _make_generic_statements_curriculum, _make_part_whole_curriculum, _make_transitive_roll_curriculum
from adam.curriculum.preposition_curriculum import make_prepositions_curriculum
from adam.curriculum.verbs_with_dynamic_prepositions_curriculum import make_verb_with_dynamic_prepositions_curriculum
from adam.language_specific.english.english_language_generator import IGNORE_COLORS
from adam.ontology import THING
from adam.ontology.phase1_ontology import CHAIR, CUP, ANIMATE, INANIMATE_OBJECT, HOLLOW, GAILA_PHASE_1_ONTOLOGY
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
    HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator,
    GAILA_PHASE_2_PERCEPTION_GENERATOR,
)
from adam.situation.templates.phase1_situation_templates import _put_in_template
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    all_possible,
    color_variable,
    sampled)


def _make_chairs_curriculum(
    perception_generator: HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator = GAILA_PHASE_2_PERCEPTION_GENERATOR
) -> Phase1InstanceGroup:
    color = color_variable("color")
    chair_templates = [
        Phase1SituationTemplate(
            "chair-object",
            salient_object_variables=[
                standard_object("chair", chair_type, added_properties=[color])
            ],
            syntax_hints=[IGNORE_COLORS],
        )
        for chair_type in [CHAIR, CHAIR_2, CHAIR_3, CHAIR_4, CHAIR_5]
    ]

    return phase2_instances(
        "each chair by itself",
        chain(
            *[
                all_possible(
                    chair_template,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_2_ONTOLOGY,
                )
                for chair_template in chair_templates
            ]
        ),
        perception_generator=perception_generator,
    )


def _make_cups_curriculum(
    perception_generator: HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator = GAILA_PHASE_2_PERCEPTION_GENERATOR
) -> Phase1InstanceGroup:
    color = color_variable("color")
    cup_templates = [
        Phase1SituationTemplate(
            "cup-object",
            salient_object_variables=[
                standard_object("cup", cup, added_properties=[color])
            ],
            syntax_hints=[IGNORE_COLORS],
        )
        for cup in [CUP, CUP_2, CUP_3, CUP_4]
    ]

    return phase2_instances(
        "each cup by itself",
        chain(
            *[
                all_possible(
                    cup_template,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_2_ONTOLOGY,
                )
                for cup_template in cup_templates
            ]
        ),
        perception_generator=perception_generator,
    )


def _make_put_in_curriculum() -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[ANIMATE])
    theme = standard_object("theme", INANIMATE_OBJECT)
    goal_in = standard_object("goal_in", INANIMATE_OBJECT, required_properties=[HOLLOW])

    return phase1_instances(
        "Capabilities - Put in",
        sampled(
            _put_in_template(agent, theme, goal_in, None),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            chooser=PHASE1_CHOOSER_FACTORY(),
            max_to_sample=20,
        )
    )


def build_gaila_m8_curriculum() -> Sequence[Phase1InstanceGroup]:
    return list(chain([
        _make_plural_objects_curriculum(), # plurals
        _make_chairs_curriculum(), # functionally defined objects
        _make_cups_curriculum(),
        _make_pass_curriculum(), # Subtle verb distinctions
        _make_generic_statements_curriculum(), # Generics
        _make_part_whole_curriculum(), # Part whole
        _make_transitive_roll_curriculum(), # External Limitations
        _make_put_in_curriculum(),
    ], list(make_imprecise_temporal_descriptions()), # Imprecise descriptions
       make_verb_with_dynamic_prepositions_curriculum(), # Dynamic prepositions
        make_prepositions_curriculum(), # Relative prepositions
        make_imprecise_size_curriculum(),
    ))
