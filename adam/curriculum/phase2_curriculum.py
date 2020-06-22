"""
Additions for the Curricula for DARPA GAILA Phase 2
"""

from itertools import chain

from adam.curriculum.curriculum_utils import (
    PHASE1_CHOOSER_FACTORY,
    Phase1InstanceGroup,
    phase1_instances,
    standard_object,
    phase2_instances)
from adam.language_specific.english.english_language_generator import (
    IGNORE_COLORS,
)
from adam.ontology.phase1_ontology import (
    GAILA_PHASE_1_ONTOLOGY,
    CHAIR, CUP)
from adam.ontology.phase2_ontology import CHAIR_2, CHAIR_3, CHAIR_4, CHAIR_5, CUP_2, CUP_3, CUP_4, \
    GAILA_PHASE_2_ONTOLOGY
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    GAILA_PHASE_1_PERCEPTION_GENERATOR,
    HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator,
    GAILA_PHASE_2_PERCEPTION_GENERATOR)
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    all_possible,
    color_variable,
)


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
