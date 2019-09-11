"""
Curricula for DARPA GAILA Phase 1
"""
from adam.curriculum import GeneratedFromSituationsInstanceGroup
from adam.language_specific.english.english_language_generator import (
    GAILA_PHASE_1_LANGUAGE_GENERATOR,
)
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    GAILA_PHASE_1_PERCEPTION_GENERATOR,
)
from adam.random_utils import RandomChooser
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    object_variable,
    GAILA_PHASE_1_TEMPLATE_GENERATOR,
)

SINGLE_OBJECT_TEMPLATE = Phase1SituationTemplate(
    object_variables=[object_variable("object")]
)

_CHOOSER = RandomChooser.for_seed(0)

EACH_OBJECT_BY_ITSELF_CURRICULUM = GeneratedFromSituationsInstanceGroup(
    "each object by itself",
    situations=GAILA_PHASE_1_TEMPLATE_GENERATOR.generate_situations(
        SINGLE_OBJECT_TEMPLATE, chooser=_CHOOSER
    ),
    language_generator=GAILA_PHASE_1_LANGUAGE_GENERATOR,
    perception_generator=GAILA_PHASE_1_PERCEPTION_GENERATOR,
    chooser=_CHOOSER,
)

