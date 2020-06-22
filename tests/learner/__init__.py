from adam.language_specific.english import ENGLISH_DETERMINERS
from adam.learner.object_recognizer import ObjectRecognizer

from adam.ontology.phase1_ontology import (
    GAILA_PHASE_1_ONTOLOGY,
    PHASE_1_CURRICULUM_OBJECTS,
    GROUND,
)

PHASE_1_CURRICULUM_OBJECTS_INCLUDE_GROUND = list(PHASE_1_CURRICULUM_OBJECTS)
PHASE_1_CURRICULUM_OBJECTS_INCLUDE_GROUND.append(GROUND)

TEST_OBJECT_RECOGNIZER = ObjectRecognizer.for_ontology_types(
    PHASE_1_CURRICULUM_OBJECTS_INCLUDE_GROUND, ENGLISH_DETERMINERS, GAILA_PHASE_1_ONTOLOGY
)
