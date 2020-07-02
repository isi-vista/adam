from adam.language.dependency import LinearizedDependencyTree
from adam.language.language_generator import LanguageGenerator
from adam.language_specific.chinese.chinese_language_generator import (
    GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR,
)
from adam.language_specific.english import ENGLISH_DETERMINERS
from adam.language_specific.english.english_language_generator import (
    GAILA_PHASE_1_LANGUAGE_GENERATOR,
    GAILA_PHASE_2_LANGUAGE_GENERATOR,
)
from adam.learner.language_mode import LanguageMode
from adam.learner.object_recognizer import ObjectRecognizer

from adam.ontology.phase1_ontology import (
    GAILA_PHASE_1_ONTOLOGY,
    PHASE_1_CURRICULUM_OBJECTS,
    GROUND,
)
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation

PHASE_1_CURRICULUM_OBJECTS_INCLUDE_GROUND = list(PHASE_1_CURRICULUM_OBJECTS)
PHASE_1_CURRICULUM_OBJECTS_INCLUDE_GROUND.append(GROUND)

TEST_OBJECT_RECOGNIZER = ObjectRecognizer.for_ontology_types(
    PHASE_1_CURRICULUM_OBJECTS_INCLUDE_GROUND, ENGLISH_DETERMINERS, GAILA_PHASE_1_ONTOLOGY
)


def phase1_language_generator(
    language_mode: LanguageMode
) -> LanguageGenerator[HighLevelSemanticsSituation, LinearizedDependencyTree]:
    if language_mode == LanguageMode.ENGLISH:
        return GAILA_PHASE_1_LANGUAGE_GENERATOR
    elif language_mode == LanguageMode.CHINESE:
        return GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR
    else:
        raise RuntimeError("Invalid language generator specified")


def phase2_language_generator(
    language_mode: LanguageMode
) -> LanguageGenerator[HighLevelSemanticsSituation, LinearizedDependencyTree]:
    if language_mode == LanguageMode.ENGLISH:
        return GAILA_PHASE_2_LANGUAGE_GENERATOR
    # elif language_mode == LanguageMode.CHINESE:
    #    return GAILA_PHASE_2_CHINESE_LANGUAGE_GENERATOR
    else:
        raise RuntimeError("Invalid language generator specified")
