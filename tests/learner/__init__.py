from adam.language_specific.english import ENGLISH_DETERMINERS
from typing import Mapping

from immutablecollections import immutabledict
from adam.language_specific.english import ENGLISH_DETERMINERS
from adam.learner.language_mode import LanguageMode
from adam.learner.object_recognizer import ObjectRecognizer
from adam.learner.objects import ObjectRecognizerAsTemplateLearner

from adam.ontology.phase1_ontology import (
    GAILA_PHASE_1_ONTOLOGY,
    PHASE_1_CURRICULUM_OBJECTS,
    GROUND,
)

PHASE_1_CURRICULUM_OBJECTS_INCLUDE_GROUND = list(PHASE_1_CURRICULUM_OBJECTS)
PHASE_1_CURRICULUM_OBJECTS_INCLUDE_GROUND.append(GROUND)


def object_recognizer_factory(language_mode: LanguageMode) -> ObjectRecognizer:
    return ObjectRecognizer.for_ontology_types(
        PHASE_1_CURRICULUM_OBJECTS_INCLUDE_GROUND,
        ENGLISH_DETERMINERS,
        GAILA_PHASE_1_ONTOLOGY,
        language_mode=language_mode,
    )


LANGUAGE_MODE_TO_OBJECT_RECOGNIZER: Mapping[
    LanguageMode, ObjectRecognizer
] = immutabledict(
    [
        (language_mode, object_recognizer_factory(language_mode))
        for language_mode in LanguageMode
    ]
)

LANGUAGE_MODE_TO_TEMPLATE_LEARNER_OBJECT_RECOGNIZER: Mapping[
    LanguageMode, ObjectRecognizerAsTemplateLearner
] = immutabledict(
    [
        (
            language_mode,
            ObjectRecognizerAsTemplateLearner(
                object_recognizer=LANGUAGE_MODE_TO_OBJECT_RECOGNIZER[language_mode],
                language_mode=language_mode,
            ),
        )
        for language_mode in LanguageMode
    ]
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
