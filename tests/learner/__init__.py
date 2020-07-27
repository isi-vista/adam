from typing import Mapping
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    GAILA_PHASE_1_PERCEPTION_GENERATOR,
)
from adam.language_specific.english import ENGLISH_DETERMINERS
from adam.learner.language_mode import LanguageMode
from adam.learner.object_recognizer import ObjectRecognizer
from adam.learner.objects import ObjectRecognizerAsTemplateLearner
from adam.ontology.phase1_ontology import (
    GAILA_PHASE_1_ONTOLOGY,
    GROUND,
    PHASE_1_CURRICULUM_OBJECTS,
)
from immutablecollections import immutabledict

PHASE_1_CURRICULUM_OBJECTS_INCLUDE_GROUND = list(PHASE_1_CURRICULUM_OBJECTS)
PHASE_1_CURRICULUM_OBJECTS_INCLUDE_GROUND.append(GROUND)


def object_recognizer_factory(language_mode: LanguageMode) -> ObjectRecognizer:
    return ObjectRecognizer.for_ontology_types(
        PHASE_1_CURRICULUM_OBJECTS_INCLUDE_GROUND,
        ENGLISH_DETERMINERS,
        GAILA_PHASE_1_ONTOLOGY,
        language_mode=language_mode,
        perception_generator=GAILA_PHASE_1_PERCEPTION_GENERATOR,
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
