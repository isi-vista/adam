from adam.language.dependency import LinearizedDependencyTree
from adam.language.language_generator import LanguageGenerator
from adam.language_specific.chinese.chinese_language_generator import (
    GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR,
    GAILA_PHASE_2_CHINESE_LANGUAGE_GENERATOR,
)
from adam.language_specific.english.english_language_generator import (
    GAILA_PHASE_1_LANGUAGE_GENERATOR,
    GAILA_PHASE_2_LANGUAGE_GENERATOR,
    INTEGRATED_EXPERIMENT_LANGUAGE_GENERATOR,
)
from adam.learner import LanguageMode
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation


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
    elif language_mode == LanguageMode.CHINESE:
        return GAILA_PHASE_2_CHINESE_LANGUAGE_GENERATOR
    else:
        raise RuntimeError("Invalid language generator specified")


def integrated_experiment_language_generator(
    language_mode: LanguageMode
) -> LanguageGenerator[HighLevelSemanticsSituation, LinearizedDependencyTree]:
    if language_mode == LanguageMode.ENGLISH:
        return INTEGRATED_EXPERIMENT_LANGUAGE_GENERATOR
    else:
        raise RuntimeError("Invalid language generator specified")
