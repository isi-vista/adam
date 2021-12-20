from itertools import chain
from typing_extensions import Protocol

from adam.curriculum.curriculum_utils import (
    Phase3InstanceGroup,
    phase3_instances,
    CHOOSER_FACTORY,
    phase3_standard_object,
)
from adam.language.dependency import LinearizedDependencyTree
from adam.language.language_generator import LanguageGenerator
from adam.language_specific.english.english_language_generator import (
    IGNORE_COLORS,
)
from adam.ontology.phase1_ontology import (
    PHASE_3_M4_STRETCH_CONCEPT,
    PHASE_3_M4_CORE_CONCEPT,
)
from adam.ontology.phase3_ontology import (
    GAILA_PHASE_3_ONTOLOGY,
)
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from vistautils.parameters import Parameters

from adam.situation.templates.phase1_templates import (
    all_possible,
    Phase1SituationTemplate,
)


class Phase3CurriculumCallable(Protocol):
    def __call__(
        self,
        num_samples: int,
        num_noise_objects: int,
        language_generator: LanguageGenerator[
            HighLevelSemanticsSituation, LinearizedDependencyTree
        ],
        params: Parameters,
    ) -> Phase3InstanceGroup:
        raise NotImplementedError(
            "The protocol Phase3CurriculumCallable does not implement a curriculum return."
        )


def phase_3_one_core_objects_curriculum(  # pylint: disable=unused-argument
    num_samples: int,
    num_noise_objects: int,
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
    params: Parameters = Parameters.empty(),
) -> Phase3InstanceGroup:
    return phase3_instances(
        "each-core-object-by-itself",
        chain(
            *[
                all_possible(
                    Phase1SituationTemplate(
                        "single-core-object-phase3",
                        salient_object_variables=[
                            phase3_standard_object(
                                "single-object",
                                required_properties=[PHASE_3_M4_CORE_CONCEPT],
                            )
                        ],
                        syntax_hints=[IGNORE_COLORS],
                    ),
                    chooser=CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_3_ONTOLOGY,
                ),
            ]
        ),
    )


def phase_3_one_stretch_objects_curriculum(  # pylint: disable=unused-argument
    num_samples: int,
    num_noise_objects: int,
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
    params: Parameters = Parameters.empty(),
) -> Phase3InstanceGroup:
    return phase3_instances(
        "each-stretch-object-by-itself",
        chain(
            *[
                all_possible(
                    Phase1SituationTemplate(
                        "stretch-single-object-phase3",
                        salient_object_variables=[
                            phase3_standard_object(
                                "single-object",
                                required_properties=[PHASE_3_M4_STRETCH_CONCEPT],
                            )
                        ],
                        syntax_hints=[IGNORE_COLORS],
                    ),
                    chooser=CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_3_ONTOLOGY,
                ),
            ]
        ),
    )


def phase_3_one_objects_curriculum(  # pylint: disable=unused-argument
    num_samples: int,
    num_noise_objects: int,
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
    params: Parameters = Parameters.empty(),
) -> Phase3InstanceGroup:
    return phase3_instances(
        "each-object-by-itself",
        chain(
            *[
                all_possible(
                    Phase1SituationTemplate(
                        "single-object-phase3",
                        salient_object_variables=[
                            phase3_standard_object("single-object")
                        ],
                        syntax_hints=[IGNORE_COLORS],
                    ),
                    chooser=CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_3_ONTOLOGY,
                ),
            ]
        ),
    )


def phase_3_m4_core_eval(  # pylint: disable=unused-argument
    num_samples: int,
    num_noise_objects: int,
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
    params: Parameters = Parameters.empty(),
) -> Phase3InstanceGroup:
    return phase3_instances(
        "phase3-m4-core-eval",
        chain(
            *[
                all_possible(
                    Phase1SituationTemplate(
                        "m4-core-eval",
                        salient_object_variables=[
                            phase3_standard_object(
                                "object-background-1",
                                required_properties=[PHASE_3_M4_CORE_CONCEPT],
                            )
                        ],
                        background_object_variables=[
                            phase3_standard_object(
                                "object-background-2",
                                required_properties=[PHASE_3_M4_STRETCH_CONCEPT],
                            )
                        ],
                        syntax_hints=[IGNORE_COLORS],
                    ),
                    chooser=CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_3_ONTOLOGY,
                ),
            ]
        ),
    )


def phase_3_m4_stretch_eval(  # pylint: disable=unused-argument
    num_samples: int,
    num_noise_objects: int,
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
    params: Parameters = Parameters.empty(),
) -> Phase3InstanceGroup:
    return phase3_instances(
        "phase3-m4-stretch-eval",
        chain(
            *[
                all_possible(
                    Phase1SituationTemplate(
                        "m4-stretch-eval",
                        salient_object_variables=[
                            phase3_standard_object(
                                "object-background-1",
                                required_properties=[PHASE_3_M4_STRETCH_CONCEPT],
                            )
                        ],
                        background_object_variables=[
                            phase3_standard_object(
                                "object-background-2",
                                required_properties=[PHASE_3_M4_STRETCH_CONCEPT],
                            )
                        ],
                        syntax_hints=[IGNORE_COLORS],
                    ),
                    chooser=CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_3_ONTOLOGY,
                ),
            ]
        ),
    )
