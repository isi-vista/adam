from abc import ABC, abstractmethod
from itertools import chain

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
    IGNORE_SHAPE_PROPERTY,
)
from adam.ontology.phase3_ontology import (
    GAILA_PHASE_3_ONTOLOGY,
    SHAPE_PROPERTY_DESCRIPTION,
    BLOCK,
)
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from vistautils.parameters import Parameters

from adam.situation.templates.phase1_templates import (
    all_possible,
    Phase1SituationTemplate,
)


class Phase3CurriculumCallable(ABC):
    @abstractmethod
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
            "The base Phase3CurriculumCallable does not implement a curriculum return."
        )


class Phase3OneObjectsCurriculum(Phase3CurriculumCallable):
    """A curriculum with a single object from the Phase 3 selection."""

    def __call__(
        self,
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
                    all_possible(
                        Phase1SituationTemplate(
                            "block_types",
                            salient_object_variables=[
                                phase3_standard_object(
                                    "block",
                                    BLOCK,
                                    added_properties=[SHAPE_PROPERTY_DESCRIPTION],
                                )
                            ],
                            syntax_hints=[IGNORE_SHAPE_PROPERTY],
                        ),
                        chooser=CHOOSER_FACTORY(),
                        ontology=GAILA_PHASE_3_ONTOLOGY,
                    ),
                ]
            ),
        )
