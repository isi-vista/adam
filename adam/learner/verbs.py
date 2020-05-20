from abc import ABC
from pathlib import Path
from typing import AbstractSet, Mapping, Union

from attr.validators import instance_of

from adam.language import LinguisticDescription
from adam.learner import LearningExample
from adam.learner.object_recognizer import (
    ObjectRecognizer,
    PerceptionGraphFromObjectRecognizer,
)
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.subset import AbstractTemplateSubsetLearner
from adam.learner.surface_templates import STANDARD_SLOT_VARIABLES, SurfaceTemplate
from adam.learner.template_learner import (
    AbstractNewStyleTemplateLearner,
    AbstractTemplateLearner,
)
from adam.perception import PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception.perception_graph import PerceptionGraph
from adam.learner.alignments import (
    LanguageConceptAlignment,
    LanguagePerceptionSemanticAlignment,
    PerceptionSemanticAlignment,
)
from adam.semantics import Concept, ObjectSemanticNode, SyntaxSemanticsVariable
from attr import attrib, attrs
from immutablecollections import immutabledict


@attrs
class AbstractVerbTemplateLearner(
    AbstractTemplateLearner, AbstractNewStyleTemplateLearner, ABC
):
    # mypy doesn't realize that fields without defaults can come after those with defaults
    # if they are keyword-only.
    _object_recognizer: ObjectRecognizer = attrib(  # type: ignore
        validator=instance_of(ObjectRecognizer), kw_only=True
    )

    def _assert_valid_input(
        self,
        to_check: Union[
            PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame],
            LearningExample[DevelopmentalPrimitivePerceptionFrame, LinguisticDescription],
        ],
    ) -> None:
        if isinstance(to_check, LearningExample):
            perception = to_check.perception
        else:
            perception = to_check
        if len(perception.frames) != 2:
            raise RuntimeError(
                "Expected exactly two frames in a perception for verb learning"
            )

    def _extract_perception_graph(
        self, perception: PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame]
    ) -> PerceptionGraph:
        return PerceptionGraph.from_dynamic_perceptual_representation(perception)

    def _preprocess_scene_for_learning(
        self, language_concept_alignment: LanguageConceptAlignment
    ) -> LanguageConceptAlignment:
        post_recognition_object_perception_alignment = self._object_recognizer.match_objects_with_language(
            language_concept_alignment
        )
        return post_recognition_object_perception_alignment

    def _preprocess_scene_for_description(
        self, perception_graph: PerceptionGraph
    ) -> PerceptionGraphFromObjectRecognizer:
        return self._object_recognizer.match_objects(perception_graph)

    def _extract_surface_template(
        self, language_concept_alignment: LanguageConceptAlignment
    ) -> SurfaceTemplate:
        if len(language_concept_alignment.aligned_nodes) > len(STANDARD_SLOT_VARIABLES):
            raise RuntimeError("Input has too many aligned nodes for us to handle.")

        object_node_to_template_variable: Mapping[
            ObjectSemanticNode, SyntaxSemanticsVariable
        ] = immutabledict(
            zip(language_concept_alignment.aligned_nodes, STANDARD_SLOT_VARIABLES)
        )
        return SurfaceTemplate.from_language_aligned_perception(
            language_concept_alignment,
            object_node_to_template_variable=object_node_to_template_variable,
            determiner_prefix_slots=object_node_to_template_variable.values(),
        )


@attrs
class SubsetVerbLearner(AbstractTemplateSubsetLearner, AbstractVerbTemplateLearner):
    def templates_for_concept(self, concept: Concept) -> AbstractSet[SurfaceTemplate]:
        raise NotImplementedError()

    def learn_from(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> None:
        raise NotImplementedError()

    def enrich_during_learning(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> LanguagePerceptionSemanticAlignment:
        raise NotImplementedError()

    def enrich_during_description(
        self, perception_semantic_alignment: PerceptionSemanticAlignment
    ) -> PerceptionSemanticAlignment:
        raise NotImplementedError()

    def log_hypotheses(self, log_output_path: Path) -> None:
        raise NotImplementedError()

    def _hypothesis_from_perception(
        self, preprocessed_input: LanguageConceptAlignment
    ) -> PerceptionGraphTemplate:
        return PerceptionGraphTemplate.from_graph(
            preprocessed_input.perception_graph,
            template_variable_to_matched_object_node=immutabledict(
                zip(STANDARD_SLOT_VARIABLES, preprocessed_input.aligned_nodes)
            ),
        )
