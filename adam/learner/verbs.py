from abc import ABC
from typing import Mapping, Union

from attr.validators import instance_of

from adam.language import LinguisticDescription
from adam.learner import LearningExample
from adam.learner.object_recognizer import (
    ObjectRecognizer,
    PerceptionGraphFromObjectRecognizer,
)
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.subset import AbstractTemplateSubsetLearner
from adam.learner.surface_templates import (
    STANDARD_SLOT_VARIABLES,
    SurfaceTemplate,
    SurfaceTemplateVariable,
)
from adam.learner.template_learner import AbstractTemplateLearner
from adam.perception import PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception.perception_graph import (
    LanguageAlignedPerception,
    MatchedObjectNode,
    PerceptionGraph,
)
from attr import attrib, attrs
from immutablecollections import immutabledict


@attrs
class AbstractVerbTemplateLearner(AbstractTemplateLearner, ABC):
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
        self, language_aligned_perception: LanguageAlignedPerception
    ) -> LanguageAlignedPerception:
        post_recognition_object_perception_alignment = self._object_recognizer.match_objects_with_language(
            language_aligned_perception
        )
        return post_recognition_object_perception_alignment

    def _preprocess_scene_for_description(
        self, perception_graph: PerceptionGraph
    ) -> PerceptionGraphFromObjectRecognizer:
        return self._object_recognizer.match_objects(perception_graph)

    def _extract_surface_template(
        self, preprocessed_input: LanguageAlignedPerception
    ) -> SurfaceTemplate:
        if len(preprocessed_input.aligned_nodes) > len(STANDARD_SLOT_VARIABLES):
            raise RuntimeError("Input has too many aligned nodes for us to handle.")

        object_node_to_template_variable: Mapping[
            MatchedObjectNode, SurfaceTemplateVariable
        ] = immutabledict(zip(preprocessed_input.aligned_nodes, STANDARD_SLOT_VARIABLES))
        return SurfaceTemplate.from_language_aligned_perception(
            preprocessed_input,
            object_node_to_template_variable=object_node_to_template_variable,
            determiner_prefix_slots=object_node_to_template_variable.values(),
        )


@attrs
class SubsetVerbLearner(AbstractTemplateSubsetLearner, AbstractVerbTemplateLearner):
    def _hypothesis_from_perception(
        self, preprocessed_input: LanguageAlignedPerception
    ) -> PerceptionGraphTemplate:
        return PerceptionGraphTemplate.from_graph(
            preprocessed_input.perception_graph,
            template_variable_to_matched_object_node=immutabledict(
                zip(STANDARD_SLOT_VARIABLES, preprocessed_input.aligned_nodes)
            ),
        )
