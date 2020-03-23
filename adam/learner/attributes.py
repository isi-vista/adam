from abc import ABC
from typing import Union

from attr.validators import instance_of

from adam.language import LinguisticDescription
from adam.learner import LearningExample
from adam.learner.learner_utils import assert_static_situation
from adam.learner.object_recognizer import (
    ObjectRecognizer,
    PerceptionGraphFromObjectRecognizer,
)
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.subset import AbstractTemplateSubsetLearner
from adam.learner.surface_templates import STANDARD_SLOT_VARIABLES, SurfaceTemplate
from adam.learner.template_learner import AbstractTemplateLearner
from adam.perception import PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception.perception_graph import LanguageAlignedPerception, PerceptionGraph
from attr import attrib, attrs
from immutablecollections import immutabledict, immutableset


@attrs
class AbstractAttributeTemplateLearner(AbstractTemplateLearner, ABC):
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
        assert_static_situation(to_check)

    def _extract_perception_graph(
        self, perception: PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame]
    ) -> PerceptionGraph:
        return PerceptionGraph.from_frame(perception.frames[0])

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
        if len(preprocessed_input.aligned_nodes) > 1:
            raise RuntimeError("Input has too many aligned nodes for us to handle.")

        return SurfaceTemplate.from_language_aligned_perception(
            preprocessed_input,
            object_node_to_template_variable=immutabledict(
                zip(preprocessed_input.aligned_nodes, STANDARD_SLOT_VARIABLES)
            ),
            # This is a hack to handle determiners.
            # For attributes at the moment we learn the determiner together with the
            # attribute, which is not ideal.
            determiner_prefix_slots=immutableset(),
        )


@attrs
class SubsetAttributeLearner(
    AbstractTemplateSubsetLearner, AbstractAttributeTemplateLearner
):
    def _hypothesis_from_perception(
        self, preprocessed_input: LanguageAlignedPerception
    ) -> PerceptionGraphTemplate:
        num_nodes_aligned_to_language = len(preprocessed_input.aligned_nodes)
        if num_nodes_aligned_to_language != 1:
            raise RuntimeError(
                f"Attribute learner can work only with a single aligned node,"
                f"but got {num_nodes_aligned_to_language}. Language is "
                f"{preprocessed_input.language.as_token_string()}"
            )

        return PerceptionGraphTemplate.from_graph(
            preprocessed_input.perception_graph,
            template_variable_to_matched_object_node=immutabledict(
                zip(STANDARD_SLOT_VARIABLES, preprocessed_input.aligned_nodes)
            ),
        )
