from abc import ABC
from pathlib import Path
from typing import AbstractSet, Union

from adam.perception.deprecated import LanguageAlignedPerception
from adam.semantics import AttributeConcept, Concept, ObjectSemanticNode
from attr.validators import instance_of

from adam.language import LinguisticDescription
from adam.learner import LearningExample
from adam.learner.learner_utils import assert_static_situation
from adam.learner.object_recognizer import (
    ObjectRecognizer,
    PerceptionGraphFromObjectRecognizer,
)
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.subset import (
    AbstractTemplateSubsetLearner,
    AbstractTemplateSubsetLearnerNew,
)
from adam.learner.surface_templates import (
    SurfaceTemplateBoundToSemanticNodes,
    SLOT1,
    STANDARD_SLOT_VARIABLES,
    SurfaceTemplate,
)
from adam.learner.template_learner import (
    AbstractTemplateLearner,
    AbstractTemplateLearnerNew,
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
from attr import attrib, attrs
from immutablecollections import immutabledict, immutableset
from vistautils.span import Span


@attrs
class AbstractAttributeTemplateLearnerNew(AbstractTemplateLearnerNew, ABC):
    def _candidate_templates(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> AbstractSet[SurfaceTemplateBoundToSemanticNodes]:
        ret = []
        language_concept_alignment = (
            language_perception_semantic_alignment.language_concept_alignment
        )
        # Find all objects we have recognized...
        for (
            object_node,
            span_for_object,
        ) in language_concept_alignment.node_to_language_span.items():
            if isinstance(object_node, ObjectSemanticNode):
                # Any words immediately before them or after them are candidate attributes.
                # See https://github.com/isi-vista/adam/issues/791 .
                preceding_token_index = span_for_object.start - 1
                if (
                    preceding_token_index >= 0
                    and not language_concept_alignment.token_index_is_aligned(
                        preceding_token_index
                    )
                ):
                    ret.append(
                        SurfaceTemplateBoundToSemanticNodes(
                            language_concept_alignment.to_surface_template(
                                {object_node: SLOT1},
                                restrict_to_span=Span(
                                    preceding_token_index, span_for_object.end
                                ),
                            ),
                            {SLOT1: object_node},
                        )
                    )
                following_token_index = span_for_object.end + 1
                if following_token_index < len(
                    language_concept_alignment.language.as_token_sequence()
                ) and not language_concept_alignment.token_index_is_aligned(
                    following_token_index
                ):
                    ret.append(
                        SurfaceTemplateBoundToSemanticNodes(
                            language_concept_alignment.to_surface_template(
                                {object_node: SLOT1},
                                restrict_to_span=Span(
                                    span_for_object.start, following_token_index
                                ),
                            ),
                            {SLOT1: object_node},
                        )
                    )

        return immutableset(ret)


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
        self, language_concept_alignment: LanguageAlignedPerception
    ) -> LanguageAlignedPerception:
        post_recognition_object_perception_alignment = self._object_recognizer.match_objects_with_language_old(
            language_concept_alignment
        )
        return post_recognition_object_perception_alignment

    def _preprocess_scene_for_description(
        self, perception_graph: PerceptionGraph
    ) -> PerceptionGraphFromObjectRecognizer:
        return self._object_recognizer.match_objects_old(perception_graph)

    def _extract_surface_template(
        self, language_concept_alignment: LanguageAlignedPerception
    ) -> SurfaceTemplate:
        if len(language_concept_alignment.aligned_nodes) > 1:
            raise RuntimeError("Input has too many aligned nodes for us to handle.")

        return language_concept_alignment.to_surface_template(
            object_node_to_template_variable=immutabledict(
                zip(language_concept_alignment.aligned_nodes, STANDARD_SLOT_VARIABLES)
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


@attrs
class SubsetAttributeLearnerNew(
    AbstractTemplateSubsetLearnerNew, AbstractAttributeTemplateLearnerNew
):
    def _can_learn_from(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> bool:
        return (
            len(
                language_perception_semantic_alignment.perception_semantic_alignment.semantic_nodes
            )
            > 1
        )

    def _preprocess_scene(
        self, perception_semantic_alignment: PerceptionSemanticAlignment
    ) -> PerceptionSemanticAlignment:
        return perception_semantic_alignment

    def _new_concept(self, debug_string: str) -> AttributeConcept:
        return AttributeConcept(debug_string)

    def _hypotheses_from_perception(
        self,
        learning_state: LanguagePerceptionSemanticAlignment,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,
    ) -> AbstractSet[PerceptionGraphTemplate]:
        # This makes a hypothesis for the whole graph, with the wildcard slot
        # at each recognized object.
        return immutableset(
            [
                PerceptionGraphTemplate.from_graph(
                    learning_state.perception_semantic_alignment.perception_graph,
                    template_variable_to_matched_object_node=bound_surface_template.slot_to_semantic_node,
                )
            ]
        )

    def _keep_hypothesis(
        self,
        *,
        hypothesis: PerceptionGraphTemplate,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,
    ) -> bool:
        if len(hypothesis.graph_pattern) < 2:
            # We need at least two nodes - a wildcard and a property -
            # for meaningful attribute semantics.
            return False
        return True
