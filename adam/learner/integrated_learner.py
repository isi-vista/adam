import logging
from pathlib import Path
from typing import AbstractSet, Iterable, Mapping, Optional

from more_itertools import flatten, one

from adam.learner.attributes import AbstractAttributeTemplateLearner
from adam.learner.prepositions import AbstractPrepositionTemplateLearner
from adam.learner.alignments import (
    LanguagePerceptionSemanticAlignment,
    PerceptionSemanticAlignment,
)
from adam.learner.verbs import AbstractVerbTemplateLearner
from adam.perception import PerceptionT, PerceptualRepresentation
from adam.perception.perception_graph import PerceptionGraph
from adam.learner.alignments import LanguageConceptAlignment
from adam.semantics import (
    ActionSemanticNode,
    AttributeSemanticNode,
    ObjectSemanticNode,
    RelationSemanticNode,
    SemanticNode,
)
from attr.validators import instance_of, optional

from adam.language import LinguisticDescription, TokenSequenceLinguisticDescription
from adam.learner import LanguageLearner, LearningExample
from adam.learner.objects import (
    AbstractNewStyleObjectTemplateLearner,
    AbstractObjectTemplateLearner,
)
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from attr import attrib, attrs
from immutablecollections import (
    ImmutableSet,
    ImmutableSetMultiDict,
    immutabledict,
    immutablesetmultidict,
)
from immutablecollections.converter_utils import _to_immutableset


@attrs
class IntegratedTemplateLearner(
    LanguageLearner[
        DevelopmentalPrimitivePerceptionFrame, TokenSequenceLinguisticDescription
    ]
):
    """
    A `LanguageLearner` which uses template-based syntax to learn objects, attributes, relations,
    and actions all at once.
    """

    object_learner: AbstractNewStyleObjectTemplateLearner = attrib(
        validator=instance_of(AbstractNewStyleObjectTemplateLearner)
    )
    attribute_learner: Optional[AbstractAttributeTemplateLearner] = attrib(
        validator=optional(instance_of(AbstractAttributeTemplateLearner))
    )
    relation_learner: Optional[AbstractPrepositionTemplateLearner] = attrib(
        validator=optional(instance_of(AbstractPrepositionTemplateLearner))
    )
    action_learner: Optional[AbstractVerbTemplateLearner] = attrib(
        validator=optional(instance_of(AbstractVerbTemplateLearner))
    )

    _observation_num: int = attrib(init=False, default=0)

    def observe(
        self,
        learning_example: LearningExample[
            DevelopmentalPrimitivePerceptionFrame, LinguisticDescription
        ],
    ) -> None:
        logging.info(
            "Observation %s: %s",
            self._observation_num,
            learning_example.linguistic_description.as_token_string(),
        )

        self._observation_num += 1

        # We need to track the alignment between perceived objects
        # and portions of the input language, so internally we operate over
        # LanguageAlignedPerceptions.
        current_learner_state = LanguagePerceptionSemanticAlignment(
            language_concept_alignment=LanguageConceptAlignment.create_unaligned(
                language=learning_example.linguistic_description
            ),
            perception_graph=self._extract_perception_graph(learning_example.perception),
        )

        # We iteratively let each "layer" of semantic analysis attempt
        # to learn from the perception,
        # and then to annotate the perception with any semantic alignments it knows.
        for sub_learner in [
            self.object_learner,
            self.attribute_learner,
            self.relation_learner,
        ]:
            if sub_learner:
                # Currently we do not attempt to learn static things from dynamic situations
                # because the static learners do not know how to deal with the temporal
                # perception graph edge wrappers.
                # @Reviewers: do not let me merge without making an issue for this.
                if not learning_example.perception.is_dynamic():
                    sub_learner.learn_from(current_learner_state)

                current_learner_state = sub_learner.enrich_during_learning(
                    current_learner_state
                )

        if learning_example.perception.is_dynamic() and self.action_learner:
            self.action_learner.learn_from(current_learner_state)

    def describe(
        self, perception: PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame]
    ) -> Mapping[LinguisticDescription, float]:

        perception_graph = self._extract_perception_graph(perception)

        cur_description_state = PerceptionSemanticAlignment.create_unaligned(
            perception_graph
        )

        for sub_learner in [
            self.object_learner,
            self.attribute_learner,
            self.relation_learner,
        ]:
            if sub_learner:
                cur_description_state = sub_learner.enrich_during_description(
                    cur_description_state
                )

        if perception.is_dynamic() and self.action_learner:
            cur_description_state = self.action_learner.enrich_during_description(
                cur_description_state
            )

        return self._linguistic_descriptions_from_semantics(
            cur_description_state.semantic_nodes
        )

    def _linguistic_descriptions_from_semantics(
        self, semantic_nodes: AbstractSet[SemanticNode]
    ) -> Mapping[LinguisticDescription, float]:
        learner_semantics = LearnerSemantics.from_nodes(semantic_nodes)
        if (
            learner_semantics.attributes
            or learner_semantics.relations
            or learner_semantics.actions
        ):
            raise RuntimeError("Currently we can only handle objects")
        return {
            template.instantiate(template_variable_to_filler=immutabledict()): 1.0
            for object_node in learner_semantics.objects
            for template in self.object_learner.templates_for_concept(object_node.concept)
        }

    def log_hypotheses(self, log_output_path: Path) -> None:
        raise NotImplementedError("implement me")

    def _extract_perception_graph(
        self, perception: PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame]
    ) -> PerceptionGraph:
        if perception.is_dynamic():
            return PerceptionGraph.from_dynamic_perceptual_representation(perception)
        else:
            return PerceptionGraph.from_frame(perception.frames[0])


@attrs(frozen=True)
class LearnerSemantics:
    """
    Represent's the learner's semantic (rather than perceptual) understanding of a situation.

    The learner is assumed to view the situation as a collection of *objects* which possess
    *attributes*, have *relations* to one another, and serve as the arguments of *actions*.
    """

    objects: ImmutableSet[ObjectSemanticNode] = attrib(converter=_to_immutableset)
    attributes: ImmutableSet[AttributeSemanticNode] = attrib(converter=_to_immutableset)
    relations: ImmutableSet[RelationSemanticNode] = attrib(converter=_to_immutableset)
    actions: ImmutableSet[ActionSemanticNode] = attrib(converter=_to_immutableset)

    objects_to_attributes: ImmutableSetMultiDict[
        ObjectSemanticNode, AttributeSemanticNode
    ] = attrib(init=False)
    objects_to_relations: ImmutableSetMultiDict[
        ObjectSemanticNode, RelationSemanticNode
    ] = attrib(init=False)
    objects_to_actions: ImmutableSetMultiDict[
        ObjectSemanticNode, ActionSemanticNode
    ] = attrib(init=False)

    @staticmethod
    def from_nodes(semantic_nodes: Iterable[SemanticNode]) -> "LearnerSemantics":
        return LearnerSemantics(
            objects=[
                node for node in semantic_nodes if isinstance(node, ObjectSemanticNode)
            ],
            attributes=[
                node for node in semantic_nodes if isinstance(node, AttributeSemanticNode)
            ],
            relations=[
                node for node in semantic_nodes if isinstance(node, RelationSemanticNode)
            ],
            actions=[
                node for node in semantic_nodes if isinstance(node, ActionSemanticNode)
            ],
        )

    @objects_to_attributes.default
    def _init_objects_to_attributes(
        self
    ) -> ImmutableSetMultiDict[ObjectSemanticNode, AttributeSemanticNode]:
        return immutablesetmultidict(
            (one(attribute.slot_fillings.values()), attribute)
            for attribute in self.attributes
        )

    @objects_to_relations.default
    def _init_objects_to_relations(
        self
    ) -> ImmutableSetMultiDict[ObjectSemanticNode, AttributeSemanticNode]:
        return immutablesetmultidict(
            flatten(
                [
                    (slot_filler, relation)
                    for slot_filler in relation.slot_fillings.values()
                ]
                for relation in self.relations
            )
        )

    @objects_to_actions.default
    def _init_objects_to_actions(
        self
    ) -> ImmutableSetMultiDict[ObjectSemanticNode, AttributeSemanticNode]:
        return immutablesetmultidict(
            flatten(
                [(slot_filler, action) for slot_filler in action.slot_fillings.values()]
                for action in self.actions
            )
        )
