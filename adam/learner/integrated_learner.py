import itertools
import logging
from itertools import chain, combinations
from pathlib import Path
from typing import AbstractSet, Iterable, Iterator, Mapping, Optional, Tuple, List
from more_itertools import flatten, one

from adam.language import LinguisticDescription, TokenSequenceLinguisticDescription
from adam.language_specific.english import ENGLISH_BLOCK_DETERMINERS
from adam.learner import LearningExample, TopLevelLanguageLearner
from adam.learner.alignments import (
    LanguageConceptAlignment,
    LanguagePerceptionSemanticAlignment,
    PerceptionSemanticAlignment,
)
from adam.learner.language_mode import LanguageMode
from adam.learner.plurals import SubsetPluralLearnerNew
from adam.learner.surface_templates import MASS_NOUNS, SLOT1
from adam.learner.template_learner import TemplateLearner
from adam.perception import PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception.perception_graph import PerceptionGraph
from adam.semantics import (
    ActionSemanticNode,
    AttributeSemanticNode,
    ObjectSemanticNode,
    RelationSemanticNode,
    SemanticNode,
    GROUND_OBJECT_CONCEPT,
)
from attr import attrib, attrs
from attr.validators import instance_of, optional
from immutablecollections import (
    ImmutableSet,
    ImmutableSetMultiDict,
    immutabledict,
    immutablesetmultidict,
)
from immutablecollections.converter_utils import _to_immutableset


class LanguageLearnerNew:
    def observe(
        self,
        learning_example: LearningExample[
            DevelopmentalPrimitivePerceptionFrame, LinguisticDescription
        ],
        observation_num: int = -1,
    ) -> None:
        pass


@attrs
class IntegratedTemplateLearner(
    TopLevelLanguageLearner[
        DevelopmentalPrimitivePerceptionFrame, TokenSequenceLinguisticDescription
    ]
):
    """
    A `LanguageLearner` which uses template-based syntax to learn objects, attributes, relations,
    and actions all at once.
    """

    object_learner: TemplateLearner = attrib(validator=instance_of(TemplateLearner))
    attribute_learner: Optional[TemplateLearner] = attrib(
        validator=optional(instance_of(TemplateLearner)), default=None
    )

    relation_learner: Optional[TemplateLearner] = attrib(
        validator=optional(instance_of(TemplateLearner)), default=None
    )

    action_learner: Optional[TemplateLearner] = attrib(
        validator=optional(instance_of(TemplateLearner)), default=None
    )

    _max_attributes_per_word: int = attrib(validator=instance_of(int), default=3)

    _observation_num: int = attrib(init=False, default=0)
    _sub_learners: List[TemplateLearner] = attrib(init=False)

    def observe(
        self,
        learning_example: LearningExample[
            DevelopmentalPrimitivePerceptionFrame, LinguisticDescription
        ],
        observation_num: int = -1,
    ) -> None:
        if observation_num >= 0:
            logging.info(
                "Observation %s: %s",
                observation_num,
                learning_example.linguistic_description.as_token_string(),
            )
        else:
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
            perception_semantic_alignment=PerceptionSemanticAlignment(
                perception_graph=self._extract_perception_graph(
                    learning_example.perception
                ),
                semantic_nodes=[],
            ),
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
                # See https://github.com/isi-vista/adam/issues/792 .
                if not learning_example.perception.is_dynamic():
                    sub_learner.learn_from(
                        current_learner_state, observation_num=observation_num
                    )

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
        ret = []
        if self.action_learner:
            ret.extend(
                [
                    (action_tokens, 1.0)
                    for action in learner_semantics.actions
                    for action_tokens in self._instantiate_action(
                        action, learner_semantics
                    )
                    # ensure we have some way of expressing this action
                    if self.action_learner.templates_for_concept(action.concept)
                ]
            )

        if self.relation_learner:
            ret.extend(
                [
                    (relation_tokens, 1.0)
                    for relation in learner_semantics.relations
                    for relation_tokens in self._instantiate_relation(
                        relation, learner_semantics
                    )
                    # ensure we have some way of expressing this relation
                    if self.relation_learner.templates_for_concept(relation.concept)
                ]
            )
        ret.extend(
            [
                (object_tokens, 1.0)
                for object_ in learner_semantics.objects
                for object_tokens in self._instantiate_object(object_, learner_semantics)
                # ensure we have some way of expressing this object
                if self.object_learner.templates_for_concept(object_.concept)
            ]
        )
        return immutabledict(
            (TokenSequenceLinguisticDescription(tokens), score) for (tokens, score) in ret
        )

    def _add_determiners(
        self, object_node: ObjectSemanticNode, cur_string: Tuple[str, ...]
    ) -> Tuple[str, ...]:
        if (
            self.object_learner._language_mode  # pylint: disable=protected-access
            != LanguageMode.ENGLISH
        ):
            return cur_string
        # If plural, we want to strip any "a" that might preceed a noun after "many" or "two"
        if isinstance(self.attribute_learner, SubsetPluralLearnerNew):
            if "a" in cur_string:
                a_position = cur_string.index("a")
                if a_position > 0 and cur_string[a_position-1] in ['many', 'two']:
                    return tuple([token for i, token in enumerate(cur_string) if i != a_position])
        # English-specific hack to deal with us not understanding determiners:
        # https://github.com/isi-vista/adam/issues/498
        # The "is lower" check is a hack to block adding a determiner to proper names.
        # Ground is a specific thing so we special case this to be assigned
        if object_node.concept == GROUND_OBJECT_CONCEPT:
            return tuple(chain(("the",), cur_string))
        elif (
            object_node.concept.debug_string not in MASS_NOUNS
            and object_node.concept.debug_string.islower()
            and not cur_string[0] in ENGLISH_BLOCK_DETERMINERS
        ):
            return tuple(chain(("a",), cur_string))
        else:
            return cur_string

    def _instantiate_object(
        self, object_node: ObjectSemanticNode, learner_semantics: "LearnerSemantics"
    ) -> Iterator[Tuple[str, ...]]:

        # For now, we assume the order in which modifiers is expressed is arbitrary.
        attributes_we_can_express = (
            [
                attribute
                for attribute in learner_semantics.objects_to_attributes[object_node]
                if self.attribute_learner.templates_for_concept(attribute.concept)
            ]
            if self.attribute_learner
            else []
        )
        # We currently cannot deal with relations that modify objects embedded in other expressions.
        # See https://github.com/isi-vista/adam/issues/794 .
        # relations_for_object = learner_semantics.objects_to_relation_in_slot1[object_node]

        for template in self.object_learner.templates_for_concept(object_node.concept):

            cur_string = template.instantiate(
                template_variable_to_filler=immutabledict()
            ).as_token_sequence()

            for num_attributes in range(
                min(len(attributes_we_can_express), self._max_attributes_per_word)
            ):
                for attribute_combinations in combinations(
                    attributes_we_can_express,
                    # +1 because the range starts at 0
                    num_attributes + 1,
                ):
                    for attribute in attribute_combinations:
                        # we know, but mypy does not, that self.attribute_learner is not None
                        for (
                            attribute_template
                        ) in self.attribute_learner.templates_for_concept(  # type: ignore
                            attribute.concept
                        ):
                            yield self._add_determiners(
                                object_node,
                                attribute_template.instantiate(
                                    template_variable_to_filler={SLOT1: cur_string}
                                ).as_token_sequence(),
                            )
            yield self._add_determiners(object_node, cur_string)

    def _instantiate_relation(
        self, relation_node: RelationSemanticNode, learner_semantics: "LearnerSemantics"
    ) -> Iterator[Tuple[str, ...]]:
        if not self.relation_learner:
            raise RuntimeError("Cannot instantiate relations without a relation learner")

        slots_to_instantiations = {
            slot: list(self._instantiate_object(slot_filler, learner_semantics))
            for (slot, slot_filler) in relation_node.slot_fillings.items()
        }
        slot_order = tuple(slots_to_instantiations.keys())

        for relation_template in self.relation_learner.templates_for_concept(
            relation_node.concept
        ):
            all_possible_slot_fillings = itertools.product(
                *slots_to_instantiations.values()
            )
            for possible_slot_filling in all_possible_slot_fillings:
                yield relation_template.instantiate(
                    immutabledict(zip(slot_order, possible_slot_filling))
                ).as_token_sequence()

    def _instantiate_action(
        self, action_node: ActionSemanticNode, learner_semantics: "LearnerSemantics"
    ) -> Iterator[Tuple[str, ...]]:
        if not self.action_learner:
            raise RuntimeError("Cannot instantiate an action without an action learner")
        slots_to_instantiations = {
            slot: list(self._instantiate_object(slot_filler, learner_semantics))
            for (slot, slot_filler) in action_node.slot_fillings.items()
        }
        slot_order = tuple(slots_to_instantiations.keys())

        for action_template in self.action_learner.templates_for_concept(
            action_node.concept
        ):
            all_possible_slot_fillings = itertools.product(
                *slots_to_instantiations.values()
            )
            for possible_slot_filling in all_possible_slot_fillings:
                yield action_template.instantiate(
                    immutabledict(zip(slot_order, possible_slot_filling))
                ).as_token_sequence()

    def log_hypotheses(self, log_output_path: Path) -> None:
        for sub_learner in self._sub_learners:
            sub_learner.log_hypotheses(log_output_path)

    def _extract_perception_graph(
        self, perception: PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame]
    ) -> PerceptionGraph:
        if perception.is_dynamic():
            return PerceptionGraph.from_dynamic_perceptual_representation(perception)
        else:
            return PerceptionGraph.from_frame(perception.frames[0])

    @_sub_learners.default
    def _init_sub_learners(self) -> List[TemplateLearner]:
        valid_sub_learners = []
        if self.object_learner:
            valid_sub_learners.append(self.object_learner)
        if self.attribute_learner:
            valid_sub_learners.append(self.attribute_learner)
        if self.relation_learner:
            valid_sub_learners.append(self.relation_learner)
        if self.action_learner:
            valid_sub_learners.append(self.action_learner)
        return valid_sub_learners


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
    objects_to_relation_in_slot1: ImmutableSetMultiDict[
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

    @objects_to_relation_in_slot1.default
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
