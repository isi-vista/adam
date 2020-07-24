import itertools
import logging
from itertools import combinations
from pathlib import Path
from typing import AbstractSet, Iterable, Iterator, List, Mapping, Optional, Tuple

import more_itertools

from adam.language import LinguisticDescription, TokenSequenceLinguisticDescription
from adam.language_specific.english import (
    ENGLISH_BLOCK_DETERMINERS,
    ENGLISH_MASS_NOUNS,
    ENGLISH_RECOGNIZED_PARTICULARS,
    ENGLISH_THE_WORDS,
)
from adam.learner import LearningExample, TopLevelLanguageLearner
from adam.learner.alignments import (
    LanguageConceptAlignment,
    LanguagePerceptionSemanticAlignment,
    PerceptionSemanticAlignment,
)
from adam.learner.language_mode import LanguageMode
from adam.learner.surface_templates import SLOT1, SLOT2, SurfaceTemplate
from adam.learner.template_learner import SemanticTemplateLearner, TemplateLearner
from adam.perception import PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception.perception_graph import PerceptionGraph
from adam.semantics import (
    ActionSemanticNode,
    AttributeSemanticNode,
    LearnerSemantics,
    NumberConcept,
    ObjectSemanticNode,
    QuantificationSemanticNode,
    RelationSemanticNode,
    SemanticNode,
)
from attr import attrib, attrs
from attr.validators import instance_of, optional
from immutablecollections import immutabledict, immutablelistmultidict
from vistautils.iter_utils import only


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

    object_learner: TemplateLearner = attrib(
        validator=instance_of(TemplateLearner), kw_only=True
    )

    number_learner: SemanticTemplateLearner = attrib(
        validator=instance_of(SemanticTemplateLearner), kw_only=True
    )

    _language_mode: LanguageMode = attrib(
        validator=instance_of(LanguageMode), kw_only=True
    )
    """
    Why does a language-independent learner need to know what language it is learning?
    This is to support language-specific logic for English determiners, 
    which is out-of-scope for ADAM and therefore hard-coded.
    """

    attribute_learner: Optional[TemplateLearner] = attrib(
        validator=optional(instance_of(TemplateLearner)), default=None, kw_only=True
    )

    relation_learner: Optional[TemplateLearner] = attrib(
        validator=optional(instance_of(TemplateLearner)), default=None, kw_only=True
    )

    action_learner: Optional[TemplateLearner] = attrib(
        validator=optional(instance_of(TemplateLearner)), default=None, kw_only=True
    )

    _max_attributes_per_word: int = attrib(
        validator=instance_of(int), default=3, kw_only=True
    )

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

        if self.number_learner:
            self.number_learner.learn_from(
                current_learner_state.language_concept_alignment,
                LearnerSemantics.from_nodes(
                    current_learner_state.perception_semantic_alignment.semantic_nodes
                ),
            )

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
        ret: List[Tuple[Tuple[str, ...], float]] = []

        # To handle numbers/plurality,  we build copies of the semantics pruned
        # to contain only those things which are predicated in common of
        # each set of multiple copies of the same object type.
        # The case of single objects is handled within this as a special case.
        for quantification_sub_semantics in self._quantification_sub_semantics(
            learner_semantics
        ):
            ret.extend(self._to_tokens(quantification_sub_semantics))

        return immutabledict(
            (TokenSequenceLinguisticDescription(tokens), score) for (tokens, score) in ret
        )

    def _quantification_sub_semantics(
        self, semantics: LearnerSemantics
    ) -> Iterable[LearnerSemantics]:
        """
        Our approach to handling plurals and other quantifiers is
        to identify when there are multiple objects of the same type,
        and produce representations of all the aspects of the semantics
        which are common to all of them.
        These "quantification sub-semantics",
        along with indications of the quantification information,
        can then be turned into tokens by `_to_tokens`.
        """
        # Below we will use the running example of a scene which has
        # * three balls (b_1,b_2,b_3), two of which are red (b_1,b_2)
        #      and one of which is green ( b_3)
        # * two tables (t_1, t_2). One red ball (b_1) is on one table (t_1),
        #      and both other balls (b2, b3) are on another (t_2).

        # First, determine groups of objects with the same concept
        # (e.g. that there are three balls and two tables in a scene).
        concept_to_objects = immutablelistmultidict(
            (object_.concept, object_) for object_ in semantics.objects
        )
        # Second, for each group, iterate determine all subsets of the objects.
        # e.g. for balls, ball by itself, {b_1, b_2}, {b_1, b_3}, {b_2, b_3}, {b_1, b_2, b_3}
        # and for tables, each table by itself and {t_1, t_2}.
        concept_to_object_combinations = immutablelistmultidict(
            (concept, combo)
            for (concept, objects) in concept_to_objects.as_dict().items()
            for combo in more_itertools.powerset(objects)
            # exclude the empty set
            if combo
        )

        # Now we take all combinations of subsets.
        # e.g. ({b_1, b_2}, t_1), ({b_1, b_2}, {t_1, t_2}), etc.
        for object_grouping in itertools.product(
            *concept_to_object_combinations.value_groups()
        ):
            # For each combination of subsets, we determine what original_attributes, relations, and verbs
            # hold true for *all* the the selected elements of a type,
            # taking into account the collapsing together of other nodes.
            # For example, if {b_1, b_2) are grouped,
            # then "red" is a valid modifier for the collapsed group.
            # on(table_obj) is a valid modified for {b_1, b_2} if t_1 and t_2 are grouped,
            # but it is not if they are not grouped.
            # Note this allows only one grouping per object type,
            # so we cannot handle e.g. "two red balls beside three green balls"

            # We start by making new "quantified" ObjectSemanticNodes to represent each group,
            # and we attach a quantification node to each.
            # We will call these "quantified objects" or "Q-Objects".
            # They are the only thing which will appear in the returned "quantified semantics".
            # The objects in the original semantics we will call "original objects" or "O-Objects".
            original_object_to_quantified_object = {}
            quantified_object_to_original_objects_builder = []
            quantified_semantics: List[SemanticNode] = []
            for concept, original_objects_for_concept in zip(
                concept_to_object_combinations.keys(), object_grouping
            ):
                num_objects = len(original_objects_for_concept)
                if num_objects > 1:
                    quantified_object = ObjectSemanticNode(concept)
                    quantified_semantics.append(quantified_object)
                    quantified_semantics.append(
                        QuantificationSemanticNode(
                            NumberConcept(num_objects, debug_string=str(num_objects)),
                            slot_fillings=[(SLOT1, quantified_object)],
                        )
                    )
                    for original_object in original_objects_for_concept:
                        original_object_to_quantified_object[
                            original_object
                        ] = quantified_object
                        quantified_object_to_original_objects_builder.append(
                            (quantified_object, original_object)
                        )
                else:
                    # If there's only one instance of a concept,
                    # we can just use the original object itself as the "quantified object".
                    original_object = only(original_objects_for_concept)
                    quantified_semantics.append(original_object)
                    quantified_semantics.append(
                        QuantificationSemanticNode(
                            NumberConcept(1, debug_string="1"),
                            slot_fillings=[(SLOT1, original_object)],
                        )
                    )
                    original_object_to_quantified_object[
                        original_object
                    ] = original_object
                    quantified_object_to_original_objects_builder.append(
                        (original_object, original_object)
                    )

            quantified_object_to_original_objects = immutablelistmultidict(
                quantified_object_to_original_objects_builder
            )

            # Each attribute in the original semantics can be "projected" up to an attribute
            # of a Q-Object in the quantified semantics,
            # if there is a matching "original attribute" asserted for each O-Object which is
            # mapped to the Q-Object.
            # e.g. if b_1 and b_2 are mapped to the same Q-Object B_0,
            # then you can say B_0 is red iff both b_1 and b_2 are red.
            for original_attribute in semantics.attributes:
                original_argument = original_attribute.slot_fillings[SLOT1]
                if original_argument not in original_object_to_quantified_object:
                    # This is the case where e.g. there are three balls,
                    # but we are only including two in this particular quantified semantics.
                    continue
                quantified_argument = original_object_to_quantified_object[
                    original_argument
                ]

                include_projection_in_quantified_semantics = True
                other_objects_mapped_to_same_quantified_object = quantified_object_to_original_objects[
                    quantified_argument
                ]

                if len(other_objects_mapped_to_same_quantified_object) == 1:
                    # Since the original object is the same as the quantified object,
                    # we can just reuse the assertion object as well.
                    quantified_semantics.append(original_attribute)
                    continue

                for (
                    other_original_argument_mapped_to_quantified_argument
                ) in other_objects_mapped_to_same_quantified_object:
                    if (
                        other_original_argument_mapped_to_quantified_argument
                        is not original_argument
                    ):
                        found_a_matching_attribute = False
                        for attribute in semantics.attributes:
                            if (
                                attribute.concept == original_attribute.concept
                                and attribute.slot_fillings[SLOT1]
                                is other_original_argument_mapped_to_quantified_argument
                            ):
                                found_a_matching_attribute = True
                        if not found_a_matching_attribute:
                            include_projection_in_quantified_semantics = False
                if include_projection_in_quantified_semantics:
                    # Note that duplicates attributes will get added to the quantified semantics
                    # because each of the original attributes which are projected to the
                    # "quantified attribute" will add a copy.
                    # But it all ends up in a set anyway, so it doesn't matter.
                    quantified_semantics.append(
                        AttributeSemanticNode(
                            concept=original_attribute.concept,
                            slot_fillings=[(SLOT1, quantified_argument)],
                        )
                    )

            # TODO: discuss difficulties with relations and actions
            for assertion_set in (semantics.relations, semantics.actions):
                for assertion in assertion_set:  # type: ignore
                    all_arguments_already_quantified = True
                    for slot in (SLOT1, SLOT2):
                        original_slot_filler = assertion.slot_fillings[slot]
                        quantified_slot_filler = original_object_to_quantified_object[
                            original_slot_filler
                        ]
                        if (
                            len(
                                quantified_object_to_original_objects[
                                    quantified_slot_filler
                                ]
                            )
                            > 1
                        ):
                            all_arguments_already_quantified = False
                            break
                    if all_arguments_already_quantified:
                        quantified_semantics.append(assertion)

            yield LearnerSemantics.from_nodes(quantified_semantics)

    def _to_tokens(
        self, semantics: LearnerSemantics
    ) -> Iterable[Tuple[Tuple[str, ...], float]]:
        ret = []
        if self.action_learner:
            ret.extend(
                [
                    (action_tokens, 1.0)
                    for action in semantics.actions
                    for action_tokens in self._instantiate_action(action, semantics)
                    # ensure we have some way of expressing this action
                    if self.action_learner.templates_for_concept(action.concept)
                ]
            )

        if self.relation_learner:
            ret.extend(
                [
                    (relation_tokens, 1.0)
                    for relation in semantics.relations
                    for relation_tokens in self._instantiate_relation(relation, semantics)
                    # ensure we have some way of expressing this relation
                    if self.relation_learner.templates_for_concept(relation.concept)
                ]
            )
        ret.extend(
            [
                (object_tokens, 1.0)
                for object_ in semantics.objects
                for object_tokens in self._instantiate_object(object_, semantics)
                # ensure we have some way of expressing this object
                if self.object_learner.templates_for_concept(object_.concept)
            ]
        )
        return ret

    def _handle_quantifiers(
        self,
        semantics: LearnerSemantics,
        object_node: ObjectSemanticNode,
        cur_string: Tuple[str, ...],
    ) -> Iterable[Tuple[str, ...]]:
        # English-specific special case, since we don't handle learning determiner information
        # https://github.com/isi-vista/adam/issues/498
        block_determiners = False
        if self._language_mode == LanguageMode.ENGLISH:
            for attribute in semantics.objects_to_attributes[object_node]:
                if attribute.concept.debug_string in ENGLISH_BLOCK_DETERMINERS:
                    # These are things like "your" which block determiners
                    block_determiners = True
            if object_node.concept.debug_string in ENGLISH_MASS_NOUNS:
                block_determiners = True
            if object_node.concept.debug_string in ENGLISH_RECOGNIZED_PARTICULARS:
                block_determiners = True
            if object_node.concept.debug_string in ENGLISH_THE_WORDS:
                yield ENGLISH_THE_TEMPLATE.instantiate(
                    {SLOT1: cur_string}
                ).as_token_sequence()
                return
        if block_determiners:
            yield cur_string
        else:
            found_a_quantifier = False
            for quantifier in semantics.quantifiers:
                if quantifier.slot_fillings[SLOT1] == object_node:
                    found_a_quantifier = True
                    for quantifier_template in self.number_learner.templates_for_concept(
                        quantifier.concept
                    ):
                        yield quantifier_template.instantiate(
                            {SLOT1: cur_string}
                        ).as_token_sequence()

            if not found_a_quantifier:
                raise RuntimeError(
                    f"Every object node should have a quantifier but could not find one "
                    f"for {object_node}"
                )

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

        for object_template in self.object_learner.templates_for_concept(
            object_node.concept
        ):

            object_string = object_template.instantiate(
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
                    cur_string = object_string
                    for attribute in attribute_combinations:
                        # we know, but mypy does not, that self.attribute_learner is not None
                        for (
                            attribute_template
                        ) in self.attribute_learner.templates_for_concept(  # type: ignore
                            attribute.concept
                        ):
                            for quantified in self._handle_quantifiers(
                                learner_semantics,
                                object_node,
                                attribute_template.instantiate(
                                    template_variable_to_filler={SLOT1: cur_string}
                                ).as_token_sequence(),
                            ):
                                yield quantified
            for quantified in self._handle_quantifiers(
                learner_semantics, object_node, object_string
            ):
                yield quantified

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


ENGLISH_THE_TEMPLATE = SurfaceTemplate(["the", SLOT1], language_mode=LanguageMode.ENGLISH)
