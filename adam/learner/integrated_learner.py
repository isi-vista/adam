import collections
import itertools
import logging
from abc import ABC, abstractmethod
from itertools import chain, combinations
from pathlib import Path
from typing import (
    Iterator,
    Mapping,
    Optional,
    Tuple,
    List,
    Dict,
    Union,
    Counter,
    Set,
    DefaultDict,
    Generic,
    Sequence,
)
from uuid import uuid4

import graphviz
from attr import attrib, attrs
from attr.validators import instance_of, optional
from immutablecollections import immutabledict, immutableset
from networkx import Graph, DiGraph

from adam.language import (
    LinguisticDescription,
    TokenSequenceLinguisticDescription,
    LinguisticDescriptionT,
)
from adam.language_specific.english import ENGLISH_BLOCK_DETERMINERS
from adam.learner import (
    LearningExample,
    TopLevelLanguageLearner,
    TopLevelLanguageLearnerDescribeReturn,
)
from adam.learner.affordances import MappingAffordanceLearner
from adam.learner.alignments import (
    LanguageConceptAlignment,
    LanguagePerceptionSemanticAlignment,
    PerceptionSemanticAlignment,
)
from adam.learner.attributes import SubsetAttributeLearner
from adam.learner.functional_learner import FunctionalLearner
from adam.learner.generics import SimpleGenericsLearner
from adam.learner.language_mode import LanguageMode
from adam.learner.learner_utils import (
    get_classifier_for_string,
    get_slot_from_semantic_node,
)
from adam.learner.plurals import SubsetPluralLearner
from adam.learner.surface_templates import MASS_NOUNS, SLOT1
from adam.learner.template_learner import TemplateLearner
from adam.ontology.phase1_ontology import PART_OF
from adam.perception import (
    PerceptualRepresentation,
    PerceptionT,
)
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception.perception_graph import (
    PerceptionGraph,
    PerceptionGraphPattern,
    Incrementer,
    NodePredicate,
    AnyObjectPerception,
    ObjectSemanticNodePerceptionPredicate,
    get_features_from_semantic_node,
    GraphLogger,
)
from adam.perception.visual_perception import VisualPerceptionFrame
from adam.semantics import (
    ActionSemanticNode,
    ObjectSemanticNode,
    RelationSemanticNode,
    GROUND_OBJECT_CONCEPT,
    LearnerSemantics,
    FunctionalObjectConcept,
    ObjectConcept,
    AttributeSemanticNode,
    Concept,
    KindConcept,
    SemanticNode,
    AffordanceSemanticNode,
)


@attrs
class IntegratedTemplateLearner(
    Generic[PerceptionT, LinguisticDescriptionT],
    TopLevelLanguageLearner[PerceptionT, LinguisticDescriptionT],
    ABC,
):
    """
    A `TopLevelLanguageLearner` which uses template-based syntax to learn objects, attributes, relations,
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
    functional_learner: Optional[FunctionalLearner] = attrib(
        validator=optional(instance_of(FunctionalLearner)), default=None
    )
    plural_learner: Optional[TemplateLearner] = attrib(
        validator=optional(instance_of(TemplateLearner)), default=None
    )
    generics_learner: Optional[SimpleGenericsLearner] = attrib(
        validator=optional(instance_of(SimpleGenericsLearner)), default=None
    )
    affordance_learner: Optional[TemplateLearner] = attrib(
        validator=optional(instance_of(TemplateLearner)), default=None
    )
    mapping_affordance_learner: Optional[TemplateLearner] = attrib(
        validator=optional(instance_of(MappingAffordanceLearner)), default=None
    )

    _max_attributes_per_word: int = attrib(validator=instance_of(int), default=3)

    _observation_num: int = attrib(init=False, default=0)
    _sub_learners: List[TemplateLearner] = attrib(init=False)

    # Used to control if runtime errors should be escalated to cause a program crash
    # Set to false if errors should not be suppressed
    _suppress_error: bool = attrib(kw_only=True, default=True)

    potential_definiteness_markers: Counter[str] = attrib(
        init=False, factory=collections.Counter
    )

    semantics_graph: DiGraph = attrib(init=False, factory=DiGraph)
    concepts_to_patterns: Dict[Concept, PerceptionGraphPattern] = attrib(
        init=False, factory=dict
    )

    def observe_common(
        self,
        current_learner_state: LanguagePerceptionSemanticAlignment,
        linguistic_description: LinguisticDescription,
        is_dynamic: bool,
        offset: int = 0,
        *,
        debug_perception_graph_logger: Optional[GraphLogger] = None,
    ):
        logging.info(
            "Observation %s: %s",
            self._observation_num + offset,
            linguistic_description.as_token_string(),
        )

        if debug_perception_graph_logger:
            debug_perception_graph_logger.log_graph(
                current_learner_state.perception_semantic_alignment.perception_graph,
                logging.DEBUG,
                f"Logging perception graph from observe: {current_learner_state.language_concept_alignment.language.as_token_string()}",
                graph_name=f"Observation {self._observation_num}: {current_learner_state.language_concept_alignment.language.as_token_string()}",
            )

        self._observation_num += 1

        # The affordance learner 'learns' unnamed features that correspond to the ability of an object to participate
        # in an action. As this learner doesn't produce verbalized semantics but instead augments the perception
        # graph with features it recognizes we need these additional features at the start of the scene to enhance
        # the other learners.
        if self.affordance_learner:
            current_learner_state = self.affordance_learner.enrich_during_learning(
                current_learner_state
            )

        # We iteratively let each "layer" of semantic analysis attempt
        # to learn from the perception,
        # and then to annotate the perception with any semantic alignments it knows.
        for sub_learner in [
            self.object_learner,
            self.attribute_learner,
            self.plural_learner,
            self.relation_learner,
        ]:
            if sub_learner:
                # Currently we do not attempt to learn static things from dynamic situations
                # because the static learners do not know how to deal with the temporal
                # perception graph edge wrappers.
                # See https://github.com/isi-vista/adam/issues/792 .
                if not is_dynamic:
                    # For more details on the try/excepts below
                    # See: https://github.com/isi-vista/adam/issues/1008
                    try:
                        sub_learner.learn_from(current_learner_state, offset=offset)
                    except (RuntimeError, KeyError) as e:
                        logging.warning(
                            f"Sub_learner ({sub_learner}) was unable to learn from instance number {self._observation_num}.\n"
                            f"Instance: {current_learner_state}.\n"
                            f"Full Error Information: {e}"
                        )
                        if not self._suppress_error:
                            raise e
                current_learner_state = sub_learner.enrich_during_learning(
                    current_learner_state
                )
                # Check definiteness after recognizing objects
                if sub_learner == self.object_learner:
                    self.learn_definiteness_markers(current_learner_state)

        if is_dynamic and self.action_learner:
            try:
                self.action_learner.learn_from(current_learner_state)
            except (RuntimeError, KeyError) as e:
                logging.warning(
                    f"Action Learner ({self.action_learner}) was unable to learn from instance number {self._observation_num}.\n"
                    f"Instance: {current_learner_state}.\n"
                    f"Full Error Information: {e}"
                )
                if not self._suppress_error:
                    raise e

            current_learner_state = self.action_learner.enrich_during_learning(
                current_learner_state
            )

            if self.functional_learner:
                self.functional_learner.learn_from(current_learner_state, offset=offset)

            if self.affordance_learner:
                self.affordance_learner.learn_from(current_learner_state)

                # We acknowledge that calling this a second time can duplicate nodes
                # As no learner processes the output after this we can handle duplication
                # In any tasks using AffordanceSemanticNodes downstream, but we need to run this
                # To get any newly learned/refined affordances
                current_learner_state = self.affordance_learner.enrich_during_learning(
                    current_learner_state
                )

                self._backpropagate_affordance(current_learner_state)

            if self.mapping_affordance_learner:
                self.mapping_affordance_learner.learn_from(current_learner_state)

        # Engage generics learner if the utterance is indefinite
        if self.generics_learner and not self.is_definite(current_learner_state):
            learner_semantics = LearnerSemantics.from_nodes(
                current_learner_state.perception_semantic_alignment.semantic_nodes,
                concept_map=current_learner_state.perception_semantic_alignment.functional_concept_to_object_concept,
            )
            # Lack of definiteness could be marking a generic statement
            # Check if the known descriptions match the utterance
            description, _ = self._linguistic_descriptions_from_semantics(
                learner_semantics
            )
            # If the statement isn't a recognized sentence, run learner
            if linguistic_description.as_token_sequence() not in set(
                desc.as_token_sequence() for desc in description
            ):
                # Pass plural markers to generics before learning from a statement
                if isinstance(self.plural_learner, SubsetPluralLearner):
                    self.generics_learner.plural_markers = (
                        list(  # pylint: disable=assigning-non-slot
                            self.plural_learner.potential_plural_markers.keys()
                        )
                    )
                self.generics_learner.learn_from(current_learner_state)

        # Update concept semantics
        self.update_concept_semantics(current_learner_state)

    @abstractmethod
    def observe(
        self,
        learning_example: LearningExample[PerceptionT, LinguisticDescription],
        offset: int = 0,
        *,
        debug_perception_graph_logger: Optional[GraphLogger] = None,
    ) -> None:
        raise NotImplementedError

    def describe_common(
        self,
        cur_description_state: PerceptionSemanticAlignment,
        is_dynamic: bool,
        *,
        debug_perception_graph_logger: Optional[GraphLogger] = None,
    ) -> TopLevelLanguageLearnerDescribeReturn:
        if debug_perception_graph_logger:
            graph_name = str(uuid4())
            debug_perception_graph_logger.log_graph(
                cur_description_state.perception_graph,
                logging.DEBUG,
                f"Logging perception graph from describe: {graph_name}",
                graph_name=graph_name,
            )

        # The affordance learner, if we have one, runs decode first as it adds additional features we want the
        # other learners to be able to learn from
        if self.affordance_learner:
            cur_description_state = self.affordance_learner.enrich_during_description(
                cur_description_state
            )

        for sub_learner in [
            self.object_learner,
            self.attribute_learner,
            self.plural_learner,
            self.relation_learner,
        ]:
            if sub_learner:
                cur_description_state = sub_learner.enrich_during_description(
                    cur_description_state
                )

        if is_dynamic and self.action_learner:
            cur_description_state = self.action_learner.enrich_during_description(
                cur_description_state
            )

            if self.functional_learner:
                cur_description_state = self.functional_learner.enrich_during_description(
                    cur_description_state
                )

        learner_semantics = LearnerSemantics.from_nodes(
            cur_description_state.semantic_nodes,
            concept_map=cur_description_state.functional_concept_to_object_concept,
        )
        (
            linguistic_to_score,
            semantic_to_linguistic,
        ) = self._linguistic_descriptions_from_semantics(learner_semantics)
        return TopLevelLanguageLearnerDescribeReturn(
            semantics_to_descriptions=semantic_to_linguistic,
            description_to_confidence=linguistic_to_score,
            semantics_to_feature_strs=self._visual_features_from_semantics(
                cur_description_state
            ),
        )

    @abstractmethod
    def describe(
        self,
        perception: PerceptualRepresentation[PerceptionT],
        *,
        debug_perception_graph_logger: Optional[GraphLogger] = None,
    ) -> TopLevelLanguageLearnerDescribeReturn:
        raise NotImplementedError()

    def _linguistic_descriptions_from_semantics(
        self, learner_semantics: LearnerSemantics
    ) -> Tuple[
        Mapping[LinguisticDescription, float],
        Mapping[SemanticNode, LinguisticDescription],
    ]:

        semantics_to_description: List[Tuple[SemanticNode, LinguisticDescription]] = []
        description_to_score: List[Tuple[LinguisticDescription, float]] = []
        if self.action_learner:
            for action in learner_semantics.actions:
                # ensure we have some way of expressing this action
                if self.action_learner.templates_for_concept(action.concept):
                    for action_tokens in self._instantiate_action(
                        action, learner_semantics
                    ):
                        token_description = TokenSequenceLinguisticDescription(
                            action_tokens
                        )
                        semantics_to_description.append((action, token_description))
                        description_to_score.append(
                            (
                                token_description,
                                action.confidence,
                            )
                        )

        if self.relation_learner:
            for relation in learner_semantics.relations:
                # ensure we have some way of expressing this relation
                if self.relation_learner.templates_for_concept(relation.concept):
                    for relation_tokens in self._instantiate_relation(
                        relation, learner_semantics
                    ):
                        token_description = TokenSequenceLinguisticDescription(
                            relation_tokens
                        )
                        semantics_to_description.append((relation, token_description))
                        description_to_score.append(
                            (
                                token_description,
                                relation.confidence,
                            )
                        )

        for object_ in learner_semantics.objects:
            # ensure we have some way of expressing this object
            if self.object_learner.templates_for_concept(object_.concept):
                for object_tokens in self._instantiate_object(object_, learner_semantics):
                    token_description = TokenSequenceLinguisticDescription(object_tokens)
                    semantics_to_description.append((object_, token_description))
                    description_to_score.append(
                        (
                            token_description,
                            object_.confidence,
                        )
                    )

        return immutabledict(description_to_score), immutabledict(
            semantics_to_description
        )

    def _instantiate_object(
        self, object_node: ObjectSemanticNode, learner_semantics: LearnerSemantics
    ) -> Iterator[Tuple[str, ...]]:
        for learner in [self.attribute_learner, self.plural_learner]:
            # For now, we assume the order in which modifiers is expressed is arbitrary.
            attributes_we_can_express = (
                [
                    attribute
                    for attribute in learner_semantics.objects_to_attributes[object_node]
                    if learner.templates_for_concept(attribute.concept)
                ]
                if learner
                else []
            )
            # We currently cannot deal with relations that modify objects embedded in other expressions.
            # See https://github.com/isi-vista/adam/issues/794 .
            # relations_for_object = learner_semantics.objects_to_relation_in_slot1[object_node]

            if (
                isinstance(object_node.concept, FunctionalObjectConcept)
                and object_node.concept
                in learner_semantics.functional_concept_to_object_concept.keys()
            ):
                concept = learner_semantics.functional_concept_to_object_concept[
                    object_node.concept
                ]
            else:
                concept = object_node.concept

            for template in self.object_learner.templates_for_concept(concept):

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
                            ) in learner.templates_for_concept(  # type: ignore
                                attribute.concept
                            ):
                                yield self.add_determiners(
                                    object_node,
                                    attribute_template.instantiate(
                                        template_variable_to_filler={SLOT1: cur_string},
                                        attribute_template=True,
                                    ).as_token_sequence(),
                                )

                yield self.add_determiners(object_node, cur_string)

    def _instantiate_relation(
        self, relation_node: RelationSemanticNode, learner_semantics: LearnerSemantics
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
        self, action_node: ActionSemanticNode, learner_semantics: LearnerSemantics
    ) -> Iterator[Tuple[str, ...]]:
        if not self.action_learner:
            raise RuntimeError("Cannot instantiate an action without an action learner")

        for action_template in self.action_learner.templates_for_concept(
            action_node.concept
        ):
            # TODO: Handle instantiate objects returning no result from functional learner
            # If that happens we should break from instantiating this utterance
            slots_to_instantiations = {
                slot: list(self._instantiate_object(slot_filler, learner_semantics))
                for (slot, slot_filler) in action_node.slot_fillings.items()
            }
            slot_order = tuple(slots_to_instantiations.keys())

            all_possible_slot_fillings = itertools.product(
                *slots_to_instantiations.values()
            )
            for possible_slot_filling in all_possible_slot_fillings:
                yield action_template.instantiate(
                    immutabledict(zip(slot_order, possible_slot_filling))
                ).as_token_sequence()

    def _visual_features_from_semantics(
        self,
        current_description_state: PerceptionSemanticAlignment,
    ) -> Mapping[SemanticNode, Sequence[str]]:
        return immutabledict(
            (
                node,
                get_features_from_semantic_node(
                    node, current_description_state.perception_graph
                ),
            )
            for node in current_description_state.semantic_nodes
        )

    def log_hypotheses(self, log_output_path: Path) -> None:
        for sub_learner in self._sub_learners:
            sub_learner.log_hypotheses(log_output_path)

    def render_semantics_to_file(  # pragma: no cover
        self, graph: Graph, graph_name: str, output_file: Path
    ) -> None:

        dot_graph = graphviz.Graph(graph_name)
        dot_graph.attr(rankdir="LR")
        # combine parallel edges to cut down on clutter
        dot_graph.attr(concentrate="true")

        next_node_id = Incrementer()

        # add all nodes to the graph
        semantics_nodes_to_dot_node_ids = {
            semantics_node: self.to_dot_node(dot_graph, semantics_node, next_node_id)
            for semantics_node in graph.nodes
        }

        for (source_node, target_node, data) in graph.edges.data():
            edge_label = " ".join([f"{k}={str(v)}" for k, v in data.items()])
            source_dot_node = semantics_nodes_to_dot_node_ids[source_node]
            target_dot_node = semantics_nodes_to_dot_node_ids[target_node]
            dot_graph.edge(source_dot_node, target_dot_node, edge_label)

        dot_graph.render(str(output_file))

    def to_dot_node(  # pragma: no cover
        self,
        dot_graph: graphviz.Graph,
        node: Union[Concept, NodePredicate],
        next_node_id: Incrementer,
    ) -> str:
        label = node.dot_label() if isinstance(node, NodePredicate) else node.debug_string
        attributes = {"label": label, "style": "solid"}
        node_id = f"node-{next_node_id.value()}"
        next_node_id.increment()
        dot_graph.node(node_id, **attributes)
        return node_id

    @abstractmethod
    def add_determiners(
        self, object_node: ObjectSemanticNode, cur_string: Tuple[str, ...]
    ) -> Tuple[str, ...]:
        """Function to add determiners strings to objects."""
        raise NotImplementedError(
            "add_determiners is not implemented in the abstract IntegratedLearner"
        )

    @abstractmethod
    def extract_perception_graph(
        self, perception: PerceptualRepresentation[PerceptionT]
    ) -> PerceptionGraph:
        raise NotImplementedError(
            "_extract_perception_graph is not implemented in the abstract IntegratedLearner"
        )

    @_sub_learners.default
    def _init_sub_learners(self) -> List[TemplateLearner]:
        valid_sub_learners = []
        if self.object_learner:
            valid_sub_learners.append(self.object_learner)
        if self.attribute_learner:
            valid_sub_learners.append(self.attribute_learner)
        if self.plural_learner:
            valid_sub_learners.append(self.plural_learner)
        if self.relation_learner:
            valid_sub_learners.append(self.relation_learner)
        if self.action_learner:
            valid_sub_learners.append(self.action_learner)
        if self.functional_learner:
            valid_sub_learners.append(self.functional_learner)
        if self.generics_learner:
            valid_sub_learners.append(self.generics_learner)
        if self.affordance_learner:
            valid_sub_learners.append(self.affordance_learner)
        if self.mapping_affordance_learner:
            valid_sub_learners.append(self.mapping_affordance_learner)
        return valid_sub_learners

    # TODO: Extract semantics learning into its own sub-learner
    # https://github.com/isi-vista/adam/issues/1050
    def learn_definiteness_markers(self, current_learner_state):
        # Helper method to learn definiteness markers from objects
        sequence = (
            current_learner_state.language_concept_alignment.language.as_token_sequence()
        )
        for (
            _,
            span,
        ) in (
            current_learner_state.language_concept_alignment.node_to_language_span.items()
        ):
            potential_marker = None
            # Special case: Could be a object with a determiner included in the span
            if span.end - span.start > 1:
                potential_marker = sequence[span.start]
            # Standard case - add token preceeding the noun
            elif span.start > 0:
                # If it's an attribute, look for the token preceeding the attribute
                if (
                    sequence[span.start - 1] in self.learned_attribute_tokens()
                    and span.start > 1
                ):
                    potential_marker = sequence[span.start - 2]
                else:
                    potential_marker = sequence[span.start - 1]
            # If we detected a marker, we update the set
            if potential_marker:
                self.potential_definiteness_markers.update([potential_marker])

    def is_definite(self, current_learner_state: LanguagePerceptionSemanticAlignment):
        # Check if it contains any potential definiteness marker
        sequence = (
            current_learner_state.language_concept_alignment.language.as_token_sequence()
        )
        definite_marker_matches = []

        markers = list(self.potential_definiteness_markers.keys())
        # Could instead use the following for most_common n:
        # most_common = [s for s, _ in self.potential_definiteness_markers.most_common(3)]

        # Remove attributes:
        for token in self.learned_attribute_tokens():
            if token in markers:
                markers.remove(token)

        # Check if any token in the sequence is a potential definiteness marker:
        for (
            node,
            span,
        ) in (
            current_learner_state.language_concept_alignment.node_to_language_span.items()
        ):
            if isinstance(node, ObjectSemanticNode):
                # Special case: Could be a object with a determiner included in the span
                if span.end - span.start > 1:
                    definite_marker_matches.append(sequence[span.start] in markers)
                # Standard case - add token preceding the noun
                elif span.start > 0:
                    # If it's an attribute, look for the token preceeding the attribute
                    if (
                        sequence[span.start - 1] in self.learned_attribute_tokens()
                        and span.start > 1
                    ):
                        definite_marker_matches.append(
                            sequence[span.start - 2] in markers
                        )
                    else:
                        definite_marker_matches.append(
                            sequence[span.start - 1] in markers
                        )
        return any(definite_marker_matches)

    def learned_attribute_tokens(self) -> List[str]:
        attribute_tokens = []
        if self.attribute_learner and isinstance(
            self.attribute_learner, SubsetAttributeLearner
        ):
            for template in self.attribute_learner.surface_template_to_concept.keys():
                for element in template.elements:
                    if isinstance(element, str):
                        attribute_tokens.append(element)
        return attribute_tokens

    def update_concept_semantics(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ):
        recognized_semantic_nodes = list(
            language_perception_semantic_alignment.perception_semantic_alignment.semantic_nodes
        )
        span = (
            language_perception_semantic_alignment.language_concept_alignment.node_to_language_span
        )

        # Get all action and attribute nodes
        # concepts = [n.concept for n in recognized_semantic_nodes]
        relevant_nodes = [
            n
            for n in recognized_semantic_nodes
            if isinstance(n, ActionSemanticNode) or isinstance(n, AttributeSemanticNode)
        ]

        # Get object concepts that are in the utterance
        recognized_object_concepts: List[ObjectConcept] = []
        for node in recognized_semantic_nodes:
            if isinstance(node, ObjectSemanticNode) and node in span:
                recognized_object_concepts.append(node.concept)

        # Update association strength for each object - other concept pair
        for object_concept in recognized_object_concepts:
            for node in relevant_nodes:
                other_concept = node.concept
                slot = get_slot_from_semantic_node(object_concept, node)
                if slot:
                    # Update if the association exists, otherwise create
                    if (
                        self.semantics_graph.has_edge(object_concept, other_concept)
                        and slot
                        == self.semantics_graph[object_concept][other_concept]["slot"]
                    ):
                        old_score = self.semantics_graph[object_concept][other_concept][
                            "weight"
                        ]
                        new_score = old_score + (1.0 - old_score) * 0.2
                        self.semantics_graph[object_concept][other_concept][
                            "weight"
                        ] = new_score
                    else:
                        self.semantics_graph.add_edge(
                            object_concept, other_concept, slot=slot, weight=0.2
                        )

        # For each object - other concept pair learner through generics, set a high association strength
        if self.generics_learner:
            for (
                obj_con,
                other_concepts_and_object_slots,
            ) in self.generics_learner.learned_representations.values():
                for other_con, slot in other_concepts_and_object_slots:
                    self.semantics_graph.add_edge(
                        obj_con, other_con, slot=slot, weight=1.0
                    )
                    # if the object is a wug - a new object heard through generics
                    if (
                        obj_con not in self.object_learner.concepts_to_patterns()
                        and isinstance(other_con, KindConcept)
                    ):
                        # Create a representation of the kind using association of its neighbors
                        kind_neighbor_associations: DefaultDict[
                            Tuple[Concept, str], float
                        ] = collections.defaultdict(float)
                        for member_of_kind in self.semantics_graph.predecessors(
                            other_con
                        ):
                            if member_of_kind == obj_con:
                                continue
                            # We want this node to share the properties other members of that kind
                            for n in self.semantics_graph.neighbors(member_of_kind):
                                if isinstance(n, KindConcept):
                                    continue
                                data = self.semantics_graph.get_edge_data(
                                    member_of_kind, n
                                )
                                kind_neighbor_slot = data["slot"]
                                kind_neighbor_strength = data["weight"]
                                kind_neighbor_associations[
                                    (n, kind_neighbor_slot)
                                ] += kind_neighbor_strength
                        if not kind_neighbor_associations.values():
                            continue
                        coefficient = 1.0 / max(kind_neighbor_associations.values())
                        for (
                            (associated_concept, associated_slot),
                            strength,
                        ) in kind_neighbor_associations.items():
                            self.semantics_graph.add_edge(
                                obj_con,
                                associated_concept,
                                slot=associated_slot,
                                weight=coefficient * strength,
                            )

        for sub_learner in self._sub_learners:
            if isinstance(sub_learner, FunctionalLearner):
                continue
            for concept, pattern in sub_learner.concepts_to_patterns().items():
                self.concepts_to_patterns[concept] = pattern

    def get_semantics_with_patterns(self) -> DiGraph:
        complete_semantics_graph = self.semantics_graph.to_directed()
        for concept, pattern in self.concepts_to_patterns.items():
            if concept not in complete_semantics_graph.nodes:
                print(concept, "not found")
                continue
            pattern_graph = pattern.copy_as_digraph().to_directed()
            if isinstance(concept, ObjectConcept):
                # Get root object perception of pattern
                potential_roots: Set[
                    Union[AnyObjectPerception, ObjectSemanticNodePerceptionPredicate]
                ] = set(
                    [n for n in pattern_graph.nodes if isinstance(n, AnyObjectPerception)]
                )
                for u, v, data in pattern_graph.edges.data():
                    if isinstance(v, AnyObjectPerception) and isinstance(
                        u, AnyObjectPerception
                    ):
                        if (
                            data["predicate"].relation_type == PART_OF
                            and u in potential_roots
                        ):
                            potential_roots.remove(u)
                try:
                    root = list(potential_roots)[0]
                except Exception as e:  # pylint: disable=broad-except
                    logging.exception(e)
                    continue
            else:
                # Many x s; red x s; x sits;
                potential_roots = set(
                    [
                        n
                        for n in pattern_graph.nodes
                        if isinstance(n, ObjectSemanticNodePerceptionPredicate)
                    ]
                )
                for u, v, data in pattern_graph.edges.data():
                    if (
                        isinstance(v, ObjectSemanticNodePerceptionPredicate)
                        and v in potential_roots
                    ):
                        potential_roots.remove(v)
                try:
                    root = list(potential_roots)[0]
                except Exception as e:  # pylint: disable=broad-except
                    logging.exception(e)
                    continue
            complete_semantics_graph.add_edge(concept, root, pattern=type(concept))
            for u, v, data in pattern_graph.edges.data():
                complete_semantics_graph.add_edge(u, v, **data)

        return complete_semantics_graph

    def _backpropagate_affordance(
        self, current_learner_state: LanguagePerceptionSemanticAlignment
    ) -> None:
        for node in current_learner_state.perception_semantic_alignment.semantic_nodes:
            if not isinstance(node, AffordanceSemanticNode):
                continue

            self.object_learner.process_affordance(
                node.slot_fillings[SLOT1].concept, node
            )


@attrs
class SymbolicIntegratedTemplateLearner(
    IntegratedTemplateLearner[
        DevelopmentalPrimitivePerceptionFrame, TokenSequenceLinguisticDescription
    ]
):
    """
    An `IntegratedTemplateLearner` which uses template-based syntax to learn objects, attributes, relations,
    and actions all at once over a symbolic perception space.
    """

    def observe(
        self,
        learning_example: LearningExample[
            DevelopmentalPrimitivePerceptionFrame, LinguisticDescription
        ],
        offset: int = 0,
        *,
        debug_perception_graph_logger: Optional[GraphLogger] = None,
    ) -> None:

        # We need to track the alignment between perceived objects
        # and portions of the input language, so internally we operate over
        # LanguageAlignedPerceptions.
        current_learner_state = LanguagePerceptionSemanticAlignment(
            language_concept_alignment=LanguageConceptAlignment.create_unaligned(
                language=learning_example.linguistic_description
            ),
            perception_semantic_alignment=PerceptionSemanticAlignment(
                perception_graph=self.extract_perception_graph(
                    learning_example.perception
                ),
                semantic_nodes=immutableset(),
            ),
        )

        self.observe_common(
            current_learner_state,
            learning_example.linguistic_description,
            learning_example.perception.is_dynamic(),
            offset,
            debug_perception_graph_logger=debug_perception_graph_logger,
        )

    def describe(
        self,
        perception: PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame],
        *,
        debug_perception_graph_logger: Optional[GraphLogger] = None,
    ) -> TopLevelLanguageLearnerDescribeReturn:
        cur_description_state = PerceptionSemanticAlignment.create_unaligned(
            self.extract_perception_graph(perception)
        )

        return self.describe_common(
            cur_description_state,
            perception.is_dynamic(),
            debug_perception_graph_logger=debug_perception_graph_logger,
        )

    def add_determiners(
        self, object_node: ObjectSemanticNode, cur_string: Tuple[str, ...]
    ) -> Tuple[str, ...]:
        # handle Chinese Classifiers by casing on the words -- this is hackish
        if (
            self.object_learner._language_mode  # pylint: disable=protected-access
            == LanguageMode.CHINESE
        ):
            if self.plural_learner:
                return tuple([token for token in cur_string if token[:3] != "yi1"])
            # specially handle the case of my and your in Chinese since these organize the classifier and attribute differently
            if cur_string[0] in ["ni3 de", "wo3 de"] and len(cur_string) > 1:
                my_your_classifier = get_classifier_for_string(cur_string[1])
                if my_your_classifier:
                    return tuple(
                        chain((cur_string[0], my_your_classifier), cur_string[1:])
                    )
                else:
                    return cur_string
            # get the classifier and add it to the language
            classifier = get_classifier_for_string(cur_string[-1])
            # if the classifier was already hypothesized by the relation learner
            if classifier and cur_string[0][:4] != "yi1_":
                return tuple(chain((classifier,), cur_string))
            else:
                return cur_string

        # handle English determiners
        elif (
            self.object_learner._language_mode  # pylint: disable=protected-access
            == LanguageMode.ENGLISH
        ):
            # If plural, we want to strip any "a" that might preceed a noun after "many" or "two"
            if self.plural_learner:
                if "a" in cur_string:
                    a_position = cur_string.index("a")
                    if a_position > 0 and cur_string[a_position - 1] in ["many", "two"]:
                        return tuple(
                            [
                                token
                                for i, token in enumerate(cur_string)
                                if i != a_position
                            ]
                        )
            # English-specific hack to deal with us not understanding determiners:
            # https://github.com/isi-vista/adam/issues/498
            # The "is lower" check is a hack to block adding a determiner to proper names.
            # Ground is a specific thing so we special case this to be assigned
            if (
                object_node.concept.debug_string not in MASS_NOUNS
                and object_node.concept.debug_string.islower()
                and not cur_string[0] in ENGLISH_BLOCK_DETERMINERS
            ):
                if object_node.concept == GROUND_OBJECT_CONCEPT:
                    return tuple(chain(("the",), cur_string))
                else:
                    return tuple(chain(("a",), cur_string))
            else:
                return cur_string
        else:
            return cur_string

    def extract_perception_graph(
        self, perception: PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame]
    ) -> PerceptionGraph:
        if perception.is_dynamic():
            return PerceptionGraph.from_dynamic_perceptual_representation(perception)
        else:
            return PerceptionGraph.from_frame(perception.frames[0])


@attrs
class SimulatedIntegratedTemplateLearner(
    IntegratedTemplateLearner[VisualPerceptionFrame, TokenSequenceLinguisticDescription]
):
    """
    An `IntegratedTemplateLearner` which uses template-based syntax to learn objects, attributes, relations,
    and actions all at once over a simulated perception space.
    """

    def observe(
        self,
        learning_example: LearningExample[VisualPerceptionFrame, LinguisticDescription],
        offset: int = 0,
        *,
        debug_perception_graph_logger: Optional[GraphLogger] = None,
    ) -> None:

        # We need to track the alignment between perceived objects
        # and portions of the input language, so internally we operate over
        # LanguageAlignedPerceptions.
        current_learner_state = LanguagePerceptionSemanticAlignment(
            language_concept_alignment=LanguageConceptAlignment.create_unaligned(
                language=learning_example.linguistic_description
            ),
            perception_semantic_alignment=PerceptionSemanticAlignment(
                perception_graph=self.extract_perception_graph(
                    learning_example.perception
                ),
                semantic_nodes=immutableset(),
            ),
        )

        self.observe_common(
            current_learner_state,
            learning_example.linguistic_description,
            learning_example.perception.is_dynamic(),
            offset,
            debug_perception_graph_logger=debug_perception_graph_logger,
        )

    def describe(
        self,
        perception: PerceptualRepresentation[VisualPerceptionFrame],
        *,
        debug_perception_graph_logger: Optional[GraphLogger] = None,
    ) -> TopLevelLanguageLearnerDescribeReturn:
        cur_description_state = PerceptionSemanticAlignment.create_unaligned(
            self.extract_perception_graph(perception)
        )

        return self.describe_common(
            cur_description_state,
            perception.is_dynamic(),
            debug_perception_graph_logger=debug_perception_graph_logger,
        )

    def add_determiners(
        self, object_node: ObjectSemanticNode, cur_string: Tuple[str, ...]
    ) -> Tuple[str, ...]:
        # handle Chinese Classifiers by casing on the words -- this is hackish
        if (
            self.object_learner._language_mode  # pylint: disable=protected-access
            == LanguageMode.CHINESE
        ):
            if self.plural_learner:
                return tuple([token for token in cur_string if token[:3] != "yi1"])
            # specially handle the case of my and your in Chinese since these organize the classifier and attribute differently
            if cur_string[0] in ["ni3 de", "wo3 de"] and len(cur_string) > 1:
                my_your_classifier = get_classifier_for_string(cur_string[1])
                if my_your_classifier:
                    return tuple(
                        chain((cur_string[0], my_your_classifier), cur_string[1:])
                    )
                else:
                    return cur_string
            # get the classifier and add it to the language
            classifier = get_classifier_for_string(cur_string[-1])
            # if the classifier was already hypothesized by the relation learner
            if classifier and cur_string[0][:4] != "yi1_":
                return tuple(chain((classifier,), cur_string))
            else:
                return cur_string

        # handle English determiners
        elif (
            self.object_learner._language_mode  # pylint: disable=protected-access
            == LanguageMode.ENGLISH
        ):
            # If plural, we want to strip any "a" that might preceed a noun after "many" or "two"
            if self.plural_learner:
                if "a" in cur_string:
                    a_position = cur_string.index("a")
                    if a_position > 0 and cur_string[a_position - 1] in ["many", "two"]:
                        return tuple(
                            [
                                token
                                for i, token in enumerate(cur_string)
                                if i != a_position
                            ]
                        )
            # English-specific hack to deal with us not understanding determiners:
            # https://github.com/isi-vista/adam/issues/498
            # The "is lower" check is a hack to block adding a determiner to proper names.
            # Ground is a specific thing so we special case this to be assigned
            if (
                object_node.concept.debug_string not in MASS_NOUNS
                and object_node.concept.debug_string.islower()
                and not cur_string[0] in ENGLISH_BLOCK_DETERMINERS
            ):
                if object_node.concept == GROUND_OBJECT_CONCEPT:
                    return tuple(chain(("the",), cur_string))
                else:
                    return tuple(chain(("a",), cur_string))
            else:
                return cur_string
        else:
            return cur_string

    def extract_perception_graph(
        self, perception: PerceptualRepresentation[VisualPerceptionFrame]
    ) -> PerceptionGraph:
        if perception.is_dynamic():
            return PerceptionGraph.from_dynamic_simulated_perception_frame(perception)
        else:
            return PerceptionGraph.from_simulated_frame(perception.frames[0])
