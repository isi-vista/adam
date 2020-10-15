import collections
import itertools
import logging
import typing
from itertools import chain, combinations
from pathlib import Path
from typing import Iterator, Mapping, Optional, Tuple, List, Dict

import graphviz
from attr import attrib, attrs
from attr.validators import instance_of, optional
from immutablecollections import immutabledict
from networkx import Graph, DiGraph

from adam.language import LinguisticDescription, TokenSequenceLinguisticDescription
from adam.language_specific.english import ENGLISH_BLOCK_DETERMINERS
from adam.learner import LearningExample, TopLevelLanguageLearner
from adam.learner.alignments import (
    LanguageConceptAlignment,
    LanguagePerceptionSemanticAlignment,
    PerceptionSemanticAlignment,
)
from adam.learner.attributes import SubsetAttributeLearnerNew
from adam.learner.functional_learner import FunctionalLearner
from adam.learner.generics import SimpleGenericsLearner
from adam.learner.language_mode import LanguageMode
from adam.learner.learner_utils import (
    get_classifier_for_string,
    get_slot_from_semantic_node,
)
from adam.learner.plurals import SubsetPluralLearnerNew
from adam.learner.surface_templates import MASS_NOUNS, SLOT1
from adam.learner.template_learner import TemplateLearner
from adam.ontology.phase1_ontology import PART_OF
from adam.perception import PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception.perception_graph import PerceptionGraph, PerceptionGraphPattern, Incrementer, NodePredicate, \
    AnyObjectPerception, ObjectSemanticNodePerceptionPredicate
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
)


class LanguageLearnerNew:
    def observe(
            self,
            learning_example: LearningExample[
                DevelopmentalPrimitivePerceptionFrame, LinguisticDescription
            ],
            offset: int = 0,
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
    functional_learner: Optional[FunctionalLearner] = attrib(
        validator=optional(instance_of(FunctionalLearner)), default=None
    )

    plural_learner: Optional[TemplateLearner] = attrib(
        validator=optional(instance_of(TemplateLearner)), default=None
    )

    generics_learner: Optional[SimpleGenericsLearner] = attrib(
        validator=optional(instance_of(SimpleGenericsLearner)), default=None
    )

    _max_attributes_per_word: int = attrib(validator=instance_of(int), default=3)

    _observation_num: int = attrib(init=False, default=0)
    _sub_learners: List[TemplateLearner] = attrib(init=False)

    potential_definiteness_markers: typing.Counter[str] = attrib(
        init=False, default=collections.Counter()
    )

    semantics_graph: DiGraph = attrib(init=False, default=DiGraph())
    concepts_to_patterns: Dict[Concept, PerceptionGraphPattern] = attrib(
        init=False, default=dict()
    )

    def observe(
            self,
            learning_example: LearningExample[
                DevelopmentalPrimitivePerceptionFrame, LinguisticDescription
            ],
            offset: int = 0,
    ) -> None:

        logging.info(
            "Observation %s: %s",
            self._observation_num + offset,
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
            self.plural_learner,
            self.relation_learner,
        ]:
            if sub_learner:
                # Currently we do not attempt to learn static things from dynamic situations
                # because the static learners do not know how to deal with the temporal
                # perception graph edge wrappers.
                # See https://github.com/isi-vista/adam/issues/792 .
                if not learning_example.perception.is_dynamic():
                    sub_learner.learn_from(current_learner_state, offset=offset)
                current_learner_state = sub_learner.enrich_during_learning(
                    current_learner_state
                )
                # Check definiteness after recognizing objects
                if sub_learner == self.object_learner:
                    self.learn_definiteness_markers(current_learner_state)

        if learning_example.perception.is_dynamic() and self.action_learner:
            self.action_learner.learn_from(current_learner_state)
            current_learner_state = self.action_learner.enrich_during_learning(
                current_learner_state
            )

            if self.functional_learner:
                self.functional_learner.learn_from(current_learner_state, offset=offset)

        # Engage generics learner if the utterance is indefinite
        if self.generics_learner and not self.is_definite(current_learner_state):
            # Lack of definiteness could me marking a generic statement
            # Check if the known descriptions match the utterance
            descs = self._linguistic_descriptions_from_semantics(
                current_learner_state.perception_semantic_alignment
            )
            # If the statement isn't a recognized sentence, run learner
            if not learning_example.linguistic_description.as_token_sequence() in [
                desc.as_token_sequence() for desc in descs
            ]:
                # Pass plural markers to generics before learning from a statement
                if isinstance(self.plural_learner, SubsetPluralLearnerNew):
                    self.generics_learner.plural_markers = list(  # pylint: disable=assigning-non-slot
                        self.plural_learner.potential_plural_markers.keys()
                    )
                self.generics_learner.learn_from(current_learner_state)

        # Update concept semantics
        self.update_concept_semantics(current_learner_state)

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
            self.plural_learner,
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

            if self.functional_learner:
                cur_description_state = self.functional_learner.enrich_during_description(
                    cur_description_state
                )
        return self._linguistic_descriptions_from_semantics(cur_description_state)

    def _linguistic_descriptions_from_semantics(
            self, description_state: PerceptionSemanticAlignment
    ) -> Mapping[LinguisticDescription, float]:

        learner_semantics = LearnerSemantics.from_nodes(
            description_state.semantic_nodes,
            concept_map=description_state.functional_concept_to_object_concept,
        )
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
            if classifier:
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
        else:
            return cur_string

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
                                yield self._add_determiners(
                                    object_node,
                                    attribute_template.instantiate(
                                        template_variable_to_filler={SLOT1: cur_string}
                                    ).as_token_sequence(),
                                )

                yield self._add_determiners(object_node, cur_string)

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
        return valid_sub_learners

    def learned_attribute_tokens(self) -> List[str]:
        attribute_tokens = []
        if self.attribute_learner and isinstance(
                self.attribute_learner, SubsetAttributeLearnerNew
        ):
            for template in self.attribute_learner.surface_template_to_concept.keys():
                for element in template.elements:
                    if isinstance(element, str):
                        attribute_tokens.append(element)
        return attribute_tokens

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

        for sub_learner in self._sub_learners:
            if isinstance(sub_learner, FunctionalLearner):
                continue
            for concept, pattern in sub_learner.concepts_to_patterns().items():
                self.concepts_to_patterns[concept] = pattern

    def get_semantics_with_patterns(self) -> DiGraph:
        complete_semantics_graph = self.semantics_graph.to_directed()
        for concept, pattern in self.concepts_to_patterns.items():
            pattern_graph = pattern._graph.to_directed()
            if concept not in complete_semantics_graph.nodes:
                print(concept, 'not found')
                continue
            if isinstance(concept, ObjectConcept):
                # Get root object perception of pattern
                potential_roots = set([n for n in pattern_graph.nodes if isinstance(n, AnyObjectPerception)])
                for u, v, data in pattern_graph.edges.data():
                    if isinstance(v, AnyObjectPerception) and isinstance(u, AnyObjectPerception):
                        if data['predicate'].relation_type == PART_OF and u in potential_roots:
                            potential_roots.remove(u)
                try:
                    root = list(potential_roots)[0]
                except:
                    continue
            else:
                # Many x s; red x s; x sits;
                potential_roots = set([n for n in pattern_graph.nodes if isinstance(n, ObjectSemanticNodePerceptionPredicate)])
                for u, v, data in pattern_graph.edges.data():
                    if isinstance(v, ObjectSemanticNodePerceptionPredicate) and v in potential_roots:
                        potential_roots.remove(v)
                try:
                    root = list(potential_roots)[0]
                except:
                    continue
            complete_semantics_graph.add_edge(concept, root, pattern=type(concept))
            for u,v,data in pattern_graph.edges.data():
                complete_semantics_graph.add_edge(u, v, **data)

        return complete_semantics_graph

    def render_semantics_to_file(  # pragma: no cover
            self, graph: Graph, graph_name: str, output_file: Path
    ) -> None:

        dot_graph = graphviz.Digraph(graph_name)
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
            edge_label = ' '.join([f'{k}={str(v)}' for k, v in data.items()])
            source_dot_node = semantics_nodes_to_dot_node_ids[source_node]
            target_dot_node = semantics_nodes_to_dot_node_ids[target_node]
            dot_graph.edge(source_dot_node, target_dot_node, edge_label)

        dot_graph.render(str(output_file))

    def to_dot_node(
            self, dot_graph: graphviz.Digraph, node: Concept, next_node_id: Incrementer
    ) -> str:
        label = node.dot_label() if isinstance(node, NodePredicate) else node.debug_string
        attributes = {"label": label, "style": "solid"}
        node_id = f"node-{next_node_id.value()}"
        next_node_id.increment()
        dot_graph.node(node_id, **attributes)
        return node_id
