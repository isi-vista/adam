import logging
from typing import Dict, Generic, List, Mapping, Optional, Tuple, cast

from attr.validators import instance_of
from more_itertools import flatten
from networkx import DiGraph, all_shortest_paths

from adam.language import LinguisticDescription
from adam.learner import LanguageLearner, LearningExample
from adam.learner.object_recognizer import ObjectRecognizer
from adam.learner.preposition_pattern import PerceptionGraphTemplate, SLOT1, SLOT2
from adam.learner.surface_templates import SurfaceTemplate, SurfaceTemplateVariable
from adam.ontology.ontology import Ontology
from adam.perception import ObjectPerception, PerceptionT, PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception.perception_graph import (
    GraphLogger,
    LanguageAlignedPerception,
    MatchedObjectNode,
    MatchedObjectPerceptionPredicate,
    PerceptionGraph,
    PerceptionGraphNode,
    _graph_node_order,
)
from adam.utils.networkx_utils import digraph_with_nodes_sorted_by, subgraph
from attr import Factory, attrib, attrs
from immutablecollections import ImmutableDict, ImmutableSet, immutabledict, immutableset

"""
This is a surface string pattern for a preposition. 
It should contain the strings MODIFIED and GROUND as stand-ins for the particular words
a preposition may be used with. For example, "MODIFIED on a GROUND". 
"""


@attrs
class PrepositionSubsetLanguageLearner(
    Generic[PerceptionT], LanguageLearner[PerceptionT, LinguisticDescription]
):
    _surface_template_to_preposition_pattern: Dict[
        SurfaceTemplate, PerceptionGraphTemplate
    ] = attrib(init=False, default=Factory(dict))

    _object_recognizer: ObjectRecognizer = attrib(validator=instance_of(ObjectRecognizer))
    _ontology: Ontology = attrib(validator=instance_of(Ontology))
    _debug_file: Optional[str] = attrib(kw_only=True, default=None)
    _graph_logger: Optional[GraphLogger] = attrib(default=None)

    def _print(self, graph: DiGraph, name: str) -> None:
        if self._debug_file:
            with open(self._debug_file, "a") as doc:
                doc.write("=== " + name + "===\n")
                for node in graph:
                    doc.write("\t" + str(node) + "\n")
                for edge in graph.edges:
                    doc.write("\t" + str(edge) + "\n")

    def observe(
        self, learning_example: LearningExample[PerceptionT, LinguisticDescription]
    ) -> None:
        perception = learning_example.perception
        original_perception = self._build_perception_graph(perception)

        if self._graph_logger:
            self._graph_logger.log_graph(
                original_perception,
                logging.INFO,
                "*** Preposition subset learner observing %s",
                learning_example.linguistic_description,
            )

        # DEBUG
        self._print(
            original_perception.copy_as_digraph(),
            learning_example.linguistic_description.as_token_string(),
        )

        post_recognition_object_perception_alignment = self._object_recognizer.match_objects_with_language(
            LanguageAlignedPerception(
                language=learning_example.linguistic_description,
                perception_graph=original_perception,
                node_to_language_span=immutabledict(),
            )
        )

        if self._graph_logger:
            self._graph_logger.log_graph(
                post_recognition_object_perception_alignment.perception_graph,
                logging.INFO,
                "Perception post-object-recognition",
            )

        num_matched_objects = len(
            post_recognition_object_perception_alignment.node_to_language_span
        )
        if num_matched_objects != 2:
            raise RuntimeError(
                f"Learning a preposition with more than two recognized objects "
                f"is not currently supported. Found {num_matched_objects} for "
                f"{learning_example.linguistic_description}."
            )

        # We represent prepositions as regex-like templates over the surface strings.
        # As an English-specific hack, the leftmost recognized object
        # is always taken to be the object modified, and the right one the ground.
        template_variables_to_object_match_nodes: ImmutableDict[
            SurfaceTemplateVariable, MatchedObjectNode
        ] = immutabledict(
            [
                (SLOT1, post_recognition_object_perception_alignment.aligned_nodes[0]),
                (SLOT2, post_recognition_object_perception_alignment.aligned_nodes[1]),
            ]
        )

        hypothesis_from_current_perception = self._hypothesis_from_perception(
            post_recognition_object_perception_alignment,
            template_variables_to_object_match_nodes,
        )

        surface_template = SurfaceTemplate.from_language_aligned_perception(
            post_recognition_object_perception_alignment,
            template_variables_to_object_match_nodes.inverse(),
        )

        # DEBUG
        self._print(
            hypothesis_from_current_perception.graph_pattern.copy_as_digraph(),
            f"New: {surface_template}",
        )

        existing_hypothesis_for_template = self._surface_template_to_preposition_pattern.get(
            surface_template
        )
        if existing_hypothesis_for_template:
            # We have seen this preposition situation before.
            # Our learning strategy is to assume the true semantics of the preposition
            # is what is in common between what we saw this time and what we saw last time.
            if self._graph_logger:
                self._graph_logger.log_graph(
                    existing_hypothesis_for_template.graph_pattern,
                    logging.INFO,
                    "We have seen this preposition template before. Here is our hypothesis "
                    "prior to seeing this example",
                )

            previous_hypothesis_as_digraph = (
                existing_hypothesis_for_template.graph_pattern.copy_as_digraph()
            )
            self._print(previous_hypothesis_as_digraph, "pre-intersection current: ")
            self._print(
                hypothesis_from_current_perception.graph_pattern.copy_as_digraph(),
                "pre-intersection intersecting",
            )
            new_hypothesis = hypothesis_from_current_perception.intersection(
                hypothesis_from_current_perception,
                graph_logger=self._graph_logger,
                ontology=self._ontology,
            )

            if new_hypothesis:
                self._surface_template_to_preposition_pattern[
                    surface_template
                ] = new_hypothesis

                if self._graph_logger:
                    self._graph_logger.log_graph(
                        new_hypothesis.graph_pattern, logging.INFO, "New hypothesis"
                    )

                self._print(
                    self._surface_template_to_preposition_pattern[
                        surface_template
                    ].graph_pattern.copy_as_digraph(),
                    "post-intersection: ",
                )
            else:
                raise RuntimeError("Intersection is the empty pattern :-(")
        else:
            logging.info(
                "This is a new preposition template; accepting current hypothesis"
            )
            # This is the first time we've seen a preposition situation like this one.
            # Remember our hypothesis about the semantics of the preposition.
            self._surface_template_to_preposition_pattern[
                surface_template
            ] = hypothesis_from_current_perception

        # DEBUG
        self._print(
            self._surface_template_to_preposition_pattern[
                surface_template
            ].graph_pattern.copy_as_digraph(),
            f"Saved: {surface_template}",
        )

    def _build_perception_graph(self, perception):
        if len(perception.frames) != 1:
            raise RuntimeError("Subset learner can only handle single frames for now")
        if isinstance(perception.frames[0], DevelopmentalPrimitivePerceptionFrame):
            original_perception = PerceptionGraph.from_frame(perception.frames[0])
        else:
            raise RuntimeError("Cannot process perception type.")
        return original_perception

    def _hypothesis_from_perception(
        self,
        scene_aligned_perception: LanguageAlignedPerception,
        template_variables_to_object_match_nodes: Mapping[
            SurfaceTemplateVariable, MatchedObjectNode
        ],
    ) -> PerceptionGraphTemplate:
        """
        Create a hypothesis for the semantics of a preposition based on the observed scene.
        
        Our current implementation is to just include the content 
        on the path between the recognized object nodes
        and one hop away from that path.
        """

        # The directions of edges in the perception graph are not necessarily meaningful
        # from the point-of-view of hypothesis generation, so we need an undirected copy
        # of the graph.
        perception_digraph = scene_aligned_perception.perception_graph.copy_as_digraph()
        perception_graph_undirected = perception_digraph.to_undirected(
            # as_view=True loses determinism
            as_view=False
        )

        self._print(perception_graph_undirected, "Undirected perception graph")

        if {SLOT1, SLOT2} != set(template_variables_to_object_match_nodes.keys()):
            raise RuntimeError(
                "Can only make a preposition hypothesis if the recognized "
                "objects are aligned to SurfaceTemplateVariables SLOT1 and SLOT2"
            )

        slot1_object = template_variables_to_object_match_nodes[SLOT1]
        slot2_object = template_variables_to_object_match_nodes[SLOT2]

        # The core of our hypothesis for the semantics of a preposition is all nodes
        # along the shortest path between the two objects involved in the perception graph.
        hypothesis_spine_nodes: ImmutableSet[PerceptionGraphNode] = immutableset(
            flatten(
                # if there are multiple paths between the object match nodes,
                # we aren't sure which are relevant, so we include them all in our hypothesis
                # and figure we can trim out irrelevant stuff as we make more observations.
                all_shortest_paths(
                    perception_graph_undirected, slot2_object, slot1_object
                )
            )
        )

        self._print(
            subgraph(perception_graph_undirected, nodes=hypothesis_spine_nodes),
            "Spine nodes",
        )

        # Along the core of our hypothesis we also want to collect the predecessors and successors
        hypothesis_nodes_mutable = []
        for node in hypothesis_spine_nodes:
            if node not in {slot1_object, slot2_object}:
                for successor in perception_digraph.successors(node):
                    if not isinstance(successor, ObjectPerception):
                        hypothesis_nodes_mutable.append(successor)
                for predecessor in perception_digraph.predecessors(node):
                    if not isinstance(predecessor, ObjectPerception):
                        hypothesis_nodes_mutable.append(predecessor)

        hypothesis_nodes_mutable.extend(hypothesis_spine_nodes)

        # We wrap the nodes in an immutable set to remove duplicates
        # while preserving iteration determinism.
        hypothesis_nodes = immutableset(hypothesis_nodes_mutable)

        preposition_pattern_graph = digraph_with_nodes_sorted_by(
            subgraph(perception_digraph, nodes=hypothesis_nodes), _graph_node_order
        )
        return PerceptionGraphTemplate.from_graph(
            preposition_pattern_graph, template_variables_to_object_match_nodes.items()
        )

    def describe(
        self, perception: PerceptualRepresentation[PerceptionT]
    ) -> Mapping[LinguisticDescription, float]:
        original_perception = self._build_perception_graph(perception)

        self._print(original_perception.copy_as_digraph(), "Describe Original")

        object_recognition_result = self._object_recognizer.match_objects(
            original_perception
        )

        post_recognition_graph = object_recognition_result.perception_graph
        recognized_objects_to_names = (
            object_recognition_result.description_to_matched_object_node.inverse()
        )

        self._print(
            post_recognition_graph.copy_as_digraph(), "Describe Recognized Objects"
        )

        # this will be our output
        description_to_score: List[Tuple[LinguisticDescription, float]] = []

        # For each preposition we've ever seen...
        for (
            preposition_surface_template,
            preposition_pattern,
        ) in self._surface_template_to_preposition_pattern.items():
            # try to see if (our model of) its semantics is present in the situation.
            matcher = preposition_pattern.graph_pattern.matcher(
                post_recognition_graph, matching_objects=False
            )
            for match in matcher.matches(use_lookahead_pruning=True):
                # if it is, use that preposition to describe the situation.
                description_to_score.append(
                    (
                        preposition_surface_template.instantiate(
                            template_variable_to_filler=immutabledict(
                                (
                                    preposition_pattern.pattern_node_to_template_variable[
                                        pattern_node
                                    ],
                                    # Wrapped in a tuple because fillers can in general be
                                    # multiple words.
                                    (
                                        recognized_objects_to_names[
                                            # We know, but the type system does not,
                                            # that if a MatchedObjectPerceptionPredicate matched,
                                            # the graph node must be a MatchedObjectNode
                                            cast(MatchedObjectNode, matched_graph_node)
                                        ],
                                    ),
                                )
                                for (
                                    pattern_node,
                                    matched_graph_node,
                                ) in match.pattern_node_to_matched_graph_node.items()
                                if isinstance(
                                    pattern_node, MatchedObjectPerceptionPredicate
                                )
                            )
                        ),
                        1.0,
                    )
                )

        return immutabledict(description_to_score)
