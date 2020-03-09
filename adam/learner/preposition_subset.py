import logging
from typing import Dict, Generic, List, Mapping, Optional, Tuple, cast

from attr.validators import instance_of
from networkx import DiGraph

from adam.language import LinguisticDescription
from adam.learner import LanguageLearner, LearningExample
from adam.learner.learner_utils import pattern_match_to_description
from adam.learner.object_recognizer import ObjectRecognizer
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.prepositions import preposition_hypothesis_from_perception
from adam.learner.surface_templates import (
    SurfaceTemplate,
    SurfaceTemplateVariable,
    SLOT1,
    SLOT2,
)
from adam.ontology.ontology import Ontology
from adam.perception import PerceptionT, PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception.perception_graph import (
    GraphLogger,
    LanguageAlignedPerception,
    MatchedObjectNode,
    MatchedObjectPerceptionPredicate,
    PerceptionGraph,
)
from attr import Factory, attrib, attrs
from immutablecollections import ImmutableDict, immutabledict

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

        hypothesis_from_current_perception = preposition_hypothesis_from_perception(
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
                        pattern_match_to_description(
                            surface_template=preposition_surface_template,
                            pattern=preposition_pattern,
                            match=match,
                            matched_objects_to_names=recognized_objects_to_names,
                        ),
                        1.0,
                    )
                )

        return immutabledict(description_to_score)
