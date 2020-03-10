import logging
from typing import Dict, Generic, Mapping, Optional, Tuple

from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.surface_templates import SurfaceTemplate
from immutablecollections import immutabledict

from adam.language import (
    LinguisticDescription,
    LinguisticDescriptionT,
    TokenSequenceLinguisticDescription,
)
from adam.learner import (
    LanguageLearner,
    LearningExample,
    get_largest_matching_pattern,
    graph_without_learner,
)
from adam.ontology.ontology import Ontology
from adam.perception import PerceptionT, PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.perception.perception_graph import (
    DebugCallableType,
    PerceptionGraph,
    PerceptionGraphPattern,
)
from attr import Factory, attrib, attrs
from attr.validators import instance_of


@attrs(slots=True)
class SubsetObjectLearner(
    Generic[PerceptionT],
    LanguageLearner[DevelopmentalPrimitivePerceptionFrame, LinguisticDescription],
):
    """
    An implementation of `LanguageLearner` for subset learning based approach for single object detection.
    """

    _descriptions_to_pattern_hypothesis: Dict[
        SurfaceTemplate, PerceptionGraphTemplate
    ] = attrib(init=False, default=Factory(dict))
    _ontology: Ontology = attrib(validator=instance_of(Ontology), kw_only=True)
    _debug_callback: Optional[DebugCallableType] = attrib(default=None, kw_only=True)

    def observe(
        self,
        learning_example: LearningExample[
            DevelopmentalPrimitivePerceptionFrame, LinguisticDescription
        ],
    ) -> None:
        perception = learning_example.perception
        if len(perception.frames) != 1:
            raise RuntimeError("Subset learner can only handle single frames for now")
        if isinstance(perception.frames[0], DevelopmentalPrimitivePerceptionFrame):
            original_perception_graph = PerceptionGraph.from_frame(perception.frames[0])
        else:
            raise RuntimeError("Cannot process perception type.")

        # Remove learner from the perception
        observed_perception_graph = graph_without_learner(original_perception_graph)
        surface_template = SurfaceTemplate(
            learning_example.linguistic_description.as_token_sequence()
        )

        if surface_template in self._descriptions_to_pattern_hypothesis:
            # If already observed, get the largest matching subgraph of the pattern in the current observation and
            # previous pattern hypothesis
            # TODO: We should relax this requirement for learning: issue #361
            previous_pattern_hypothesis = self._descriptions_to_pattern_hypothesis[
                surface_template
            ]

            # Get largest subgraph match using the pattern and the graph
            hypothesis_pattern_common_subgraph = get_largest_matching_pattern(
                previous_pattern_hypothesis.graph_pattern,
                observed_perception_graph,
                debug_callback=self._debug_callback,
                ontology=self._ontology,
                matching_objects=True,
            )
            if hypothesis_pattern_common_subgraph:
                # Update the leading hypothesis
                self._descriptions_to_pattern_hypothesis[
                    surface_template
                ] = PerceptionGraphTemplate(hypothesis_pattern_common_subgraph)
            else:
                logging.warning(
                    "Intersection of graphs had empty result; keeping original pattern"
                )

        else:
            # If it's a new description, learn a new hypothesis/pattern, generated as a pattern graph frm the
            # perception graph.
            observed_pattern_graph = PerceptionGraphPattern.from_graph(
                observed_perception_graph.copy_as_digraph()
            ).perception_graph_pattern
            self._descriptions_to_pattern_hypothesis[
                surface_template
            ] = PerceptionGraphTemplate(observed_pattern_graph)

    def describe(
        self, perception: PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame]
    ) -> Mapping[LinguisticDescription, float]:
        if len(perception.frames) != 1:
            raise RuntimeError("Subset learner can only handle single frames for now")
        if isinstance(perception.frames[0], DevelopmentalPrimitivePerceptionFrame):
            original_perception_graph = PerceptionGraph.from_frame(perception.frames[0])
        else:
            raise RuntimeError("Cannot process perception type.")
        observed_perception_graph = graph_without_learner(original_perception_graph)

        # get the learned description for which there are the maximum number of matching properties (i.e. most specific)
        max_matching_subgraph_size = 0
        learned_description: Optional[SurfaceTemplate] = None
        for (
            description,
            pattern_hypothesis,
        ) in self._descriptions_to_pattern_hypothesis.items():
            # get the largest common match
            largest_matching_pattern = get_largest_matching_pattern(
                pattern_hypothesis.graph_pattern,
                observed_perception_graph,
                debug_callback=self._debug_callback,
                ontology=self._ontology,
                matching_objects=True,
            )
            common_pattern_size = (
                len(largest_matching_pattern.copy_as_digraph().nodes)
                if largest_matching_pattern
                else 0
            )
            if (
                largest_matching_pattern
                and common_pattern_size > max_matching_subgraph_size
            ):
                learned_description = description
                max_matching_subgraph_size = common_pattern_size
        if learned_description:
            return immutabledict(
                (
                    (
                        learned_description.instantiate(
                            template_variable_to_filler=immutabledict()
                        ),
                        1.0,
                    ),
                )
            )
        else:
            return immutabledict()
