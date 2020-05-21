import logging
from abc import abstractmethod, ABC
from pathlib import Path
from typing import AbstractSet, Dict, Mapping, Optional, Set, Tuple, Iterable, Sequence

from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.surface_templates import SurfaceTemplate
from adam.learner.template_learner import (
    AbstractTemplateLearner,
    AbstractTemplateLearnerNew,
)
from adam.semantics import Concept
from immutablecollections import immutabledict, immutableset

from adam.language import TokenSequenceLinguisticDescription
from adam.ontology.ontology import Ontology
from adam.perception.perception_graph import DebugCallableType
from adam.learner.alignments import (
    LanguageConceptAlignment,
    LanguagePerceptionSemanticAlignment,
)
from attr import Factory, attrib, attrs
from attr.validators import instance_of


@attrs
class AbstractSubsetLearner(AbstractTemplateLearner, ABC):
    _surface_template_to_hypothesis: Dict[
        SurfaceTemplate, PerceptionGraphTemplate
    ] = attrib(init=False, default=Factory(dict))
    _ontology: Ontology = attrib(validator=instance_of(Ontology), kw_only=True)
    _debug_callback: Optional[DebugCallableType] = attrib(default=None, kw_only=True)

    def _learning_step(
        self,
        language_concept_alignment: LanguageConceptAlignment,
        surface_template: SurfaceTemplate,
    ) -> None:
        if surface_template in self._surface_template_to_hypothesis:
            # If already observed, get the largest matching subgraph of the pattern in the
            # current observation and
            # previous pattern hypothesis
            # TODO: We should relax this requirement for learning: issue #361
            previous_pattern_hypothesis = self._surface_template_to_hypothesis[
                surface_template
            ]

            updated_hypothesis = previous_pattern_hypothesis.intersection(
                self._hypothesis_from_perception(language_concept_alignment),
                ontology=self._ontology,
            )

            if updated_hypothesis:
                # Update the leading hypothesis
                self._surface_template_to_hypothesis[
                    surface_template
                ] = updated_hypothesis
            else:
                logging.warning(
                    "Intersection of graphs had empty result; keeping original pattern"
                )

        else:
            # If it's a new description, learn a new hypothesis/pattern, generated as a pattern
            # graph frm the
            # perception graph.
            self._surface_template_to_hypothesis[
                surface_template
            ] = self._hypothesis_from_perception(language_concept_alignment)

    @abstractmethod
    def _hypothesis_from_perception(
        self, preprocessed_input: LanguageConceptAlignment
    ) -> PerceptionGraphTemplate:
        pass

    def _primary_templates(
        self
    ) -> Iterable[Tuple[SurfaceTemplate, PerceptionGraphTemplate, float]]:
        return (
            (surface_template, hypothesis, 1.0)
            for (
                surface_template,
                hypothesis,
            ) in self._surface_template_to_hypothesis.items()
        )

    def _fallback_templates(
        self
    ) -> Iterable[Tuple[SurfaceTemplate, PerceptionGraphTemplate, float]]:
        return tuple()

    def _post_process_descriptions(
        self,
        match_results: Sequence[
            Tuple[TokenSequenceLinguisticDescription, PerceptionGraphTemplate, float]
        ],
    ) -> Mapping[TokenSequenceLinguisticDescription, float]:
        if not match_results:
            return immutabledict()

        largest_pattern_num_nodes = max(
            len(template.graph_pattern) for (_, template, _) in match_results
        )

        return immutabledict(
            (description, len(template.graph_pattern) / largest_pattern_num_nodes)
            for (description, template, score) in match_results
        )


@attrs
class AbstractSubsetLearnerNew(AbstractTemplateLearnerNew, ABC):
    _concept_to_hypothesis: Dict[Concept, PerceptionGraphTemplate] = attrib(
        init=False, default=Factory(dict)
    )
    _concept_to_surface_template: Dict[Concept, SurfaceTemplate] = attrib(
        init=False, default=Factory(dict)
    )
    _surface_template_to_concept: Dict[SurfaceTemplate, Concept] = attrib(
        init=False, default=Factory(dict)
    )
    _ontology: Ontology = attrib(validator=instance_of(Ontology), kw_only=True)
    _debug_callback: Optional[DebugCallableType] = attrib(default=None, kw_only=True)

    _known_bad_patterns: Set[SurfaceTemplate] = attrib(init=False, default=Factory(set))

    def _learning_step(
        self,
        language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment,
        surface_templates: Iterable[SurfaceTemplate],
    ) -> None:
        for surface_template in surface_templates:
            if surface_template in self._known_bad_patterns:
                # We tried to learn an alignment for this surface template previously
                # and it didn't work out.
                continue

            if surface_template in self._surface_template_to_concept:
                # If already observed, get the largest matching subgraph of the pattern in the
                # current observation and
                # previous pattern hypothesis
                # TODO: We should relax this requirement for learning: issue #361
                concept_for_surface_template = self._surface_template_to_concept[
                    surface_template
                ]
                previous_pattern_hypothesis = self._concept_to_hypothesis[
                    concept_for_surface_template
                ]

                updated_hypothesis = previous_pattern_hypothesis.intersection(
                    self._hypothesis_from_perception(
                        language_perception_semantic_alignment,
                        surface_template=surface_template,
                    ),
                    ontology=self._ontology,
                )

                if updated_hypothesis:
                    # Update the leading hypothesis
                    self._concept_to_hypothesis[
                        concept_for_surface_template
                    ] = updated_hypothesis
                else:
                    logging.debug(
                        "Intersection of graphs had empty result; assuming surface template %s "
                        "is not of the target type",
                        surface_template,
                    )
                    self._known_bad_patterns.add(surface_template)
            else:
                # If it's a new description, learn a new hypothesis/pattern, generated as a pattern
                # graph frm the perception graph.
                concept = self._new_concept(surface_template.to_short_string())
                hypothesis = self._hypothesis_from_perception(
                    language_perception_semantic_alignment, surface_template
                )
                self._surface_template_to_concept[surface_template] = concept
                self._concept_to_surface_template[concept] = surface_template
                self._concept_to_hypothesis[concept] = hypothesis

    def templates_for_concept(self, concept: Concept) -> AbstractSet[SurfaceTemplate]:
        if concept in self._concept_to_surface_template:
            return immutableset([self._concept_to_surface_template[concept]])
        else:
            return immutableset()

    @abstractmethod
    def _new_concept(self, debug_string: str) -> Concept:
        """
        Create a new `Concept` of the appropriate type with the given *debug_string*.
        """

    @abstractmethod
    def _hypothesis_from_perception(
        self,
        learning_state: LanguagePerceptionSemanticAlignment,
        surface_template: SurfaceTemplate,
    ) -> PerceptionGraphTemplate:
        """
        Get a hypothesis for the meaning of *surface_template* from a given *learning_state*.
        """

    def _primary_templates(
        self
    ) -> Iterable[Tuple[Concept, PerceptionGraphTemplate, float]]:
        return (
            (concept, hypothesis, 1.0)
            for (concept, hypothesis) in self._concept_to_hypothesis.items()
        )

    def _fallback_templates(
        self
    ) -> Iterable[Tuple[Concept, PerceptionGraphTemplate, float]]:
        return tuple()


@attrs  # pylint:disable=abstract-method
class AbstractTemplateSubsetLearner(AbstractSubsetLearner, AbstractTemplateLearner, ABC):
    def log_hypotheses(self, log_output_path: Path) -> None:
        logging.info(
            "Logging %s hypotheses to %s",
            len(self._surface_template_to_hypothesis),
            log_output_path,
        )
        for (
            surface_template,
            hypothesis,
        ) in self._surface_template_to_hypothesis.items():
            template_string = surface_template.to_short_string()
            hypothesis.render_to_file(template_string, log_output_path / template_string)


class AbstractTemplateSubsetLearnerNew(
    AbstractSubsetLearnerNew, AbstractTemplateLearnerNew, ABC
):
    def log_hypotheses(self, log_output_path: Path) -> None:
        logging.info(
            "Logging %s hypotheses to %s",
            len(self._concept_to_hypothesis),
            log_output_path,
        )
        for (concept, hypothesis) in self._concept_to_hypothesis.items():
            hypothesis.render_to_file(
                concept.debug_string, log_output_path / concept.debug_string
            )
