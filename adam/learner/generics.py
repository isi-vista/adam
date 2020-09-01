import logging
from pathlib import Path
from typing import AbstractSet, Optional, Tuple, Dict, Set

from attr import attrs, attrib, Factory

from adam.learner import SurfaceTemplate
from adam.learner.alignments import (
    LanguagePerceptionSemanticAlignment,
    PerceptionSemanticAlignment,
)
from adam.learner.template_learner import TemplateLearner
from adam.semantics import (
    Concept,
    ActionSemanticNode,
    ObjectSemanticNode,
    ObjectConcept,
    ActionConcept,
)


@attrs
class SimpleGenericsLearner(TemplateLearner):
    learned_representations: Dict[
        Tuple[str, ...], Tuple[ObjectConcept, Set[ActionConcept]]
    ] = attrib(init=False, default=Factory(dict))

    def enrich_during_learning(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> LanguagePerceptionSemanticAlignment:
        return language_perception_semantic_alignment

    def enrich_during_description(
        self, perception_semantic_alignment: PerceptionSemanticAlignment
    ) -> PerceptionSemanticAlignment:
        return perception_semantic_alignment

    def log_hypotheses(self, log_output_path: Path) -> None:
        logging.info(
            "Logging %s hypotheses to %s",
            len(self.learned_representations),
            log_output_path,
        )
        Path(log_output_path).mkdir(parents=True, exist_ok=True)
        with open(log_output_path / f"generics_log.txt", "w") as out:
            for (
                sequence,
                (object_concept, actions),
            ) in self.learned_representations.items():
                out.write(f'Learned generic: {" ".join(sequence)} \n')
                out.write(f"Related objects: {object_concept} \n")
                out.write(f"Related actions: {actions} \n\n")

    def templates_for_concept(self, concept: Concept) -> AbstractSet[SurfaceTemplate]:
        return set()

    def learn_from(
        self,
        language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment,
        observation_num: int = -1,
    ) -> None:
        sequence = (
            language_perception_semantic_alignment.language_concept_alignment.language.as_token_sequence()
        )
        recognized_semantic_nodes = (
            language_perception_semantic_alignment.perception_semantic_alignment.semantic_nodes
        )
        span = (
            language_perception_semantic_alignment.language_concept_alignment.node_to_language_span
        )

        # Check if both an action and an object is recognized
        action_nodes = [
            n for n in recognized_semantic_nodes if isinstance(n, ActionSemanticNode)
        ]
        object_nodes = [
            n for n in recognized_semantic_nodes if isinstance(n, ObjectSemanticNode)
        ]

        significant_object_node: Optional[ObjectSemanticNode] = None
        # Check if a recognized object matches the heard utterance
        for node in recognized_semantic_nodes:
            if isinstance(node, ObjectSemanticNode) and node in span:
                significant_object_node = node

        if action_nodes and object_nodes and significant_object_node:
            # Generic!
            action_concepts = set([a.concept for a in action_nodes])
            if sequence in self.learned_representations:
                known_representation = self.learned_representations[sequence]
                known_representation[1].update(action_concepts)
            else:
                self.learned_representations[sequence] = (
                    significant_object_node.concept,
                    action_concepts,
                )

