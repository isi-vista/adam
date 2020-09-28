import logging
from pathlib import Path
from typing import AbstractSet, Optional, Tuple, Dict, Set

from attr import attrs, attrib, Factory

from adam.learner import SurfaceTemplate
from adam.learner.alignments import (
    LanguagePerceptionSemanticAlignment,
    PerceptionSemanticAlignment,
)
from adam.learner.learner_utils import get_slot_from_semantic_node
from adam.learner.template_learner import TemplateLearner
from adam.semantics import (
    Concept,
    ObjectSemanticNode,
    ActionSemanticNode,
    SyntaxSemanticsVariable,
)


@attrs
class SimpleGenericsLearner(TemplateLearner):
    learned_representations: Dict[
        Tuple[str, ...], Tuple[Concept, Set[Tuple[Concept, SyntaxSemanticsVariable]]]
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
        offset: int = 0,
    ) -> None:
        sequence = (
            language_perception_semantic_alignment.language_concept_alignment.language.as_token_sequence()
        )
        recognized_semantic_nodes = list(
            language_perception_semantic_alignment.perception_semantic_alignment.semantic_nodes
        )
        span = (
            language_perception_semantic_alignment.language_concept_alignment.node_to_language_span
        )

        # Get actions and attributes that are recognized in the scene
        action_nodes = [
            n for n in recognized_semantic_nodes if isinstance(n, ActionSemanticNode)
        ]

        # Check if a recognized object matches the heard utterance
        significant_object_concept: Optional[Concept] = None
        for node in recognized_semantic_nodes:
            if isinstance(node, ObjectSemanticNode) and node in span:
                significant_object_concept = node.concept

        # Actions: E.g dog s walk
        # TODO: Attributes: E.g cookies are brown
        # If there is a recognized object node that matches the scene, and a generic action OR attribute, learn!
        if significant_object_concept and action_nodes:
            # Get the slot of object for each recognized action in the statement
            action_concepts_and_object_slots = []
            for node in action_nodes:
                slot = get_slot_from_semantic_node(significant_object_concept, node)
                if slot:
                    action_concepts_and_object_slots.append((node.concept, slot))
            # Update the representation for learned generics for the sequence
            if sequence in self.learned_representations:
                known_representation = self.learned_representations[sequence]
                known_representation[1].update(action_concepts_and_object_slots)
            else:
                self.learned_representations[sequence] = (
                    significant_object_concept,
                    set(action_concepts_and_object_slots),
                )
