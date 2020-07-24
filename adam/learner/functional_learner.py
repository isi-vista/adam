import logging
from pathlib import Path

from immutablecollections import immutabledict
from more_itertools import first
from typing import Tuple, Dict, List

from attr.validators import instance_of

from attr import attrs, attrib, Factory

from adam.learner import (
    LanguageMode,
    LanguagePerceptionSemanticAlignment,
    PerceptionSemanticAlignment,
)
from adam.semantics import (
    SyntaxSemanticsVariable,
    LearnerSemantics,
    ObjectConcept,
    FunctionalObjectConcept,
    ActionConcept,
)


# This class needs a better name?
@attrs(slots=True)
class ConceptFunctionCounter:
    _concept_to_count: Dict[ObjectConcept, int] = attrib(
        init=False, default=Factory(dict)
    )
    _num_instances_seen: int = attrib(init=False, default=0)

    def get_best_concept(self) -> ObjectConcept:
        def sort_by_counts(tok_to_count: Tuple[ObjectConcept, int]) -> int:
            _, count = tok_to_count
            return count

        sorted_by_count = [(k, v) for k, v in self._concept_to_count.items()]
        sorted_by_count.sort(key=sort_by_counts, reverse=True)
        concept, _ = first(sorted_by_count)
        # This should apply the tolerance principal to get the assumed token
        # If we don't know what the observation is
        # But for now we just return the highest seen argument
        return concept

    def add_example(self, concept: ObjectConcept) -> None:
        if not isinstance(concept, FunctionalObjectConcept):
            if concept in self._concept_to_count.keys():
                self._concept_to_count[concept] += 1
            else:
                self._concept_to_count[concept] = 1
            self._num_instances_seen += 1


@attrs
class FunctionalLearner:
    _observation_num: int = attrib(init=False, default=0)
    _language_mode: LanguageMode = attrib(validator=instance_of(LanguageMode))
    _concept_to_slots_to_function_counter: Dict[
        ActionConcept, Dict[SyntaxSemanticsVariable, ConceptFunctionCounter]
    ] = attrib(init=False, default=dict())

    def learn_from(
        self,
        language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment,
        *,
        observation_num: int = -1,
    ):
        if observation_num >= 0:
            logging.info(
                "Observation %s: %s",
                observation_num,
                language_perception_semantic_alignment.language_concept_alignment.language.as_token_string(),
            )
        else:
            logging.info(
                "Observation %s: %s",
                self._observation_num,
                language_perception_semantic_alignment.language_concept_alignment.language.as_token_string(),
            )

        self._observation_num += 1

        semantics = LearnerSemantics.from_nodes(
            language_perception_semantic_alignment.perception_semantic_alignment.semantic_nodes
        )
        for semantic_node in semantics.actions:
            if (
                semantic_node.concept
                not in self._concept_to_slots_to_function_counter.keys()
            ):
                self._concept_to_slots_to_function_counter[semantic_node.concept] = dict(
                    (slot, ConceptFunctionCounter())
                    for slot in semantic_node.slot_fillings.keys()
                )
            for (slot, slot_filler) in semantic_node.slot_fillings.items():
                if (
                    slot
                    not in self._concept_to_slots_to_function_counter[
                        semantic_node.concept
                    ].keys()
                ):
                    raise RuntimeError(
                        f"Tried to align functional use to slot: {slot} in concept {semantic_node.concept} but {slot} didn't exist in the concept"
                    )
                self._concept_to_slots_to_function_counter[semantic_node.concept][
                    slot
                ].add_example(slot_filler.concept)

    def enrich_during_description(
        self, perception_semantic_alignment: PerceptionSemanticAlignment
    ):
        semantics = LearnerSemantics.from_nodes(
            perception_semantic_alignment.semantic_nodes
        )
        list_of_matches: List[Tuple[FunctionalObjectConcept, ObjectConcept]] = []
        for action_semantic_node in semantics.actions:
            if action_semantic_node.concept in self._concept_to_slots_to_function_counter:
                for slot, slot_filler in action_semantic_node.slot_fillings.items():
                    if (
                        slot
                        in self._concept_to_slots_to_function_counter[
                            action_semantic_node.concept
                        ]
                    ):
                        if isinstance(slot_filler.concept, FunctionalObjectConcept):
                            list_of_matches.append(
                                (
                                    slot_filler.concept,
                                    self._concept_to_slots_to_function_counter[
                                        action_semantic_node.concept
                                    ][slot].get_best_concept(),
                                )
                            )

        return perception_semantic_alignment.copy_with_mapping(
            mapping=immutabledict(list_of_matches)
        )

    def log_hypotheses(self, log_output_path: Path) -> None:
        pass
