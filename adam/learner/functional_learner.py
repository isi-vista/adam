import logging
from pathlib import Path

from immutablecollections import immutabledict
from more_itertools import first
from typing import Tuple, Dict, List, AbstractSet

from attr.validators import instance_of

from attr import attrs, attrib, Factory

from adam.learner import (
    LanguageMode,
    LanguagePerceptionSemanticAlignment,
    PerceptionSemanticAlignment,
    SurfaceTemplate,
)
from adam.learner.template_learner import TemplateLearner
from adam.semantics import (
    SyntaxSemanticsVariable,
    LearnerSemantics,
    ObjectConcept,
    FunctionalObjectConcept,
    ActionConcept,
    Concept,
)


# This class needs a better name?
@attrs(slots=True)
class ConceptFunctionCounter:
    """
    A ConceptFunctionalCounter counts the number of times in which an object concept
    is used as an argument in an action concept. This information is then used
    when the learner encounters an unknown object concept in the same argument
    slot as we have seen previously.
    """

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
class FunctionalLearner(TemplateLearner):
    _observation_num: int = attrib(init=False, default=0)
    _language_mode: LanguageMode = attrib(validator=instance_of(LanguageMode))
    _concept_to_slots_to_function_counter: Dict[
        ActionConcept, Dict[SyntaxSemanticsVariable, ConceptFunctionCounter]
    ] = attrib(init=False, default=dict())

    def learn_from(
        self,
        language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment,
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

        # The functional learner is a 'semantics' learner rather than a perception learner
        # That is, the functional learner 'learns' information from the perceptual learner's output
        # Do to easily process over the semantics we go over these learner semantics here
        semantics = LearnerSemantics.from_nodes(
            language_perception_semantic_alignment.perception_semantic_alignment.semantic_nodes
        )
        # Right now, the functional learner only learns from actions so for each action concept in
        # the scene we do the following
        for semantic_node in semantics.actions:
            # Check if this action concept is one the functional learner has sceen before
            # If not, add an entry for it with a dictionary for the slot fillers
            if (
                semantic_node.concept
                not in self._concept_to_slots_to_function_counter.keys()
            ):
                self._concept_to_slots_to_function_counter[semantic_node.concept] = dict(
                    (slot, ConceptFunctionCounter())
                    for slot in semantic_node.slot_fillings.keys()
                )
            # Then for each slot filler alignment in the scene
            for (slot, slot_filler) in semantic_node.slot_fillings.items():
                # See if that slot is in our dictionary, it should be! Otherwise the
                # semantic node is invalid
                if (
                    slot
                    not in self._concept_to_slots_to_function_counter[
                        semantic_node.concept
                    ].keys()
                ):
                    raise RuntimeError(
                        f"Tried to align functional use to slot: {slot} in concept {semantic_node.concept} but {slot} didn't exist in the concept"
                    )
                # Finally count this entry in a functional concept counter
                self._concept_to_slots_to_function_counter[semantic_node.concept][
                    slot
                ].add_example(slot_filler.concept)

    def _enrich_common(
        self, perception_semantic_alignment: PerceptionSemanticAlignment
    ) -> PerceptionSemanticAlignment:
        # The functional learner is a 'semantics' learner rather than a perception learner
        # That is, the functional learner 'learns' information from the perceptual learner's output
        # Do to easily process over the semantics we go over these learner semantics here
        semantics = LearnerSemantics.from_nodes(
            perception_semantic_alignment.semantic_nodes
        )
        list_of_matches: List[Tuple[FunctionalObjectConcept, ObjectConcept]] = []
        # The functional learner only deals with action semantics so
        # we go over each action semantics in the scene and if a slot filler
        # is of an FunctionalObjectConcept type, then we add a mapping from that
        # FunctionalObjectConcept to a known concept with our best guess concept
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

    def enrich_during_learning(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> LanguagePerceptionSemanticAlignment:
        post_enrichment_perception_semantics = self._enrich_common(
            language_perception_semantic_alignment.perception_semantic_alignment
        )
        return LanguagePerceptionSemanticAlignment(
            language_concept_alignment=language_perception_semantic_alignment.language_concept_alignment,
            perception_semantic_alignment=post_enrichment_perception_semantics,
        )

    def enrich_during_description(
        self, perception_semantic_alignment: PerceptionSemanticAlignment
    ) -> PerceptionSemanticAlignment:
        return self._enrich_common(perception_semantic_alignment)

    def templates_for_concept(self, concept: Concept) -> AbstractSet[SurfaceTemplate]:
        # Our Functional Learner doesn't store concepts the same way as other
        # Template learners
        pass

    def log_hypotheses(self, log_output_path: Path) -> None:
        # Could possibly be improved see: https://github.com/isi-vista/adam/issues/938
        with open(log_output_path / "functional_learner.txt", "w") as file:
            for concept in self._concept_to_slots_to_function_counter:
                file.write(f"{concept}\n")
                for slot in self._concept_to_slots_to_function_counter[concept]:
                    file.write(f"\t{slot}\n")
                    for (
                        functional_concept
                    ) in self._concept_to_slots_to_function_counter[  # pylint:disable=protected-access
                        concept
                    ][
                        slot
                    ]._concept_to_count:  # pylint:disable=protected-access
                        file.write(
                            f"\t\t{functional_concept} - {self._concept_to_slots_to_function_counter[concept][slot]._concept_to_count[functional_concept]}\n"  # pylint:disable=protected-access
                        )
