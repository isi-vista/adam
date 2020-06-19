import logging
from abc import ABC
from pathlib import Path
from random import Random
from typing import AbstractSet, Iterable, List, Optional, Sequence, Union
from adam.language_specific.english.english_language_generator import (
    GAILA_PHASE_1_LANGUAGE_GENERATOR,
)

from adam.language import LinguisticDescription
from adam.learner import (
    LearningExample,
    get_largest_matching_pattern,
    graph_without_learner,
)
from adam.learner.alignments import (
    LanguagePerceptionSemanticAlignment,
    PerceptionSemanticAlignment,
)
from adam.learner.learner_utils import assert_static_situation
from adam.learner.object_recognizer import (
    ObjectRecognizer,
    PerceptionGraphFromObjectRecognizer,
    extract_candidate_objects,
)
from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.pursuit import AbstractPursuitLearner, HypothesisLogger
from adam.learner.subset import (
    AbstractTemplateSubsetLearner,
    AbstractTemplateSubsetLearnerNew,
)
from adam.learner.surface_templates import (
    SurfaceTemplate,
    SurfaceTemplateBoundToSemanticNodes,
)
from adam.learner.template_learner import (
    AbstractTemplateLearner,
    AbstractTemplateLearnerNew,
    TemplateLearner,
)
from adam.ontology.phase1_ontology import GAILA_PHASE_1_ONTOLOGY
from adam.ontology.phase1_spatial_relations import Region
from adam.perception import ObjectPerception, PerceptualRepresentation
from adam.perception.deprecated import LanguageAlignedPerception
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
    RgbColorPerception,
)
from adam.perception.perception_graph import PerceptionGraph, PerceptionGraphPattern
from adam.semantics import Concept, ObjectConcept, GROUND_OBJECT_CONCEPT
from adam.utils import networkx_utils
from attr import attrib, attrs, evolve
from attr.validators import instance_of, optional
from immutablecollections import (
    ImmutableSet,
    ImmutableSetMultiDict,
    immutabledict,
    immutableset,
    immutablesetmultidict,
)
from vistautils.parameters import Parameters


class AbstractObjectTemplateLearnerNew(AbstractTemplateLearnerNew):
    # pylint:disable=abstract-method
    def _can_learn_from(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> bool:
        # We can try to learn objects from anything, as long as the scene isn't already
        # completely understood.
        return (
            not language_perception_semantic_alignment.language_concept_alignment.is_entirely_aligned
        )

    def _preprocess_scene(
        self, perception_semantic_alignment: PerceptionSemanticAlignment
    ) -> PerceptionSemanticAlignment:
        # Avoid accidentally identifying a word with the learner itself.
        return perception_semantic_alignment.copy_with_updated_graph_and_added_nodes(
            new_graph=graph_without_learner(
                perception_semantic_alignment.perception_graph
            ),
            new_nodes=[],
        )

    def _candidate_templates(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> AbstractSet[SurfaceTemplateBoundToSemanticNodes]:
        # We can only learn single words for objects at the moment.
        # See https://github.com/isi-vista/adam/issues/793 .

        # Attempt to align every unaligned token to some object in the scene.
        language_alignment = (
            language_perception_semantic_alignment.language_concept_alignment
        )
        return immutableset(
            SurfaceTemplateBoundToSemanticNodes(
                SurfaceTemplate.for_object_name(token), slot_to_semantic_node={}
            )
            for (tok_idx, token) in enumerate(
                language_alignment.language.as_token_sequence()
            )
            if not language_alignment.token_index_is_aligned(tok_idx)
        )


class AbstractObjectTemplateLearner(AbstractTemplateLearner, ABC):
    def _assert_valid_input(
        self,
        to_check: Union[
            LearningExample[DevelopmentalPrimitivePerceptionFrame, LinguisticDescription],
            PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame],
        ],
    ) -> None:
        assert_static_situation(to_check)

    def _extract_perception_graph(
        self, perception: PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame]
    ) -> PerceptionGraph:
        return PerceptionGraph.from_frame(perception.frames[0])

    def _preprocess_scene_for_learning(
        self,
        language_concept_alignment: LanguageAlignedPerception,
        language_generator=None,
    ) -> LanguageAlignedPerception:
        return evolve(
            language_concept_alignment,
            perception_graph=self._common_preprocessing(
                language_concept_alignment.perception_graph
            ),
        )

    def _preprocess_scene_for_description(
        self, perception_graph: PerceptionGraph, language_generator=None
    ) -> PerceptionGraphFromObjectRecognizer:
        return PerceptionGraphFromObjectRecognizer(
            self._common_preprocessing(perception_graph),
            description_to_matched_object_node=immutabledict(),
        )

    def _common_preprocessing(self, perception_graph: PerceptionGraph) -> PerceptionGraph:
        return graph_without_learner(perception_graph)

    def _extract_surface_template(
        self, language_concept_alignment: LanguageAlignedPerception
    ) -> SurfaceTemplate:
        return SurfaceTemplate(language_concept_alignment.language.as_token_sequence())


@attrs
class ObjectPursuitLearner(AbstractPursuitLearner, AbstractObjectTemplateLearner):
    """
    An implementation of pursuit learner for object recognition
    """

    def _candidate_hypotheses(
        self, language_aligned_perception: LanguageAlignedPerception
    ) -> Sequence[PerceptionGraphTemplate]:
        """
        Given a learning input, returns all possible meaning hypotheses.
        """
        return [
            self._hypothesis_from_perception(object_)
            for object_ in self._candidate_perceptions(
                language_aligned_perception.perception_graph
            )
        ]

    def get_objects_from_perception(
        self, observed_perception_graph: PerceptionGraph
    ) -> List[PerceptionGraph]:
        perception_as_digraph = observed_perception_graph.copy_as_digraph()
        perception_as_graph = perception_as_digraph.to_undirected()

        meanings = []

        # 1) Take all of the obj perc that dont have part of relationships with anything else
        root_object_percetion_nodes = []
        for node in perception_as_graph.nodes:
            if isinstance(node, ObjectPerception) and node.debug_handle != "the ground":
                if not any(
                    [
                        u == node and str(data["label"]) == "partOf"
                        for u, v, data in perception_as_digraph.edges.data()
                    ]
                ):
                    root_object_percetion_nodes.append(node)

        # 2) for each of these, walk along the part of relationships backwards,
        # i.e find all of the subparts of the root object
        for root_object_perception_node in root_object_percetion_nodes:
            # Iteratively get all other object perceptions that connect to a root with a part of
            # relation
            all_object_perception_nodes = [root_object_perception_node]
            frontier = [root_object_perception_node]
            updated = True
            while updated:
                updated = False
                new_frontier = []
                for frontier_node in frontier:
                    for node in perception_as_graph.neighbors(frontier_node):
                        edge_data = perception_as_digraph.get_edge_data(
                            node, frontier_node, default=-1
                        )
                        if edge_data != -1 and str(edge_data["label"]) == "partOf":
                            new_frontier.append(node)

                if new_frontier:
                    all_object_perception_nodes.extend(new_frontier)
                    updated = True
                    frontier = new_frontier

            # Now we have a list of all perceptions that are connected
            # 3) For each of these objects including root object, get axes, properties,
            # and relations and regions which are between these internal object perceptions
            other_nodes = []
            for node in all_object_perception_nodes:
                for neighbor in perception_as_graph.neighbors(node):
                    # Filter out regions that don't have a reference in all object perception nodes
                    # TODO: We currently remove colors to achieve a match - otherwise finding
                    #  patterns fails.
                    if (
                        isinstance(neighbor, Region)
                        and neighbor.reference_object not in all_object_perception_nodes
                        or isinstance(neighbor, RgbColorPerception)
                    ):
                        continue
                    # Append all other none-object nodes to be kept in the subgraph
                    if not isinstance(neighbor, ObjectPerception):
                        other_nodes.append(neighbor)

            subgraph = networkx_utils.subgraph(
                perception_as_digraph, all_object_perception_nodes + other_nodes
            )
            meanings.append(PerceptionGraph(subgraph))
        logging.info(f"Got {len(meanings)} candidate meanings")
        return meanings

    def _hypothesis_from_perception(
        self, perception: PerceptionGraph
    ) -> PerceptionGraphTemplate:
        return PerceptionGraphTemplate(
            graph_pattern=PerceptionGraphPattern.from_graph(
                perception
            ).perception_graph_pattern
        )

    def _candidate_perceptions(self, observed_perception_graph) -> List[PerceptionGraph]:
        return self.get_objects_from_perception(observed_perception_graph)

    def _matches(
        self,
        *,
        hypothesis: PerceptionGraphTemplate,
        observed_perception_graph: PerceptionGraph,
    ) -> bool:
        matcher = hypothesis.graph_pattern.matcher(
            observed_perception_graph, matching_objects=True
        )
        return any(
            matcher.matches(
                use_lookahead_pruning=True, graph_logger=self._hypothesis_logger
            )
        )

    @attrs(frozen=True)
    class ObjectHypothesisPartialMatch(AbstractPursuitLearner.PartialMatch):
        partial_match_hypothesis: Optional[PerceptionGraphTemplate] = attrib(
            validator=optional(instance_of(PerceptionGraphTemplate))
        )
        num_nodes_matched: int = attrib(validator=instance_of(int), kw_only=True)
        num_nodes_in_pattern: int = attrib(validator=instance_of(int), kw_only=True)

        def matched_exactly(self) -> bool:
            return self.num_nodes_matched == self.num_nodes_in_pattern

        def match_score(self) -> float:
            return self.num_nodes_matched / self.num_nodes_in_pattern

    def _find_partial_match(
        self, hypothesis: PerceptionGraphTemplate, graph: PerceptionGraph
    ) -> "ObjectPursuitLearner.ObjectHypothesisPartialMatch":
        pattern = hypothesis.graph_pattern
        hypothesis_pattern_common_subgraph = get_largest_matching_pattern(
            pattern,
            graph,
            debug_callback=self._debug_callback,
            graph_logger=self._hypothesis_logger,
            ontology=self._ontology,
            matching_objects=True,
        )
        self.debug_counter += 1

        leading_hypothesis_num_nodes = len(pattern)
        num_nodes_matched = (
            len(hypothesis_pattern_common_subgraph.copy_as_digraph().nodes)
            if hypothesis_pattern_common_subgraph
            else 0
        )

        return ObjectPursuitLearner.ObjectHypothesisPartialMatch(
            PerceptionGraphTemplate(graph_pattern=hypothesis_pattern_common_subgraph)
            if hypothesis_pattern_common_subgraph
            else None,
            num_nodes_matched=num_nodes_matched,
            num_nodes_in_pattern=leading_hypothesis_num_nodes,
        )

    def _find_identical_hypothesis(
        self,
        new_hypothesis: PerceptionGraphTemplate,
        candidates: Iterable[PerceptionGraphTemplate],
    ) -> Optional[PerceptionGraphTemplate]:
        for candidate in candidates:
            if new_hypothesis.graph_pattern.check_isomorphism(candidate.graph_pattern):
                return candidate
        return None

    def _are_isomorphic(
        self, h: PerceptionGraphTemplate, hypothesis: PerceptionGraphTemplate
    ) -> bool:
        return h.graph_pattern.check_isomorphism(hypothesis.graph_pattern)

    @staticmethod
    def from_parameters(
        params: Parameters, *, graph_logger: Optional[HypothesisLogger] = None
    ) -> "ObjectPursuitLearner":  # type: ignore
        log_word_hypotheses_dir = params.optional_creatable_directory(
            "log_word_hypotheses_dir"
        )
        if log_word_hypotheses_dir:
            logging.info("Hypotheses will be logged to %s", log_word_hypotheses_dir)

        rng = Random()
        rng.seed(params.optional_integer("random_seed", default=0))

        return ObjectPursuitLearner(
            learning_factor=params.floating_point("learning_factor"),
            graph_match_confirmation_threshold=params.floating_point(
                "graph_match_confirmation_threshold"
            ),
            lexicon_entry_threshold=params.floating_point("lexicon_entry_threshold"),
            smoothing_parameter=params.floating_point("smoothing_parameter"),
            hypothesis_logger=graph_logger,
            log_learned_item_hypotheses_to=log_word_hypotheses_dir,
            rng=rng,
            ontology=GAILA_PHASE_1_ONTOLOGY,
        )

    def log_hypotheses(self, log_output_path: Path) -> None:
        for (surface_template, hypothesis) in self._lexicon.items():
            template_string = surface_template.to_short_string()
            hypothesis.render_to_file(template_string, log_output_path / template_string)


@attrs(slots=True)
class SubsetObjectLearner(AbstractTemplateSubsetLearner, AbstractObjectTemplateLearner):
    """
    An implementation of `LanguageLearner` for subset learning based approach for single object detection.
    """

    def _hypothesis_from_perception(
        self, preprocessed_input: LanguageAlignedPerception
    ) -> PerceptionGraphTemplate:
        new_hypothesis = PerceptionGraphPattern.from_graph(
            preprocessed_input.perception_graph
        ).perception_graph_pattern
        return PerceptionGraphTemplate(
            graph_pattern=new_hypothesis,
            template_variable_to_pattern_node=immutabledict(),
        )


@attrs(slots=True)
class SubsetObjectLearnerNew(
    AbstractObjectTemplateLearnerNew, AbstractTemplateSubsetLearnerNew
):
    """
    An implementation of `LanguageLearner` for subset learning based approach for single object detection.
    """

    def _new_concept(self, debug_string: str) -> ObjectConcept:
        return ObjectConcept(debug_string)

    def _hypotheses_from_perception(
        self,
        learning_state: LanguagePerceptionSemanticAlignment,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,
    ) -> AbstractSet[PerceptionGraphTemplate]:
        if bound_surface_template.slot_to_semantic_node:
            raise RuntimeError(
                "Object learner should not have slot to semantic node alignments!"
            )

        return immutableset(
            PerceptionGraphTemplate(
                graph_pattern=PerceptionGraphPattern.from_graph(
                    candidate_object
                ).perception_graph_pattern,
                template_variable_to_pattern_node=immutabledict(),
            )
            for candidate_object in extract_candidate_objects(
                learning_state.perception_semantic_alignment.perception_graph
            )
        )

    # I can't spot the difference in arguments pylint claims?
    def _keep_hypothesis(  # pylint: disable=arguments-differ
        self,
        hypothesis: PerceptionGraphTemplate,
        bound_surface_template: SurfaceTemplateBoundToSemanticNodes,  # pylint:disable=unused-argument
    ) -> bool:
        if len(hypothesis.graph_pattern) < 2:
            # A one node graph is to small to meaningfully describe an object
            return False
        if all(isinstance(node, ObjectPerception) for node in hypothesis.graph_pattern):
            # A hypothesis which consists of just sub-object structure
            # with no other content is insufficiently distinctive.
            return False
        return True


@attrs(frozen=True, kw_only=True)
class ObjectRecognizerAsTemplateLearner(TemplateLearner):
    _object_recognizer: ObjectRecognizer = attrib(validator=instance_of(ObjectRecognizer))
    _concepts_to_templates: ImmutableSetMultiDict[Concept, SurfaceTemplate] = attrib(
        init=False
    )

    def learn_from(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> None:
        # The object recognizer doesn't learn anything.
        # It just recognizes predefined object patterns.
        pass

    def enrich_during_learning(
        self,
        language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment,
        language_generator=GAILA_PHASE_1_LANGUAGE_GENERATOR,
    ) -> LanguagePerceptionSemanticAlignment:
        return self._object_recognizer.match_objects_with_language(
            language_perception_semantic_alignment, language_generator=language_generator
        )

    def enrich_during_description(
        self,
        perception_semantic_alignment: PerceptionSemanticAlignment,
        language_generator=GAILA_PHASE_1_LANGUAGE_GENERATOR,
    ) -> PerceptionSemanticAlignment:
        (new_perception_semantic_alignment, _) = self._object_recognizer.match_objects(
            perception_semantic_alignment, language_generator=language_generator
        )
        return new_perception_semantic_alignment

    def templates_for_concept(self, concept: Concept) -> ImmutableSet[SurfaceTemplate]:
        return self._concepts_to_templates[concept]

    def log_hypotheses(self, log_output_path: Path) -> None:
        pass

    @_concepts_to_templates.default
    def _init_concepts_to_templates(
        self
    ) -> ImmutableSetMultiDict[Concept, SurfaceTemplate]:
        # Ground is added explicitly to this list because the code
        # Which matches the ground matches by recognition and not shape
        # See: `ObjectRecognizer.match_objects`
        return immutablesetmultidict(
            (concept, SurfaceTemplate.for_object_name(name))
            for (concept, name) in (
                list(
                    self._object_recognizer._concepts_to_names.items()  # pylint:disable=protected-access
                )
                + [(GROUND_OBJECT_CONCEPT, "ground")]
            )
        )
