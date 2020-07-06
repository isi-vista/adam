import logging
from itertools import chain
from typing import AbstractSet, Iterable, List, Mapping, Sequence, Set, Tuple, Union
from adam.language_specific.chinese.chinese_phase_1_lexicon import (
    GAILA_PHASE_1_CHINESE_LEXICON,
)
from adam.learner.language_mode import LanguageMode
from contexttimer import Timer
from more_itertools import first
from networkx import DiGraph

from adam.axes import GRAVITATIONAL_DOWN_TO_UP_AXIS, LEARNER_AXES, WORLD_AXES
from adam.language import LinguisticDescription
from adam.learner.alignments import (
    LanguageConceptAlignment,
    LanguagePerceptionSemanticAlignment,
    PerceptionSemanticAlignment,
)
from adam.ontology import OntologyNode
from adam.ontology.ontology import Ontology
from adam.ontology.phase1_ontology import (
    GAILA_PHASE_1_ONTOLOGY,
    PART_OF,
    PHASE_1_CURRICULUM_OBJECTS,
    BIGGER_THAN,
    SMALLER_THAN,
)
from adam.perception import (
    GROUND_PERCEPTION,
    LEARNER_PERCEPTION,
    ObjectPerception,
    MatchMode,
)
from adam.perception.deprecated import LanguageAlignedPerception
from adam.perception.perception_graph import (
    AnyObjectPerception,
    ENTIRE_SCENE,
    EdgeLabel,
    HAS_PROPERTY_LABEL,
    PerceptionGraph,
    PerceptionGraphNode,
    PerceptionGraphPattern,
    PerceptionGraphPatternMatch,
    TemporallyScopedEdgeLabel,
    edge_equals_ignoring_temporal_scope,
    raise_graph_exception,
)
from adam.semantics import (
    Concept,
    ObjectConcept,
    ObjectSemanticNode,
    GROUND_OBJECT_CONCEPT,
)
from adam.utils.networkx_utils import subgraph
from attr import attrib, attrs
from attr.validators import deep_iterable, deep_mapping, instance_of
from immutablecollections import ImmutableDict, ImmutableSet, immutabledict, immutableset
from immutablecollections.converter_utils import _to_immutabledict, _to_immutableset
from vistautils.span import Span

_LIST_OF_PERCEIVED_PATTERNS = immutableset(
    (
        node.handle,
        PerceptionGraphPattern.from_schema(
            first(GAILA_PHASE_1_ONTOLOGY.structural_schemata(node))
        ),
    )
    for node in PHASE_1_CURRICULUM_OBJECTS
    if node
    in GAILA_PHASE_1_ONTOLOGY._structural_schemata.keys()  # pylint:disable=protected-access
)


@attrs(frozen=True, slots=True, auto_attribs=True)
class PerceptionGraphFromObjectRecognizer:
    """
    See `ObjectRecognizer.match_objects`
    """

    perception_graph: PerceptionGraph
    description_to_matched_object_node: ImmutableDict[
        Tuple[str, ...], ObjectSemanticNode
    ] = attrib(converter=_to_immutabledict)

    def __attrs_post_init__(self) -> None:
        matched_object_nodes = set(
            node
            for node in self.perception_graph.copy_as_digraph()
            if isinstance(node, ObjectSemanticNode)
        )
        described_matched_object_nodes = set(
            self.description_to_matched_object_node.values()
        )
        if matched_object_nodes != described_matched_object_nodes:
            raise RuntimeError(
                f"A matched object node should be present in the graph"
                f"if and only if it is described. Got matches objects "
                f"{matched_object_nodes} but those described were {described_matched_object_nodes}"
            )


# these are shared aspects of the world which, although they might be referenced by
# object recognition patterns, should not be deleted when those patterns are match.
# For example, a geon axis local to an object is no longer needed when the object
# has been recognized, but we still need the gravitational axes
SHARED_WORLD_ITEMS = set(
    chain([GRAVITATIONAL_DOWN_TO_UP_AXIS], WORLD_AXES.all_axes, LEARNER_AXES.all_axes)
)


# Used by ObjectRecognizer below.
# See: https://github.com/isi-vista/adam/issues/648
def _sort_mapping_by_pattern_complexity(
    pairs
) -> ImmutableDict[str, PerceptionGraphPattern]:
    # we type: ignore because the proper typing of pairs is huge and mypy is going to screw it up
    # anyway.
    unsorted = immutabledict(pairs)  # type: ignore
    return immutabledict(
        (string, pattern)
        for (string, pattern) in sorted(
            unsorted.items(),
            key=lambda item: len(item[1]._graph.nodes),  # pylint:disable=protected-access
            reverse=True,
        )
    )


cumulative_millis_in_successful_matches_ms = 0  # pylint: disable=invalid-name
cumulative_millis_in_failed_matches_ms = 0  # pylint: disable=invalid-name


@attrs(frozen=True)
class ObjectRecognizer:
    """
    The ObjectRecognizer finds object matches in the scene pattern and adds a `ObjectSemanticNodePerceptionPredicate`
    which can be used to learn additional semantics which relate objects to other objects

    If applied to a dynamic situation, this will only recognize objects
    which are present in both the BEFORE and AFTER frames.
    """

    # Because static patterns must be applied to static perceptions
    # and dynamic patterns to dynamic situations,
    # we need to store our patterns both ways.
    _concepts_to_static_patterns: ImmutableDict[
        ObjectConcept, PerceptionGraphPattern
    ] = attrib(
        validator=deep_mapping(
            instance_of(ObjectConcept), instance_of(PerceptionGraphPattern)
        ),
        converter=_to_immutabledict,
    )
    _concepts_to_names: ImmutableDict[ObjectConcept, str] = attrib(
        validator=deep_mapping(instance_of(ObjectConcept), instance_of(str)),
        converter=_to_immutabledict,
    )

    # We derive these from the static patterns.
    _concepts_to_dynamic_patterns: ImmutableDict[
        ObjectConcept, PerceptionGraphPattern
    ] = attrib(init=False)
    determiners: ImmutableSet[str] = attrib(
        converter=_to_immutableset, validator=deep_iterable(instance_of(str))
    )
    """
    This is a hack to handle determiners.
    See https://github.com/isi-vista/adam/issues/498
    """
    _concept_to_num_subobjects: ImmutableDict[Concept, int] = attrib(init=False)
    """
    Used for a performance optimization in match_objects.
    """
    _language_mode: LanguageMode = attrib(
        validator=instance_of(LanguageMode), kw_only=True
    )

    def __attrs_post_init__(self) -> None:
        non_lowercase_determiners = [
            determiner
            for determiner in self.determiners
            if determiner.lower() != determiner
        ]
        if non_lowercase_determiners:
            raise RuntimeError(
                f"All determiners must be specified in lowercase, but got "
                f"{non_lowercase_determiners}"
            )

    @staticmethod
    def for_ontology_types(
        ontology_types: Iterable[OntologyNode],
        determiners: Iterable[str],
        ontology: Ontology,
        language_mode: LanguageMode,
    ) -> "ObjectRecognizer":
        ontology_types_to_concepts = {
            obj_type: ObjectConcept(obj_type.handle) for obj_type in ontology_types
        }

        return ObjectRecognizer(
            concepts_to_static_patterns=_sort_mapping_by_pattern_complexity(
                immutabledict(
                    (
                        concept,
                        PerceptionGraphPattern.from_ontology_node(obj_type, ontology),
                    )
                    for (obj_type, concept) in ontology_types_to_concepts.items()
                )
            ),
            determiners=determiners,
            concepts_to_names={
                concept: obj_type.handle
                for obj_type, concept in ontology_types_to_concepts.items()
            },
            language_mode=language_mode,
        )

    def match_objects_old(
        self, perception_graph: PerceptionGraph
    ) -> PerceptionGraphFromObjectRecognizer:
        new_style_input = PerceptionSemanticAlignment(
            perception_graph=perception_graph, semantic_nodes=[]
        )
        new_style_output = self.match_objects(new_style_input)
        return PerceptionGraphFromObjectRecognizer(
            perception_graph=new_style_output[0].perception_graph,
            description_to_matched_object_node=new_style_output[1],
        )

    def match_objects(
        self, perception_semantic_alignment: PerceptionSemanticAlignment
    ) -> Tuple[PerceptionSemanticAlignment, Mapping[Tuple[str, ...], ObjectSemanticNode]]:
        r"""
        Recognize known objects in a `PerceptionGraph`.

        The matched portion of the graph will be replaced with an `ObjectSemanticNode`\ s
        which will inherit all relationships of any nodes internal to the matched portion
        with any external nodes.

        This is useful as a pre-processing step
        before prepositional and verbal learning experiments.
        """
        # pylint: disable=global-statement,invalid-name
        global cumulative_millis_in_successful_matches_ms
        global cumulative_millis_in_failed_matches_ms

        object_nodes: List[Tuple[Tuple[str, ...], ObjectSemanticNode]] = []
        perception_graph = perception_semantic_alignment.perception_graph
        is_dynamic = perception_semantic_alignment.perception_graph.dynamic

        if is_dynamic:
            concepts_to_patterns = self._concepts_to_dynamic_patterns
        else:
            concepts_to_patterns = self._concepts_to_static_patterns

        # We special case handling the ground perception
        # Because we don't want to remove it from the graph, we just want to use it's
        # Object node as a recognized object. The situation "a box on the ground"
        # Prompted the need to recognize the ground
        graph_to_return = perception_graph
        for node in graph_to_return._graph.nodes:  # pylint:disable=protected-access
            if node == GROUND_PERCEPTION:
                matched_object_node = ObjectSemanticNode(GROUND_OBJECT_CONCEPT)
                if LanguageMode.ENGLISH == self._language_mode:
                    object_nodes.append(
                        ((f"{GROUND_OBJECT_CONCEPT.debug_string}",), matched_object_node)
                    )
                elif LanguageMode.CHINESE == self._language_mode:
                    object_nodes.append((("di4 myan4",), matched_object_node))
                else:
                    raise RuntimeError("Invalid language_generator")
                # We construct a fake match which is only the ground perception node
                subgraph_of_root = subgraph(perception_graph.copy_as_digraph(), [node])
                pattern_match = PerceptionGraphPatternMatch(
                    matched_pattern=PerceptionGraphPattern(
                        graph=subgraph_of_root, dynamic=perception_graph.dynamic
                    ),
                    graph_matched_against=perception_graph,
                    matched_sub_graph=PerceptionGraph(
                        graph=subgraph_of_root, dynamic=perception_graph.dynamic
                    ),
                    pattern_node_to_matched_graph_node=immutabledict(),
                )
                graph_to_return = replace_match_with_object_graph_node(
                    matched_object_node, graph_to_return, pattern_match
                )

        candidate_object_subgraphs = extract_candidate_objects(perception_graph)

        for candidate_object_graph in candidate_object_subgraphs:

            num_object_nodes = candidate_object_graph.count_nodes_matching(
                lambda node: isinstance(node, ObjectPerception)
            )
            for (concept, pattern) in concepts_to_patterns.items():
                # As an optimization, we count how many sub-object nodes
                # are in the graph and the pattern.
                # If they aren't the same, the match is impossible
                # and we can bail out early.
                if num_object_nodes != self._concept_to_num_subobjects[concept]:
                    continue

                with Timer(factor=1000) as t:
                    matcher = pattern.matcher(
                        candidate_object_graph, match_mode=MatchMode.OBJECT
                    )
                    pattern_match = first(
                        matcher.matches(use_lookahead_pruning=True), None
                    )
                if pattern_match:
                    cumulative_millis_in_successful_matches_ms += t.elapsed

                    matched_object_node = ObjectSemanticNode(concept)

                    # We wrap the concept in a tuple because it could in theory be multiple
                    # tokens,
                    # even though currently it never is.
                    if self._language_mode == LanguageMode.ENGLISH:
                        object_nodes.append(
                            ((concept.debug_string,), matched_object_node)
                        )
                    elif self._language_mode == LanguageMode.CHINESE:
                        mappings = (
                            GAILA_PHASE_1_CHINESE_LEXICON._ontology_node_to_word  # pylint:disable=protected-access
                        )
                        for k, v in mappings.items():
                            if k.handle == concept.debug_string:
                                debug_string = str(v.base_form)
                                object_nodes.append(
                                    ((debug_string,), matched_object_node)
                                )

                    graph_to_return = replace_match_with_object_graph_node(
                        matched_object_node, graph_to_return, pattern_match
                    )
                    # We match each candidate objects against only one object type.
                    # See https://github.com/isi-vista/adam/issues/627
                    break
                else:
                    cumulative_millis_in_failed_matches_ms += t.elapsed
        if object_nodes:
            logging.info(
                "Object recognizer recognized: %s",
                [concept for (concept, _) in object_nodes],
            )
        logging.info(
            "object matching: ms in success: %s, ms in failed: %s",
            cumulative_millis_in_successful_matches_ms,
            cumulative_millis_in_failed_matches_ms,
        )
        semantic_object_nodes = immutableset(node for (_, node) in object_nodes)
        return (
            perception_semantic_alignment.copy_with_updated_graph_and_added_nodes(
                new_graph=graph_to_return, new_nodes=semantic_object_nodes
            ),
            immutabledict(object_nodes),
        )

    def match_objects_with_language_old(
        self, language_aligned_perception: LanguageAlignedPerception
    ) -> LanguageAlignedPerception:
        if language_aligned_perception.node_to_language_span:
            raise RuntimeError(
                "Don't know how to handle a non-empty node-to-language-span"
            )
        new_style_input = LanguagePerceptionSemanticAlignment(
            language_concept_alignment=LanguageConceptAlignment(
                language_aligned_perception.language, node_to_language_span=[]
            ),
            perception_semantic_alignment=PerceptionSemanticAlignment(
                perception_graph=language_aligned_perception.perception_graph,
                semantic_nodes=[],
            ),
        )
        new_style_output = self.match_objects_with_language(new_style_input)
        return LanguageAlignedPerception(
            language=new_style_output.language_concept_alignment.language,
            perception_graph=new_style_output.perception_semantic_alignment.perception_graph,
            node_to_language_span=new_style_output.language_concept_alignment.node_to_language_span,
        )

    def match_objects_with_language(
        self, language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment
    ) -> LanguagePerceptionSemanticAlignment:
        """
        Recognize known objects in a `LanguagePerceptionSemanticAlignment`.

        For each node matched, this will identify the relevant portion of the linguistic input
        and record the correspondence.

        The matched portion of the graph will be replaced with an `ObjectSemanticNode`
        which will inherit all relationships of any nodes internal to the matched portion
        with any external nodes.

        This is useful as a pre-processing step
        before prepositional and verbal learning experiments.
        """
        if (
            language_perception_semantic_alignment.perception_semantic_alignment.semantic_nodes
        ):
            raise RuntimeError(
                "We assume ObjectRecognizer is run first, with no previous " "alignments"
            )

        (
            post_match_perception_semantic_alignment,
            tokens_to_object_nodes,
        ) = self.match_objects(
            language_perception_semantic_alignment.perception_semantic_alignment
        )
        return LanguagePerceptionSemanticAlignment(
            language_concept_alignment=language_perception_semantic_alignment.language_concept_alignment.copy_with_added_token_alignments(
                self._align_objects_to_tokens(
                    tokens_to_object_nodes,
                    language_perception_semantic_alignment.language_concept_alignment.language,
                )
            ),
            perception_semantic_alignment=post_match_perception_semantic_alignment,
        )

    def _align_objects_to_tokens(
        self,
        description_to_object_node: Mapping[Tuple[str, ...], ObjectSemanticNode],
        language: LinguisticDescription,
    ) -> Mapping[ObjectSemanticNode, Span]:
        result: List[Tuple[ObjectSemanticNode, Span]] = []

        # We want to ban the same token index from being aligned twice.
        matched_token_indices: Set[int] = set()

        for (description_tuple, object_node) in description_to_object_node.items():
            if len(description_tuple) != 1:
                raise RuntimeError(
                    f"Multi-token descriptions are not yet supported:"
                    f"{description_tuple}"
                )
            description = description_tuple[0]
            try:
                end_index_inclusive = language.index(description)
            except ValueError:
                # A scene might contain things which are not referred to by the associated language.
                continue

            start_index = end_index_inclusive
            # This is a somewhat language-dependent hack to gobble up preceding determiners.
            # See https://github.com/isi-vista/adam/issues/498 .
            if end_index_inclusive > 0:
                possible_determiner_index = end_index_inclusive - 1
                if language[possible_determiner_index].lower() in self.determiners:
                    start_index = possible_determiner_index

            # We record what tokens were covered so we can block the same tokens being used twice.
            for included_token_index in range(start_index, end_index_inclusive + 1):
                if included_token_index in matched_token_indices:
                    raise RuntimeError(
                        "We do not currently support the same object "
                        "being mentioned twice in a sentence."
                    )
                matched_token_indices.add(included_token_index)

            result.append(
                (
                    object_node,
                    language.span(
                        start_index, end_index_exclusive=end_index_inclusive + 1
                    ),
                )
            )
        return immutabledict(result)

    @_concepts_to_dynamic_patterns.default
    def _init_concepts_to_dynamic_patterns(
        self
    ) -> ImmutableDict[ObjectConcept, PerceptionGraphPattern]:
        return immutabledict(
            (concept, static_pattern.copy_with_temporal_scopes(ENTIRE_SCENE))
            for (concept, static_pattern) in self._concepts_to_static_patterns.items()
        )

    @_concept_to_num_subobjects.default
    def _init_patterns_to_num_subobjects(self) -> ImmutableDict[ObjectConcept, int]:
        return immutabledict(
            (
                concept,
                pattern.count_nodes_matching(
                    lambda node: isinstance(node, AnyObjectPerception)
                ),
            )
            for (concept, pattern) in self._concepts_to_static_patterns.items()
        )


def extract_candidate_objects(
    whole_scene_perception_graph: PerceptionGraph
) -> Sequence[PerceptionGraph]:
    """
    Pulls out distinct objects from a scene.

    We will attempt to recognize only these and will ignore other parts of the scene.
    """
    scene_digraph = whole_scene_perception_graph.copy_as_digraph()

    def is_part_of_label(label) -> bool:
        return label == PART_OF or (
            isinstance(label, TemporallyScopedEdgeLabel) and label.attribute == PART_OF
        )

    # We first identify root object nodes, which are object nodes with no part-of
    # relationship with other object nodes.
    def is_root_object_node(node) -> bool:
        if isinstance(node, ObjectPerception):
            for (_, _, edge_label) in scene_digraph.out_edges(node, data="label"):
                if is_part_of_label(edge_label):
                    # This object node is part of another object and cannot be a root.
                    return False
            return True
        return False

    candidate_object_root_nodes = [
        node
        for node in scene_digraph.nodes
        if is_root_object_node(node)
        and node not in (GROUND_PERCEPTION, LEARNER_PERCEPTION)
    ]

    candidate_objects: List[PerceptionGraph] = []
    for root_object_node in candidate_object_root_nodes:
        # Having identified the root nodes of the candidate objects,
        # we now gather all sub-object nodes.
        object_nodes_in_object_list = []
        nodes_to_examine = [root_object_node]

        # This would be clearer recursively
        # but I'm betting this implementation is a bit faster in Python.

        nodes_visited: Set[PerceptionGraphNode] = set()
        while nodes_to_examine:
            node_to_examine = nodes_to_examine.pop()
            if node_to_examine in nodes_visited:
                continue
            nodes_visited.add(node_to_examine)
            object_nodes_in_object_list.append(node_to_examine)
            for (next_node, _, edge_label) in scene_digraph.in_edges(
                node_to_examine, data="label"
            ):
                if is_part_of_label(edge_label):
                    nodes_to_examine.append(next_node)
        object_nodes_in_object = immutableset(object_nodes_in_object_list)

        # Now we know all object nodes for this candidate object.
        # Finally, we find the sub-graph to match against which could possibly correspond
        # to this candidate object
        # by performing a BFS over the graph
        # but *stopping whenever we encounter an object node
        # which is not part of this candidate object*.
        # This is a little more generous than we need to be, but it's simple.
        nodes_to_examine = [root_object_node]
        candidate_subgraph_nodes = []
        nodes_visited.clear()
        while nodes_to_examine:
            node_to_examine = nodes_to_examine.pop()
            is_allowable_node = (
                not isinstance(node_to_examine, ObjectPerception)
                or node_to_examine in object_nodes_in_object
            )
            if node_to_examine not in nodes_visited and is_allowable_node:
                nodes_visited.add(node_to_examine)
                candidate_subgraph_nodes.append(node_to_examine)
                nodes_to_examine.extend(
                    out_neighbor
                    for (_, out_neighbor) in scene_digraph.out_edges(node_to_examine)
                )
                nodes_to_examine.extend(
                    in_neighbor
                    for (in_neighbor, _) in scene_digraph.in_edges(node_to_examine)
                )
        candidate_objects.append(
            whole_scene_perception_graph.subgraph_by_nodes(
                immutableset(candidate_subgraph_nodes)
            )
        )
    return candidate_objects


def _get_edge_label(
    graph: DiGraph, source: PerceptionGraphNode, target: PerceptionGraphNode
) -> EdgeLabel:
    return graph.get_edge_data(source, target)["label"]


def _get_root_object_perception(
    graph: DiGraph, matched_subgraph_nodes: AbstractSet[PerceptionGraphNode]
) -> ObjectPerception:
    matched_object_perceptions = immutableset(
        node for node in matched_subgraph_nodes if isinstance(node, ObjectPerception)
    )
    roots = [
        node
        for node in matched_object_perceptions
        if not (
            any(succ in matched_object_perceptions for succ in graph.successors(node))
        )
    ]
    if len(roots) == 1:
        return roots[0]
    elif roots:
        raise RuntimeError(f"Got multiple roots for object match: {roots}")
    else:
        raise RuntimeError(
            f"Could not find a root for object match: {matched_subgraph_nodes}"
        )


def _add_external_properties_linked_to_root_object_perception(
    *,
    original_graph: DiGraph,
    output_graph: DiGraph,
    matched_nodes: AbstractSet[PerceptionGraphNode],
    matched_object_node: ObjectSemanticNode,
) -> None:
    # We take two graphs as input because we are assuming object-internal properties
    # have already been deleted from the output_graph, so we have to look for them
    # in the original, unaltered graph
    linked_properties_and_labels: List[Tuple[PerceptionGraphNode, EdgeLabel]] = []
    root_node = _get_root_object_perception(original_graph, matched_nodes)
    for succ in original_graph.successors(root_node):
        edge_label = _get_edge_label(original_graph, root_node, succ)
        if edge_equals_ignoring_temporal_scope(edge_label, HAS_PROPERTY_LABEL):
            linked_properties_and_labels.append((succ, edge_label))
    for (linked_property_node, edge_label) in linked_properties_and_labels:
        output_graph.add_edge(matched_object_node, linked_property_node, label=edge_label)


def _add_relationships_linked_to_root_object_perception(
    *,
    original_graph: DiGraph,
    output_graph: DiGraph,
    matched_nodes: AbstractSet[PerceptionGraphNode],
    matched_object_node: ObjectSemanticNode,
) -> None:
    # We take two graphs as input because we are assuming object-internal properties
    # have already been deleted from the output_graph, so we have to look for them
    # in the original, unaltered graph
    root_node = _get_root_object_perception(original_graph, matched_nodes)

    linked_properties_and_labels: List[Tuple[PerceptionGraphNode, EdgeLabel]] = []
    for pred in original_graph.predecessors(root_node):
        edge_label = _get_edge_label(original_graph, pred, root_node)
        if edge_equals_ignoring_temporal_scope(edge_label, HAS_PROPERTY_LABEL):
            linked_properties_and_labels.append((pred, edge_label))
    for (linked_property_node, edge_label) in linked_properties_and_labels:
        output_graph.add_edge(linked_property_node, matched_object_node, label=edge_label)

    linked_properties_and_labels.clear()
    for succ in original_graph.successors(root_node):
        edge_label = _get_edge_label(original_graph, root_node, succ)
        if edge_equals_ignoring_temporal_scope(edge_label, HAS_PROPERTY_LABEL):
            linked_properties_and_labels.append((succ, edge_label))
    for (linked_property_node, edge_label) in linked_properties_and_labels:
        output_graph.add_edge(matched_object_node, linked_property_node, label=edge_label)


def replace_match_with_object_graph_node(
    matched_object_node: ObjectSemanticNode,
    current_perception: PerceptionGraph,
    pattern_match: PerceptionGraphPatternMatch,
) -> PerceptionGraph:
    """
    Internal function to replace the nodes of the perception matched by the object pattern
    with an `ObjectSemanticNode`.

    Any external relationships those nodes had is inherited by the `ObjectSemanticNode`.
    """
    perception_digraph = current_perception.copy_as_digraph()
    perception_digraph.add_node(matched_object_node)

    matched_subgraph_nodes: ImmutableSet[PerceptionGraphNode] = immutableset(
        pattern_match.matched_sub_graph._graph.nodes,  # pylint:disable=protected-access
        disable_order_check=True,
    )

    # Multiple sub-objects of a matched object may link to the same property
    # (for example, to a color shared by all the parts).
    # In this case, we want the shared object node to link to this property only once.
    external_properties: Set[Union[OntologyNode, ObjectSemanticNode]] = set()
    duplicate_nodes_to_remove: List[PerceptionGraphNode] = []

    for matched_subgraph_node in matched_subgraph_nodes:
        if isinstance(matched_subgraph_node, ObjectSemanticNode):
            raise RuntimeError(
                f"We do not currently allow object recognitions to themselves "
                f"operate over other object recognitions, but got match "
                f"{pattern_match.matched_sub_graph}"
            )

        # A pattern might refer to shared parts of the world like the learner
        # or the ground, and we don't want the replacement root to inherit the
        # shared world item's relationships.
        if matched_subgraph_node in SHARED_WORLD_ITEMS:
            continue

        # If there is an edge from the matched sub-graph to a node outside it,
        # also add an edge from the object match node to that node.
        for matched_subgraph_node_successor in perception_digraph.successors(
            matched_subgraph_node
        ):
            edge_label = _get_edge_label(
                perception_digraph, matched_subgraph_node, matched_subgraph_node_successor
            )

            # don't want to add edges which are internal to the matched sub-graph
            if matched_subgraph_node_successor not in matched_subgraph_nodes:
                if edge_equals_ignoring_temporal_scope(edge_label, HAS_PROPERTY_LABEL):
                    # Prevent multiple `has-property` assertions to the same color node
                    # On a recognized object
                    # Also prevent size relations from being inherited on root object.
                    if matched_subgraph_node_successor[
                        0
                    ] in external_properties or matched_subgraph_node_successor[0] in {
                        SMALLER_THAN,
                        BIGGER_THAN,
                    }:
                        if (
                            perception_digraph.degree(matched_subgraph_node_successor)
                            != 1
                        ):
                            raise_graph_exception(
                                f"Node {matched_subgraph_node_successor} "
                                f"appears to be a duplicate property node, "
                                f"but has degree != 1",
                                current_perception,
                            )
                        duplicate_nodes_to_remove.append(matched_subgraph_node_successor)
                        continue
                    else:
                        external_properties.add(matched_subgraph_node_successor[0])

                perception_digraph.add_edge(
                    matched_object_node, matched_subgraph_node_successor, label=edge_label
                )

        # If there is an edge to the matched sub-graph from a node outside it,
        # also add an edge to the object match node from that node.
        for matched_subgraph_node_predecessor in perception_digraph.predecessors(
            matched_subgraph_node
        ):
            edge_label = _get_edge_label(
                perception_digraph,
                matched_subgraph_node_predecessor,
                matched_subgraph_node,
            )

            # don't want to add edges which are internal to the matched sub-graph
            if matched_subgraph_node_predecessor not in matched_subgraph_nodes:
                if edge_equals_ignoring_temporal_scope(edge_label, HAS_PROPERTY_LABEL):
                    # Prevent multiple `has-property` assertions to the same color node
                    # On a recognized object
                    if isinstance(matched_subgraph_node_predecessor, ObjectSemanticNode):
                        prop = matched_subgraph_node_predecessor
                    else:
                        prop = matched_subgraph_node_predecessor[0]
                    if prop in external_properties:
                        if (
                            perception_digraph.degree(matched_subgraph_node_predecessor)
                            != 1
                        ):
                            raise_graph_exception(
                                f"Node {matched_subgraph_node_predecessor} "
                                f"appears to be a duplicate property node, "
                                f"but has degree != 1",
                                current_perception,
                            )
                        duplicate_nodes_to_remove.append(
                            matched_subgraph_node_predecessor
                        )
                        continue
                    else:
                        external_properties.add(prop)

                perception_digraph.add_edge(
                    matched_subgraph_node_predecessor,
                    matched_object_node,
                    label=edge_label,
                )

    # Remove all matched nodes which are not shared world items (e.g. gravity, the learner)
    perception_digraph.remove_nodes_from(
        matched_node
        for matched_node in matched_subgraph_nodes
        if matched_node not in SHARED_WORLD_ITEMS
    )
    perception_digraph.remove_nodes_from(duplicate_nodes_to_remove)

    # We want to re-add any properties linked directly to the root node of an object.
    # Example: water is a liquid
    # These may be relevant to learning verb semantics
    # (e.g. you can only drink a liquid)
    _add_external_properties_linked_to_root_object_perception(
        original_graph=current_perception.copy_as_digraph(),
        output_graph=perception_digraph,
        matched_nodes=matched_subgraph_nodes,
        matched_object_node=matched_object_node,
    )

    return PerceptionGraph(perception_digraph, dynamic=current_perception.dynamic)


def replace_match_root_with_object_semantic_node(
    object_semantic_node: ObjectSemanticNode,
    current_perception: PerceptionGraph,
    pattern_match: PerceptionGraphPatternMatch,
) -> PerceptionGraph:
    """
    Internal function to replace the root node of the perception matched by the object pattern
    with an `ObjectSemanticNode`.

    The `ObjectSemanticNode` inherits both the root node's internal relationships and all external
    relationships involving either the root node or its children.
    """
    perception_digraph = current_perception.copy_as_digraph()
    perception_digraph.add_node(object_semantic_node)

    matched_subgraph_nodes: ImmutableSet[PerceptionGraphNode] = immutableset(
        pattern_match.matched_sub_graph._graph.nodes,  # pylint:disable=protected-access
        disable_order_check=True,
    )

    root = _get_root_object_perception(perception_digraph, matched_subgraph_nodes)

    # Multiple sub-objects of a matched object may link to the same property
    # (for example, to a color shared by all the parts).
    # In this case, we want the shared object node to link to this property only once.
    external_properties: Set[Union[OntologyNode, ObjectSemanticNode]] = set()

    for matched_subgraph_node in matched_subgraph_nodes:
        if isinstance(matched_subgraph_node, ObjectSemanticNode):
            raise RuntimeError(
                f"We do not currently allow object recognitions to themselves "
                f"operate over other object recognitions, but got match "
                f"{pattern_match.matched_sub_graph}"
            )

        # A pattern might refer to shared parts of the world like the learner
        # or the ground, and we don't want the replacement root to inherit the
        # shared world item's relationships.
        if matched_subgraph_node in SHARED_WORLD_ITEMS:
            continue

        # If there is an edge from the matched sub-graph to a node outside it,
        # also add an edge from the object match node to that node.
        for matched_subgraph_node_successor in perception_digraph.successors(
            matched_subgraph_node
        ):
            edge_label = _get_edge_label(
                perception_digraph, matched_subgraph_node, matched_subgraph_node_successor
            )

            # don't want to add edges which are internal to the matched sub-graph
            if matched_subgraph_node_successor not in matched_subgraph_nodes:
                if edge_equals_ignoring_temporal_scope(edge_label, HAS_PROPERTY_LABEL):
                    # Prevent multiple `has-property` assertions to the same color node
                    # On a recognized object
                    if matched_subgraph_node_successor[
                        0
                    ] in external_properties or matched_subgraph_node_successor[0] in {
                        SMALLER_THAN,
                        BIGGER_THAN,
                    }:
                        if (
                            perception_digraph.degree(matched_subgraph_node_successor)
                            != 1
                        ):
                            raise_graph_exception(
                                f"Node {matched_subgraph_node_successor} "
                                f"appears to be a duplicate property node, "
                                f"but has degree != 1",
                                current_perception,
                            )
                        continue
                    else:
                        external_properties.add(matched_subgraph_node_successor[0])

                perception_digraph.add_edge(
                    object_semantic_node,
                    matched_subgraph_node_successor,
                    label=edge_label,
                )

        # If there is an edge to the matched sub-graph from a node outside it,
        # also add an edge to the object match node from that node.
        for matched_subgraph_node_predecessor in perception_digraph.predecessors(
            matched_subgraph_node
        ):
            edge_label = _get_edge_label(
                perception_digraph,
                matched_subgraph_node_predecessor,
                matched_subgraph_node,
            )

            # don't want to add edges which are internal to the matched sub-graph
            if matched_subgraph_node_predecessor not in matched_subgraph_nodes:
                if edge_equals_ignoring_temporal_scope(edge_label, HAS_PROPERTY_LABEL):
                    # Prevent multiple `has-property` assertions to the same color node
                    # On a recognized object
                    if isinstance(matched_subgraph_node_predecessor, ObjectSemanticNode):
                        prop = matched_subgraph_node_predecessor
                    else:
                        prop = matched_subgraph_node_predecessor[0]
                    if prop in external_properties:
                        if (
                            perception_digraph.degree(matched_subgraph_node_predecessor)
                            != 1
                        ):
                            raise_graph_exception(
                                f"Node {matched_subgraph_node_predecessor} "
                                f"appears to be a duplicate property node, "
                                f"but has degree != 1",
                                current_perception,
                            )
                        continue
                    else:
                        external_properties.add(prop)

                perception_digraph.add_edge(
                    matched_subgraph_node_predecessor,
                    object_semantic_node,
                    label=edge_label,
                )

    if root in SHARED_WORLD_ITEMS:
        raise RuntimeError(
            f"Pattern match root cannot be a shared world item, "
            f"but got match {pattern_match}"
        )

    perception_digraph.remove_node(root)

    # We want to re-add any relationships linked directly to the root node of an object.
    # Example: water is a liquid
    # Example: this hand is a part of this person
    # These may be relevant to learning verb semantics
    # (e.g. you can only drink a liquid)
    _add_relationships_linked_to_root_object_perception(
        original_graph=current_perception.copy_as_digraph(),
        output_graph=perception_digraph,
        matched_nodes=matched_subgraph_nodes,
        matched_object_node=object_semantic_node,
    )

    return PerceptionGraph(perception_digraph, dynamic=current_perception.dynamic)
