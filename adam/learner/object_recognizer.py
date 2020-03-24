import logging
from itertools import chain
from typing import Iterable, List, Mapping, Set, Tuple, Sequence

from contexttimer import Timer

from adam.perception import ObjectPerception, GROUND_PERCEPTION, LEARNER_PERCEPTION
from attr import attrib, attrs
from attr.validators import deep_iterable, deep_mapping, instance_of
from immutablecollections import ImmutableDict, ImmutableSet, immutabledict, immutableset
from immutablecollections.converter_utils import _to_immutabledict, _to_immutableset
from more_itertools import first
from vistautils.span import Span

from adam.axes import GRAVITATIONAL_DOWN_TO_UP_AXIS, LEARNER_AXES, WORLD_AXES
from adam.language import LinguisticDescription
from adam.ontology import OntologyNode
from adam.ontology.ontology import Ontology
from adam.ontology.phase1_ontology import (
    GAILA_PHASE_1_ONTOLOGY,
    PHASE_1_CURRICULUM_OBJECTS,
    PART_OF,
)
from adam.perception.perception_graph import (
    LanguageAlignedPerception,
    MatchedObjectNode,
    PerceptionGraph,
    PerceptionGraphNode,
    PerceptionGraphPattern,
    PerceptionGraphPatternMatch,
    RelationTypeIsPredicate,
    ENTIRE_SCENE,
    TemporallyScopedEdgeLabel,
    AnyObjectPerception,
)

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

MATCHED_OBJECT_PATTERN_LABEL = OntologyNode("has-matched-object-pattern")


@attrs(frozen=True, slots=True, auto_attribs=True)
class PerceptionGraphFromObjectRecognizer:
    """
    See `ObjectRecognizer.match_objects`
    """

    perception_graph: PerceptionGraph
    description_to_matched_object_node: ImmutableDict[
        Tuple[str, ...], MatchedObjectNode
    ] = attrib(converter=_to_immutabledict)

    def __attrs_post_init__(self) -> None:
        matched_object_nodes = set(
            node
            for node in self.perception_graph.copy_as_digraph()
            if isinstance(node, MatchedObjectNode)
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
    The ObjectRecognizer finds object matches in the scene pattern and adds a `MatchedObjectPerceptionPredicate`
    which can be used to learn additional semantics which relate objects to other objects

    If applied to a dynamic situation, this will only recognize objects
    which are present in both the BEFORE and AFTER frames.
    """

    # Because static patterns must be applied to static perceptions
    # and dynamic patterns to dynamic situations,
    # we need to store our patterns both ways.
    _object_names_to_static_patterns: ImmutableDict[str, PerceptionGraphPattern] = attrib(
        validator=deep_mapping(instance_of(str), instance_of(PerceptionGraphPattern)),
        converter=_to_immutabledict,
    )
    # We derive these from the static patterns.
    _object_names_to_dynamic_patterns: ImmutableDict[
        str, PerceptionGraphPattern
    ] = attrib(init=False)
    determiners: ImmutableSet[str] = attrib(
        converter=_to_immutableset, validator=deep_iterable(instance_of(str))
    )
    """
    This is a hack to handle determiners.
    See https://github.com/isi-vista/adam/issues/498
    """
    _object_name_to_num_subobjects: ImmutableDict[str, int] = attrib(init=False)
    """
    Used for a performance optimization in match_objects.
    """

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
    ) -> "ObjectRecognizer":
        return ObjectRecognizer(
            object_names_to_static_patterns=_sort_mapping_by_pattern_complexity(
                immutabledict(
                    (
                        obj_type.handle,
                        PerceptionGraphPattern.from_ontology_node(obj_type, ontology),
                    )
                    for obj_type in ontology_types
                )
            ),
            determiners=determiners,
        )

    def match_objects(
        self, perception_graph: PerceptionGraph
    ) -> PerceptionGraphFromObjectRecognizer:
        """
        Recognize known objects in a `PerceptionGraph`.

        The matched portion of the graph will be replaced with an `MatchedObjectNode`
        which will inherit all relationships of any nodes internal to the matched portion
        with any external nodes.

        This is useful as a pre-processing step
        before prepositional and verbal learning experiments.
        """
        # pylint: disable=global-statement,invalid-name
        global cumulative_millis_in_successful_matches_ms
        global cumulative_millis_in_failed_matches_ms

        matched_object_nodes: List[Tuple[Tuple[str, ...], MatchedObjectNode]] = []
        graph_to_return = perception_graph
        is_dynamic = perception_graph.dynamic

        if is_dynamic:
            object_names_to_patterns = self._object_names_to_dynamic_patterns
        else:
            object_names_to_patterns = self._object_names_to_static_patterns

        candidate_object_subgraphs = self.extract_candidate_objects(perception_graph)

        for candidate_object_graph in candidate_object_subgraphs:
            num_object_nodes = candidate_object_graph.count_nodes_matching(
                lambda node: isinstance(node, ObjectPerception)
            )
            for (description, pattern) in object_names_to_patterns.items():
                # As an optimization, we count how many sub-object nodes
                # are in the graph and the pattern.
                # If they aren't the same, the match is impossible
                # and we can bail out early.
                if num_object_nodes != self._object_name_to_num_subobjects[description]:
                    continue

                with Timer(factor=1000) as t:
                    matcher = pattern.matcher(
                        candidate_object_graph, matching_objects=True
                    )
                    pattern_match = first(
                        matcher.matches(use_lookahead_pruning=True), None
                    )
                if pattern_match:
                    cumulative_millis_in_successful_matches_ms += t.elapsed
                    graph_to_return = self._replace_match_with_object_graph_node(
                        graph_to_return, pattern_match, matched_object_nodes, description
                    )
                    # We match each candidate objects against only one object type.
                    # See https://github.com/isi-vista/adam/issues/627
                    break
                else:
                    cumulative_millis_in_failed_matches_ms += t.elapsed
        if matched_object_nodes:
            logging.info(
                "Object recognizer recognized: %s",
                [description for (description, _) in matched_object_nodes],
            )
        logging.info(
            "object matching: ms in success: %s, ms in failed: %s",
            cumulative_millis_in_successful_matches_ms,
            cumulative_millis_in_failed_matches_ms,
        )
        return PerceptionGraphFromObjectRecognizer(graph_to_return, matched_object_nodes)

    def extract_candidate_objects(
        self, whole_scene_perception_graph: PerceptionGraph
    ) -> Sequence[PerceptionGraph]:
        """
        Pulls out distinct objects from a scene.

        We will attempt to recognize only these and will ignore other parts of the scene.
        """
        scene_digraph = whole_scene_perception_graph.copy_as_digraph()

        def is_part_of_label(label) -> bool:
            return label == PART_OF or (
                isinstance(label, TemporallyScopedEdgeLabel)
                and label.attribute == PART_OF
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

    def match_objects_with_language(
        self, language_aligned_perception: LanguageAlignedPerception
    ) -> LanguageAlignedPerception:
        """
        Recognize known objects in a `LanguageAlignedPerception`.

        For each node matched, this will identify the relevant portion of the linguistic input
        and record the correspondence.

        The matched portion of the graph will be replaced with an `MatchedObjectNode`
        which will inherit all relationships of any nodes internal to the matched portion
        with any external nodes.

        This is useful as a pre-processing step
        before prepositional and verbal learning experiments.
        """
        match_result = self.match_objects(language_aligned_perception.perception_graph)
        return LanguageAlignedPerception(
            language=language_aligned_perception.language,
            perception_graph=match_result.perception_graph,
            node_to_language_span=self._align_objects_to_tokens(
                match_result.description_to_matched_object_node,
                language_aligned_perception.language,
            ),
        )

    def _replace_match_with_object_graph_node(
        self,
        current_perception: PerceptionGraph,
        pattern_match: PerceptionGraphPatternMatch,
        matched_object_nodes: List[Tuple[Tuple[str, ...], MatchedObjectNode]],
        description: str,
    ) -> PerceptionGraph:
        """
        Internal function to copy existing relationships from the matched object pattern onto a
        `MatchedObjectPerceptionPredicate`
        """
        matched_object_node = MatchedObjectNode(name=(description,))

        # We wrap the description in a tuple because it could in theory be multiple tokens,
        # even though currently it never is.
        matched_object_nodes.append(((description,), matched_object_node))
        perception_digraph = current_perception.copy_as_digraph()
        perception_digraph.add_node(matched_object_node)
        # We don't want to remove internal property nodes connected to the Object's root
        internal_property_nodes_to_keep: Set[OntologyNode] = set()

        matched_subgraph_nodes: ImmutableSet[PerceptionGraphNode] = immutableset(
            pattern_match.matched_sub_graph._graph.nodes,  # pylint:disable=protected-access
            disable_order_check=True,
        )

        for matched_subgraph_node in matched_subgraph_nodes:
            if isinstance(matched_subgraph_node, MatchedObjectNode):
                raise RuntimeError(
                    f"We do not currently allow object recognitions to themselves "
                    f"operate over other object recognitions, but got match "
                    f"{pattern_match.matched_sub_graph}"
                )

            if matched_subgraph_node in SHARED_WORLD_ITEMS:
                continue

            # We don't want to make multiple links to property nodes from the root node
            linked_property_nodes: Set[OntologyNode] = set()

            # If there is an edge from the matched sub-graph to a node outside it,
            # also add an edge from the object match node to that node.
            for matched_subgraph_node_successor in perception_digraph.successors(
                matched_subgraph_node
            ):
                edge_data = perception_digraph.get_edge_data(
                    matched_subgraph_node, matched_subgraph_node_successor
                )
                label = edge_data["label"]
                # This is the root node of the object
                # We want to keep properties about it that may be internal to
                # to definition of the object. Example: water is a liquid
                if (
                    isinstance(matched_subgraph_node, ObjectPerception)
                    and matched_subgraph_node_successor in matched_subgraph_nodes
                ):
                    if isinstance(label, RelationTypeIsPredicate):
                        if label.dot_label == "rel(" "has-property)":
                            internal_property_nodes_to_keep.add(
                                matched_subgraph_node_successor
                            )
                            perception_digraph.add_edge(
                                matched_object_node,
                                matched_subgraph_node_successor,
                                **edge_data,
                            )

                # don't want to add edges which are internal to the matched sub-graph
                if matched_subgraph_node_successor not in matched_subgraph_nodes:
                    if isinstance(label, RelationTypeIsPredicate):
                        if label.dot_label == "rel(" "has-matched-object-pattern)":
                            raise RuntimeError(
                                f"Overlapping nodes in object recognition: "
                                f"{matched_subgraph_node}, "
                                f"{matched_subgraph_node_successor}"
                            )
                        # Prevent multiple `has-property` assertions to the same color node
                        # On a recognized object
                        elif label.dot_label == "rel(" "has-property)":
                            if (
                                matched_subgraph_node_successor[0]
                                in linked_property_nodes
                            ):
                                continue
                            else:
                                linked_property_nodes.add(
                                    matched_subgraph_node_successor[0]
                                )

                    perception_digraph.add_edge(
                        matched_object_node, matched_subgraph_node_successor, **edge_data
                    )

            # If there is an edge to the matched sub-graph from a node outside it,
            # also add an edge to the object match node from that node.
            for matched_subgraph_node_predecessor in perception_digraph.predecessors(
                matched_subgraph_node
            ):
                edge_data = perception_digraph.get_edge_data(
                    matched_subgraph_node_predecessor, matched_subgraph_node
                )
                label = edge_data["label"]
                # This is the root node of the object
                # We want to keep properties about it that may be internal to
                # to definition of the object. Example: water is a liquid
                if (
                    isinstance(matched_subgraph_node, ObjectPerception)
                    and matched_subgraph_node_predecessor in matched_subgraph_nodes
                ):
                    if isinstance(label, RelationTypeIsPredicate):
                        if label.dot_label == "rel(" "has-property)":
                            internal_property_nodes_to_keep.add(
                                matched_subgraph_node_predecessor
                            )
                            perception_digraph.add_edge(
                                matched_object_node,
                                matched_subgraph_node_predecessor,
                                **edge_data,
                            )

                # don't want to add edges which are internal to the matched sub-graph
                if matched_subgraph_node_predecessor not in matched_subgraph_nodes:
                    if isinstance(label, RelationTypeIsPredicate):
                        if label.dot_label == "rel(" "has-matched-object-pattern)":
                            raise RuntimeError(
                                f"Overlapping nodes in object recognition: "
                                f"{matched_subgraph_node}, "
                                f"{matched_subgraph_node_predecessor}"
                            )
                        # Prevent multiple `has-property` assertions to the same color node
                        # On a recognized object
                        elif label.dot_label == "rel(" "has-property)":
                            if (
                                matched_subgraph_node_predecessor[0]
                                in linked_property_nodes
                            ):
                                continue
                            else:
                                linked_property_nodes.add(
                                    matched_subgraph_node_predecessor[0]
                                )

                    perception_digraph.add_edge(
                        matched_subgraph_node_predecessor,
                        matched_object_node,
                        **edge_data,
                    )

            # we also link every node in the matched sub-graph to the newly introduced node
            # representing the object match.
            # networkx_graph_to_modify_in_place.add_edge(
            #     matched_subgraph_node,
            #     matched_object_node,
            #     label=MATCHED_OBJECT_PATTERN_LABEL,
            # )
        perception_digraph.remove_nodes_from(
            matched_node
            for matched_node in matched_subgraph_nodes
            if matched_node not in SHARED_WORLD_ITEMS
            or matched_node not in internal_property_nodes_to_keep
        )
        return PerceptionGraph(perception_digraph, dynamic=current_perception.dynamic)

    def _align_objects_to_tokens(
        self,
        description_to_object_node: Mapping[Tuple[str, ...], MatchedObjectNode],
        language: LinguisticDescription,
    ) -> Mapping[MatchedObjectNode, Span]:
        result: List[Tuple[MatchedObjectNode, Span]] = []

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

    @_object_names_to_dynamic_patterns.default
    def _init_object_names_to_dynamic_patterns(
        self
    ) -> ImmutableDict[str, PerceptionGraphPattern]:
        return immutabledict(
            (description, static_pattern.copy_with_temporal_scopes(ENTIRE_SCENE))
            for (
                description,
                static_pattern,
            ) in self._object_names_to_static_patterns.items()
        )

    @_object_name_to_num_subobjects.default
    def _init_patterns_to_num_subobjects(self) -> ImmutableDict[str, int]:
        return immutabledict(
            (
                object_name,
                pattern.count_nodes_matching(
                    lambda node: isinstance(node, AnyObjectPerception)
                ),
            )
            for (object_name, pattern) in self._object_names_to_static_patterns.items()
        )
