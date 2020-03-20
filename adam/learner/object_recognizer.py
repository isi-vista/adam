import logging
from itertools import chain
from typing import Iterable, List, Mapping, Set, Tuple

from contexttimer import Timer

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
                "A matched object node should be present in the graph"
                "if and only if it is described"
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


cumulative_millis_in_successful_matches_ms = 0
cumulative_millis_in_failed_matches_ms = 0

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
        global cumulative_millis_in_successful_matches_ms
        global cumulative_millis_in_failed_matches_ms

        matched_object_nodes: List[Tuple[Tuple[str, ...], MatchedObjectNode]] = []
        graph_to_return = perception_graph
        is_dynamic = perception_graph.dynamic

        if is_dynamic:
            object_names_to_patterns = self._object_names_to_dynamic_patterns
        else:
            object_names_to_patterns = self._object_names_to_static_patterns

        for (description, pattern) in object_names_to_patterns.items():
            with Timer(factor=1000) as t:
                matcher = pattern.matcher(graph_to_return, matching_objects=True)
                pattern_match = first(matcher.matches(use_lookahead_pruning=True), None)
            if pattern_match:
                cumulative_millis_in_successful_matches_ms += t.elapsed
            else:
                cumulative_millis_in_failed_matches_ms += t.elapsed

            # It's important not to simply iterate over pattern matches
            # because they might overlap, or be variants of the same match
            # (e.g. permutations of how table legs match)
            while pattern_match:
                graph_to_return = self._replace_match_with_object_graph_node(
                    graph_to_return, pattern_match, matched_object_nodes, description
                )
                # matcher = pattern.matcher(graph_to_return, matching_objects=True)
                # pattern_match = first(matcher.matches(use_lookahead_pruning=True), None)
                # TODO: we currently match each object type only once!
                # https://github.com/isi-vista/adam/issues/627
                break
        if matched_object_nodes:
            logging.info(
                "Object recognizer recognized: %s",
                [description for (description, _) in matched_object_nodes],
            )
        logging.info("object matching: ms in success: %s, ms in failed: %s",
                     cumulative_millis_in_successful_matches_ms,
                     cumulative_millis_in_failed_matches_ms)
        return PerceptionGraphFromObjectRecognizer(graph_to_return, matched_object_nodes)

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
                # don't want to add edges which are internal to the matched sub-graph
                if matched_subgraph_node_successor not in matched_subgraph_nodes:
                    edge_data = perception_digraph.get_edge_data(
                        matched_subgraph_node, matched_subgraph_node_successor
                    )
                    label = edge_data["label"]
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
                # don't want to add edges which are internal to the matched sub-graph
                if matched_subgraph_node_predecessor not in matched_subgraph_nodes:
                    edge_data = perception_digraph.get_edge_data(
                        matched_subgraph_node_predecessor, matched_subgraph_node
                    )
                    label = edge_data["label"]
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
