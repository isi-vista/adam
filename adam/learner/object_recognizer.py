import logging
from itertools import chain
from typing import Iterable, List, Mapping, Set, Tuple

from attr.validators import deep_iterable, deep_mapping, instance_of
from more_itertools import first
from networkx import DiGraph

from adam.axes import GRAVITATIONAL_DOWN_TO_UP_AXIS, LEARNER_AXES, WORLD_AXES
from adam.language import LinguisticDescription
from adam.ontology import OntologyNode
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
)
from attr import attrib, attrs
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


@attrs(frozen=True)
class ObjectRecognizer:
    """
    The ObjectRecognizer finds object matches in the scene pattern and adds a `MatchedObjectPerceptionPredicate`
    which can be used to learn additional semantics which relate objects to other objects
    """

    object_names_to_patterns: Mapping[str, PerceptionGraphPattern] = attrib(
        validator=deep_mapping(instance_of(str), instance_of(PerceptionGraphPattern))
    )
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
        ontology_types: Iterable[OntologyNode], determiners: Iterable[str]
    ) -> "ObjectRecognizer":
        return ObjectRecognizer(
            object_names_to_patterns=immutabledict(
                (
                    obj_type.handle,
                    PerceptionGraphPattern.from_schema(
                        first(GAILA_PHASE_1_ONTOLOGY.structural_schemata(obj_type))
                    ),
                )
                for obj_type in ontology_types
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
        matched_object_nodes: List[Tuple[Tuple[str, ...], MatchedObjectNode]] = []
        graph_to_return = perception_graph.copy_as_digraph()
        is_dynamic = perception_graph.dynamic
        for (description, pattern) in self.object_names_to_patterns.items():
            matcher = pattern.matcher(
                PerceptionGraph(graph_to_return, is_dynamic), matching_objects=True
            )
            pattern_match = first(matcher.matches(use_lookahead_pruning=True), None)
            # It's important not to simply iterate over pattern matches
            # because they might overlap, or be variants of the same match
            # (e.g. permutations of how table legs match)
            while pattern_match:
                self._replace_match_with_object_graph_node(
                    graph_to_return, pattern_match, matched_object_nodes, description
                )
                matcher = pattern.matcher(
                    PerceptionGraph(graph_to_return, is_dynamic), matching_objects=True
                )
                pattern_match = first(matcher.matches(use_lookahead_pruning=True), None)
                # TODO: we currently match each object type only once!
                # https://github.com/isi-vista/adam/issues/627
                break
        if matched_object_nodes:
            logging.info(
                "Object recognizer recognized: %s",
                [description for (description, _) in matched_object_nodes],
            )
        return PerceptionGraphFromObjectRecognizer(
            PerceptionGraph(graph=graph_to_return, dynamic=perception_graph.dynamic),
            matched_object_nodes,
        )

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
        networkx_graph_to_modify_in_place: DiGraph,
        pattern_match: PerceptionGraphPatternMatch,
        matched_object_nodes: List[Tuple[Tuple[str, ...], MatchedObjectNode]],
        description: str,
    ):
        """
        Internal function to copy existing relationships from the matched object pattern onto a
        `MatchedObjectPerceptionPredicate`
        """
        matched_object_node = MatchedObjectNode(name=(description,))

        # We wrap the description in a tuple because it could in theory be multiple tokens,
        # even though currently it never is.
        matched_object_nodes.append(((description,), matched_object_node))
        networkx_graph_to_modify_in_place.add_node(matched_object_node)

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

            # If there is an edge from the matched sub-graph to a node outside it,
            # also add an edge from the object match node to that node.
            for (
                matched_subgraph_node_successor
            ) in networkx_graph_to_modify_in_place.successors(matched_subgraph_node):
                # don't want to add edges which are internal to the matched sub-graph
                if matched_subgraph_node_successor not in matched_subgraph_nodes:
                    edge_data = networkx_graph_to_modify_in_place.get_edge_data(
                        matched_subgraph_node, matched_subgraph_node_successor
                    )
                    label = edge_data["label"]
                    if (
                        isinstance(label, RelationTypeIsPredicate)
                        and label.dot_label == "rel(" "has-matched-object-pattern)"
                    ):
                        raise RuntimeError(
                            f"Overlapping nodes in object recognition: "
                            f"{matched_subgraph_node}, "
                            f"{matched_subgraph_node_successor}"
                        )
                    networkx_graph_to_modify_in_place.add_edge(
                        matched_object_node, matched_subgraph_node_successor, **edge_data
                    )

            # If there is an edge to the matched sub-graph from a node outside it,
            # also add an edge to the object match node from that node.
            for (
                matched_subgraph_node_predecessor
            ) in networkx_graph_to_modify_in_place.predecessors(matched_subgraph_node):
                # don't want to add edges which are internal to the matched sub-graph
                if matched_subgraph_node_predecessor not in matched_subgraph_nodes:
                    edge_data = networkx_graph_to_modify_in_place.get_edge_data(
                        matched_subgraph_node_predecessor, matched_subgraph_node
                    )
                    label = edge_data["label"]
                    if (
                        isinstance(label, RelationTypeIsPredicate)
                        and label.dot_label == "rel(" "has-matched-object-pattern)"
                    ):
                        raise RuntimeError(
                            f"Overlapping nodes in object recognition: "
                            f"{matched_subgraph_node}, "
                            f"{matched_subgraph_node_predecessor}"
                        )

                    networkx_graph_to_modify_in_place.add_edge(
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
        networkx_graph_to_modify_in_place.remove_nodes_from(
            matched_node
            for matched_node in matched_subgraph_nodes
            if matched_node not in SHARED_WORLD_ITEMS
        )

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
