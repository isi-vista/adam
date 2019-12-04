r"""
Code for representing `DevelopmentalPrimitivePerceptionFrame`\ s as directed graphs
and for matching patterns against such graph.
Such patterns could be used to implement object recognition,
among other things.

This file first defines `PerceptionGraph`\ s,
then defines `PerceptionGraphPattern`\ s to match them.
"""
from abc import ABC, abstractmethod
from copy import copy
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
    Set,
    List,
    Type,
)

import graphviz
from adam.utils.networkx_utils import digraph_with_nodes_sorted_by
from attr.validators import instance_of, optional
from immutablecollections import ImmutableDict, immutabledict, immutableset, ImmutableSet
from immutablecollections.converter_utils import _to_immutabledict, _to_tuple
from more_itertools import first
from networkx import DiGraph
from typing_extensions import Protocol

from adam.axes import AxesInfo, HasAxes
from adam.axis import GeonAxis
from adam.geon import Geon, MaybeHasGeon
from adam.ontology import OntologyNode
from adam.ontology.phase1_ontology import GAILA_PHASE_1_ONTOLOGY, PART_OF
from adam.ontology.phase1_spatial_relations import Direction, Distance, Region
from adam.ontology.structural_schema import ObjectStructuralSchema
from adam.perception import ObjectPerception
from adam.perception._matcher import GraphMatching
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
    HasBinaryProperty,
    HasColor,
    RgbColorPerception,
)
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator,
)
from adam.random_utils import RandomChooser
from adam.situation import SituationObject
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.utils.networkx_utils import subgraph
from attr import attrib, attrs

from time import process_time


class Incrementer:
    def __init__(self, initial_value=0) -> None:
        self._value = initial_value

    def value(self) -> int:
        return self._value

    def increment(self, amount=1) -> None:
        self._value += amount


DebugCallableType = Callable[[DiGraph, Dict[Any, Any]], None]


@attrs(frozen=True, eq=False, slots=True)
class MatchedObjectNode:
    """
    A `MatchedObjectNode` is the PerceptionGraph node to indicate an object which
    have been identified in the graph
    """

    name: Tuple[str] = attrib()


PerceptionGraphNode = Union[
    ObjectPerception,
    OntologyNode,
    Tuple[Region[Any], int],
    Tuple[Geon, int],
    GeonAxis,
    MatchedObjectNode,
]
PerceptionGraphEdgeLabel = Union[OntologyNode, str, Direction[Any]]

# certain constant edges used by PerceptionGraphs
REFERENCE_OBJECT_LABEL = OntologyNode("reference-object")
"""
Edge label in a `PerceptionGraph` linking a `Region` to its reference object.
"""
PRIMARY_AXIS_LABEL = OntologyNode("primary-axis")
"""
Edge label in a `PerceptionGraph` linking a `Geon` to its primary `GeonAxis`.
"""

HAS_AXIS_LABEL = OntologyNode("has-axis")
"""
Edge label in a `PerceptionGraph` linking any node of type `HasAxes` 
to its associated axes.
"""
GENERATING_AXIS_LABEL = OntologyNode("generating-axis")
"""
Edge label in a `PerceptionGraph` linking a `Geon` to its generating `GeonAxis`.
"""
HAS_GEON_LABEL = OntologyNode("geon")
"""
Edge label in a `PerceptionGraph` linking an `ObjectPerception` to its associated `Geon`.
"""
HAS_PROPERTY_LABEL = OntologyNode("has-property")
"""
Edge label in a `PerceptionGraph` linking an `ObjectPerception` to its associated `Property`.
"""
GOVERNED = OntologyNode("governed")
"""
An object match governed in a preposition relationship
"""
MODIFIED = OntologyNode("modified")
"""
An object match modified in a preposition relationship
"""


class PerceptionGraphProtocol(Protocol):
    _graph: DiGraph = attrib(validator=instance_of(DiGraph))

    def copy_as_digraph(self) -> DiGraph:
        return copy(self._graph)

    def render_to_file(
        self,
        graph_name: str,
        output_file: Path,
        *,
        match_correspondence_ids: Mapping[Any, str] = immutabledict(),
    ) -> None:
        """
        Debugging tool to render the graph to PDF using *dot*.
        """


@attrs(frozen=True)
class PerceptionGraph(PerceptionGraphProtocol):
    r"""
    Represents a `DevelopmentalPrimitivePerceptionFrame` as a directed graph.

    `ObjectPerception`\ s, properties, `Geon`\ s, `GeonAxis`\ s, and `Region`\ s are nodes.

    These can be matched against by `PerceptionGraphPattern`\ s.
    """
    _graph: DiGraph = attrib(validator=instance_of(DiGraph))

    def copy_as_digraph(self) -> DiGraph:
        return copy(self._graph)

    @staticmethod
    def from_frame(frame: DevelopmentalPrimitivePerceptionFrame) -> "PerceptionGraph":
        """
        Gets the `PerceptionGraph` corresponding to a `DevelopmentalPrimitivePerceptionFrame`.
        """
        graph = DiGraph()

        def map_node(obj: Any):
            # Regions and Geons are normally treated as value objects,
            # but we want to maintain their distinctness in the perceptual graph
            # for the purpose of matching patterns, so we make their corresponding
            # graph nods compare by identity.
            if isinstance(obj, (Region, Geon)):
                return (obj, id(obj))
            else:
                return obj

        def map_edge(label: Any):
            return {"label": label}

        for perceived_object in frame.perceived_objects:
            # Every perceived object is a node in the graph.
            graph.add_node(perceived_object)
            # And so are each of its axes.
            _translate_axes(
                graph, perceived_object, perceived_object, map_node, map_edge=map_edge
            )
            _translate_geon(
                graph,
                perceived_object,
                mapped_owner=perceived_object,
                map_geon=map_node,
                map_axis=map_node,
                map_edge=map_edge,
            )

        regions = immutableset(
            relation.second_slot
            for relation in frame.relations
            if isinstance(relation.second_slot, Region)
        )

        for region in regions:
            _translate_region(
                graph,
                region,
                map_node=map_node,
                map_edge=map_edge,
                axes_info=frame.axis_info,
            )

        # Every relation is handled as a directed graph edge
        # from the first argument to the second
        for relation in frame.relations:
            graph.add_edge(
                map_node(relation.first_slot),
                map_node(relation.second_slot),
                label=relation.relation_type,
            )

        dest_node: Any
        for property_ in frame.property_assertions:
            source_node = map_node(property_.perceived_object)
            if isinstance(property_, HasBinaryProperty):
                dest_node = map_node(property_.binary_property)
            elif isinstance(property_, HasColor):
                dest_node = map_node(property_.color)
            else:
                raise RuntimeError(f"Don't know how to translate property {property_}")
            graph.add_edge(source_node, dest_node, label=HAS_PROPERTY_LABEL)

        return PerceptionGraph(graph)

    def render_to_file(  # pragma: no cover
        self,
        graph_name: str,
        output_file: Path,
        *,
        match_correspondence_ids: Mapping[Any, str] = immutabledict(),
    ) -> None:
        """
        Debugging tool to render the graph to PDF using *dot*.

        If this graph has been matched against a pattern,
        the matched nodes can be highlighted and given labels which show
        what the correspond to in a pattern by supplying
        *match_correspondence_ids* which maps graph nodes to
        the desired correspondence labels.
        """
        dot_graph = graphviz.Digraph(graph_name)
        dot_graph.attr(rankdir="LR")
        # combine parallel edges to cut down on clutter
        dot_graph.attr(concentrate="true")

        next_node_id = Incrementer()

        # add all nodes to the graph
        perception_nodes_to_dot_node_ids = {
            perception_node: self._to_dot_node(
                dot_graph, perception_node, next_node_id, match_correspondence_ids
            )
            for perception_node in self._graph.nodes
        }

        for (source_node, target_node, label) in self._graph.edges.data("label"):
            if isinstance(label, Direction):
                sign_string = "+" if label.positive else "-"
                edge_label = f"{sign_string}relative-to"
            else:
                edge_label = str(label)

            source_dot_node = perception_nodes_to_dot_node_ids[source_node]
            target_dot_node = perception_nodes_to_dot_node_ids[target_node]

            # we want bigger things on the left, smaller things on the right,
            # but graphviz wants to put edge sources on the left,
            # so we need to reverse these edges from GraphViz's point-of-view.
            reverse_rank_order = label == PART_OF or edge_label == "reference-object"

            # in-region relationships can run anywhich way
            # without respect to a good layout hierarchy
            constraint = edge_label != "in-region"
            constraint_string = "true" if constraint else "false"

            if reverse_rank_order:
                dot_graph.edge(
                    target_dot_node,
                    source_dot_node,
                    edge_label,
                    dir="back",
                    constraint=constraint_string,
                )
            else:
                dot_graph.edge(
                    source_dot_node,
                    target_dot_node,
                    edge_label,
                    constraint=constraint_string,
                )

        dot_graph.render(str(output_file))

    def _to_dot_node(
        self,
        dot_graph: graphviz.Digraph,
        perception_node: PerceptionGraphNode,
        next_node_id: Incrementer,
        match_correspondence_ids: Mapping[Any, str],
    ) -> str:
        if isinstance(perception_node, tuple):
            perception_node = perception_node[0]

        # object perceptions have no content, so they are blank nodes
        if isinstance(perception_node, ObjectPerception):
            label = perception_node.debug_handle
        # regions do have content but we express those as edges to other nodes
        elif isinstance(perception_node, Region):
            if perception_node.distance:
                dist_string = f"[{perception_node.distance.name}]"
            else:
                dist_string = ""
            label = f"reg:{dist_string}"
        elif isinstance(perception_node, GeonAxis):
            label = f"axis:{perception_node.debug_name}"
        elif isinstance(perception_node, RgbColorPerception):
            label = perception_node.hex
        elif isinstance(perception_node, OntologyNode):
            label = perception_node.handle
        elif isinstance(perception_node, Geon):
            label = str(perception_node.cross_section) + str(
                perception_node.cross_section_size
            )
        elif isinstance(perception_node, MatchedObjectNode):
            label = " ".join(perception_node.name)
        else:
            raise RuntimeError(
                f"Do not know how to perception node render node "
                f"{perception_node} with dot"
            )

        # if we are rendering a pattern which partially matched against a graph,
        # the user can supply IDs for those nodes which matched to show the correspondence.
        # We also make these nodes bold.
        mapping_id = match_correspondence_ids.get(perception_node)
        if mapping_id is not None:
            attributes = {
                "label": f"{label} [{mapping_id}]",
                "style": "filled",
                "fillcolor": "gray",
            }
        else:
            attributes = {"label": label, "style": "solid"}

        node_id = f"node-{next_node_id.value()}"
        next_node_id.increment()

        dot_graph.node(node_id, **attributes)

        return node_id


@attrs(frozen=True, slots=True)
class PerceptionGraphPattern(PerceptionGraphProtocol):
    r"""
    A pattern which can match `PerceptionGraph`\ s.

    Such patterns could be used, for example, to represent a learner's
    knowledge of an object for object recognition.
    """

    _graph: DiGraph = attrib(validator=instance_of(DiGraph))

    def copy_as_digraph(self) -> DiGraph:
        return copy(self._graph)

    def matcher(
        self,
        graph_to_match_against: PerceptionGraphProtocol,
        *,
        debug_callback: Optional[DebugCallableType] = None,
    ) -> "PatternMatching":
        """
        Creates an object representing an attempt to match this pattern
        against *graph_to_match_against*.
        """
        return PatternMatching(
            pattern=self,
            graph_to_match_against=graph_to_match_against,
            debug_callback=debug_callback,
        )

    @staticmethod
    def from_schema(object_schema: ObjectStructuralSchema) -> "PerceptionGraphPattern":
        """
        Creates a pattern for recognizing an object based on its *object_schema*.
        """

        # First, we generate a PerceptionGraph corresponding to this schema
        # by instantiating a situation which contains a single object
        # of the desired type.
        schema_situation_object = SituationObject.instantiate_ontology_node(
            ontology_node=object_schema.ontology_node, ontology=GAILA_PHASE_1_ONTOLOGY
        )
        situation = HighLevelSemanticsSituation(
            ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[schema_situation_object]
        )
        perception_generator = HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator(
            GAILA_PHASE_1_ONTOLOGY
        )
        # We explicitly exclude ground and learner in perception generation, which were not specified in the schema
        perception = perception_generator.generate_perception(
            situation,
            chooser=RandomChooser.for_seed(0),
            include_ground=False,
            include_learner=False,
        )
        perception_graph = PerceptionGraph.from_frame(
            first(perception.frames)
        ).copy_as_digraph()

        # We remove color and properties that is added by situation generation, but is not in schema
        nodes_to_remove = [
            node
            for node in perception_graph
            if isinstance(node, RgbColorPerception) or isinstance(node, OntologyNode)
        ]
        perception_graph.remove_nodes_from(nodes_to_remove)

        # Finally, we convert the PerceptionGraph DiGraph representation to a PerceptionGraphPattern
        return PerceptionGraphPattern.from_graph(
            perception_graph=perception_graph
        ).perception_graph_pattern

    @staticmethod
    def from_graph(perception_graph: DiGraph) -> "PerceptionGraphPatternFromGraph":
        """
        Creates a pattern for recognizing an object based on its *perception_graph*.
        """
        pattern_graph = DiGraph()
        perception_node_to_pattern_node: Dict[PerceptionGraphNode, NodePredicate] = {}
        PerceptionGraphPattern._translate_graph(
            perception_graph=perception_graph,
            pattern_graph=pattern_graph,
            perception_node_to_pattern_node=perception_node_to_pattern_node,
        )
        return PerceptionGraphPatternFromGraph(
            perception_graph_pattern=PerceptionGraphPattern(pattern_graph),
            perception_graph_node_to_pattern_node=perception_node_to_pattern_node,
        )

    @staticmethod
    def _translate_graph(
        perception_graph: DiGraph,
        pattern_graph: DiGraph,
        *,
        perception_node_to_pattern_node: Dict[Any, "NodePredicate"],
    ) -> None:
        # Two mapping methods that map nodes and edges from the source PerceptionGraph onto the corresponding
        # node and edge representations on the PerceptionGraphPattern.
        def map_node(node: Any) -> "NodePredicate":

            key = node
            if key not in perception_node_to_pattern_node:
                if isinstance(node, tuple):
                    node = node[0]
                    if isinstance(node, Geon):
                        perception_node_to_pattern_node[
                            key
                        ] = GeonPredicate.exactly_matching(node)
                    elif isinstance(node, Region):
                        perception_node_to_pattern_node[
                            key
                        ] = RegionPredicate.matching_distance(node)
                    else:
                        raise RuntimeError(f"Don't know how to map tuple node {node}")
                elif isinstance(node, GeonAxis):
                    perception_node_to_pattern_node[key] = AxisPredicate.from_axis(node)
                elif isinstance(node, ObjectPerception):
                    perception_node_to_pattern_node[key] = AnyObjectPerception(
                        debug_handle=node.debug_handle
                    )
                elif isinstance(node, OntologyNode):
                    perception_node_to_pattern_node[key] = IsOntologyNodePredicate(node)
                elif isinstance(node, RgbColorPerception):
                    perception_node_to_pattern_node[key] = IsColorNodePredicate(node)
                elif isinstance(node, MatchedObjectNode):
                    perception_node_to_pattern_node[
                        key
                    ] = MatchedObjectPerceptionPredicate()
                else:
                    raise RuntimeError(f"Don't know how to map node {node}")
            return perception_node_to_pattern_node[key]

        def map_edge(label: Any) -> Mapping[str, "EdgePredicate"]:
            if isinstance(label, OntologyNode):
                return {"predicate": RelationTypeIsPredicate(label)}
            elif isinstance(label, Direction):
                return {"predicate": DirectionPredicate.exactly_matching(label)}
            else:
                raise RuntimeError(f"Cannot map edge {label}")

        # We add every node in the source graph, after translating their type with map_node
        for original_node in perception_graph.nodes:
            # Add each node
            pattern_node = map_node(original_node)
            pattern_graph.add_node(pattern_node)

        # Once all nodes are translated, we add all edges from the source graph by iterating over each node and
        # extracting its edges.
        for original_node in perception_graph.nodes:
            edges_from_node = perception_graph.edges(original_node, data=True)
            for (_, original_dest_node, original_edge_data) in edges_from_node:
                pattern_from_node = perception_node_to_pattern_node[original_node]
                pattern_to_node = perception_node_to_pattern_node[original_dest_node]
                pattern_graph.add_edge(pattern_from_node, pattern_to_node)
                mapped_edge = map_edge(original_edge_data["label"])
                pattern_graph.edges[pattern_from_node, pattern_to_node].update(
                    mapped_edge
                )

    def render_to_file(  # pragma: no cover
        self,
        graph_name: str,
        output_file: Path,
        *,
        match_correspondence_ids: Mapping[Any, str] = immutabledict(),
    ) -> None:
        """
        Debugging tool to render the pattern to PDF using *dot*.

        If this pattern has been matched against a `PerceptionGraph`,
        the matched nodes can be highlighted and given labels which show
        what the correspond to in a pattern by supplying
        *match_correspondence_ids* which maps graph nodes to
        the desired correspondence labels.
        """

        dot_graph = graphviz.Digraph(graph_name)
        dot_graph.attr(rankdir="LR")

        next_node_id = Incrementer()

        def to_dot_node(pattern_node: "NodePredicate") -> str:
            node_id = f"node-{next_node_id.value()}"
            next_node_id.increment()
            base_label = pattern_node.dot_label()

            # if we are rendering a match against another graph,
            # we show IDs that align the nodes between the graphs
            # and make the nodes which have matches bold.
            correspondence_id = match_correspondence_ids.get(pattern_node)
            if correspondence_id is not None:
                attributes = {
                    "label": f"{base_label} [{correspondence_id}]",
                    "style": "filled",
                    "fillcolor": "gray",
                }
            else:
                attributes = {"label": base_label, "style": "solid"}

            dot_graph.node(node_id, **attributes)
            return node_id

        pattern_nodes_to_dot_node_ids = {
            pattern_node: to_dot_node(pattern_node) for pattern_node in self._graph.nodes
        }

        for (source_node, target_node, predicate) in self._graph.edges.data("predicate"):
            source_dot_node = pattern_nodes_to_dot_node_ids[source_node]
            target_dot_node = pattern_nodes_to_dot_node_ids[target_node]

            if not predicate.reverse_in_dot_graph():
                dot_graph.edge(source_dot_node, target_dot_node, predicate.dot_label())
            else:
                dot_graph.edge(
                    target_dot_node, source_dot_node, predicate.dot_label(), back="true"
                )

        dot_graph.render(output_file)

    def intersection(
        self,
        graph_pattern: "PerceptionGraphPattern",
        *,
        debug_callback: Optional[DebugCallableType] = None,
    ) -> "PerceptionGraphPattern":
        debug_mapping: Dict[Any, Any] = dict()
        matcher = PatternMatching(
            pattern=graph_pattern,
            graph_to_match_against=self,
            debug_callback=debug_callback,
        )
        matches = immutableset(
            matcher.matches(debug_mapping_sink=debug_mapping, matching_pattern=True)
        )
        if not matches.empty():
            # We found a match! This means our two patterns are the same and we should just keep the original
            return self
        else:
            # No match found so we use the biggest match provided by the debug sink. This provides an 'intersection'
            # between the two patterns
            return PerceptionGraphPattern(
                graph=subgraph(self._graph, nodes=debug_mapping)
            )


class DumpPartialMatchCallback:
    """
        Helper callable class for debugging purposes. An instance of this object can be provided as the `debug_callback` argument of `GraphMatching.match` to render
        the match search process at every 100 time steps. We start rendering after the first 60 seconds.
    """

    def __init__(self, render_path) -> None:
        self.render_path = render_path
        self.calls_to_match_counter = 0
        self.start_time = process_time()
        self.seconds_to_wait_before_rendering = 60
        self.mod = 100

    def __call__(
        self, graph: DiGraph, graph_node_to_pattern_node: Dict[Any, Any]
    ) -> None:
        self.calls_to_match_counter += 1
        current_time = process_time()
        if (
            self.calls_to_match_counter % self.mod == 0
            and (current_time - self.start_time) > self.seconds_to_wait_before_rendering
        ):
            perception_graph = PerceptionGraph(graph)
            title = (
                "id_"
                + str(id(self))
                + "_graph_"
                + str(id(graph))
                + "_call_"
                + str(self.calls_to_match_counter).zfill(4)
            )
            mapping = {k: "match" for k, v in graph_node_to_pattern_node.items()}
            perception_graph.render_to_file(
                graph_name=title,
                output_file=Path(self.render_path + title),
                match_correspondence_ids=mapping,
            )


@attrs(frozen=True, slots=True, auto_attribs=True)
class PerceptionGraphPatternFromGraph:
    """
    See `PerceptionGraphPattern.from_graph`
    """

    perception_graph_pattern: PerceptionGraphPattern
    perception_graph_node_to_pattern_node: ImmutableDict[
        PerceptionGraphNode, "NodePredicate"
    ] = attrib(converter=_to_immutabledict)


@attrs(frozen=True, slots=True, eq=False)
class PatternMatching:
    """
    An attempt to align a `PerceptionGraphPattern` to nodes in a `PerceptionGraph`.

    This is equivalent to finding a sub-graph of *graph_to_match*
    which is isomorphic to *pattern*.
    Currently we only handle node-induced sub-graph isomorphisms,
    but we might really want edge-induced: https://github.com/isi-vista/adam/issues/400
    """

    pattern: PerceptionGraphPattern = attrib(
        validator=instance_of(PerceptionGraphPattern)
    )
    graph_to_match_against: PerceptionGraphProtocol = attrib(
        validator=instance_of(PerceptionGraphProtocol)
    )

    # Callable object for debugging purposes. We use this to track the number of calls to match and render the graphs.
    debug_callback: Optional[DebugCallableType] = attrib(default=None, kw_only=True)

    def matches(
        self,
        *,
        debug_mapping_sink: Optional[Dict[Any, Any]] = None,
        use_lookahead_pruning: bool = False,
        matching_pattern: bool = False,
        suppress_multiple_alignments_to_same_nodes: bool = True,
    ) -> Iterable["PerceptionGraphPatternMatch"]:
        """
        Attempt the matching and returns a generator over the set of possible matches.

        Currently matching with look-ahead pruning seems to give false negatives,
        so we recommend disabling it and have set the default to *False*:
        https://github.com/isi-vista/adam/issues/401
        When we fix the bug, we may change the default back to *True*.

        If *debug_mapping_sink* is provided, the best partial matching found
        will be written to it in case of a failed match.

        If *suppress_multiple_alignments_to_same_nodes* is *True* (default *True*),
        then only the first alignment encountered for a given set of nodes will be returned.
        This prevents you from e.g. getting multiple matches for different ways
        of aligning axes for symmetric objects.
        The cost is that we need to keep around a memory of previous node matches.
        """

        # Controlling the iteration order of the graphs
        # controls the order in which nodes are matched.
        # This has a significant, benchmark-confirmed impact on performance.
        sorted_graph_to_match_against = digraph_with_nodes_sorted_by(
            self.graph_to_match_against._graph,  # pylint: disable=W0212
            _graph_node_order if not matching_pattern else _pattern_matching_node_order,
        )
        sorted_pattern = digraph_with_nodes_sorted_by(
            self.pattern._graph, _pattern_matching_node_order  # pylint: disable=W0212
        )

        matching = GraphMatching(
            sorted_graph_to_match_against,
            sorted_pattern,
            use_lookahead_pruning=use_lookahead_pruning,
        )

        sets_of_nodes_matched: Set[ImmutableSet[PerceptionGraphNode]] = set()

        got_a_match = False
        for graph_node_to_matching_pattern_node in matching.subgraph_isomorphisms_iter(
            debug=(debug_mapping_sink is not None),
            matching_pattern=matching_pattern,
            debug_callback=self.debug_callback,
        ):
            got_a_match = True
            matched_graph_nodes: ImmutableSet[PerceptionGraphNode] = immutableset(
                graph_node_to_matching_pattern_node
            )
            if (
                matched_graph_nodes not in sets_of_nodes_matched
                or not suppress_multiple_alignments_to_same_nodes
            ):
                yield PerceptionGraphPatternMatch(
                    graph_matched_against=self.graph_to_match_against,
                    matched_pattern=self.pattern,
                    matched_sub_graph=PerceptionGraph(
                        subgraph(
                            matching.graph, graph_node_to_matching_pattern_node.keys()
                        ).copy()
                    ),
                    pattern_node_to_matched_graph_node=_invert_to_immutabledict(
                        graph_node_to_matching_pattern_node
                    ),
                )
            sets_of_nodes_matched.add(matched_graph_nodes)
        if debug_mapping_sink is not None and not got_a_match:
            # we failed to match the pattern.
            # If the user requested it, we provide the largest matching we could find
            # for debugging purposes.
            debug_mapping_sink.clear()
            debug_mapping_sink.update(matching.debug_largest_match)

    def debug_matching(
        self,
        *,
        use_lookahead_pruning: bool = True,
        render_match_to: Optional[Path] = None,
    ) -> GraphMatching:
        """
        Similar to `matches`, but returns the internal `GraphMatching` object
        at the end of the matching process for analysis.

        It also writes PDFs showing the best partial match to *render_match_to*,
        if specified.
        """
        matching = GraphMatching(
            self.graph_to_match_against._graph,  # pylint:disable=protected-access
            self.pattern._graph,  # pylint:disable=protected-access
            use_lookahead_pruning=use_lookahead_pruning,
        )
        for _ in matching.subgraph_isomorphisms_iter(debug=True):
            pass
        if render_match_to:
            pattern_node_to_correspondence_index = {}
            graph_node_to_correspondence_index = {}
            for (idx, (pattern_node, graph_node)) in enumerate(
                matching.debug_largest_match.items()
            ):
                pattern_node_to_correspondence_index[pattern_node] = str(idx)
                graph_node_to_correspondence_index[graph_node] = str(idx)

            self.pattern.render_to_file(
                "pattern",
                render_match_to / "pattern",
                match_correspondence_ids=pattern_node_to_correspondence_index,
            )
            self.graph_to_match_against.render_to_file(
                "graph",
                render_match_to / "graph",
                match_correspondence_ids=graph_node_to_correspondence_index,
            )

        return matching


@attrs(frozen=True, slots=True, eq=False)
class PerceptionGraphPatternMatch:
    """
    Represents a match of a `PerceptionPatternGraph` against a `PerceptionGraph`.
    """

    matched_pattern: PerceptionGraphPattern = attrib(
        validator=instance_of(PerceptionGraphPattern), kw_only=True
    )
    graph_matched_against: PerceptionGraphProtocol = attrib(
        validator=instance_of(PerceptionGraphProtocol), kw_only=True
    )
    matched_sub_graph: PerceptionGraph = attrib(
        validator=instance_of(PerceptionGraph), kw_only=True
    )
    pattern_node_to_matched_graph_node: Mapping[
        "NodePredicate", PerceptionGraphNode
    ] = attrib(converter=_to_immutabledict, kw_only=True)
    """
    A mapping of pattern nodes from `matched_pattern` to the nodes
    in `matched_sub_graph` they were aligned to.
    """


# Below are various types of noes and edges which can appear in a pattern graph.
# The nodes and edges of perception graphs are just ordinary ADAM objects,
# so nothing special is needed for them.


class NodePredicate(ABC):
    r"""
    Super-class for pattern graph nodes.

    All `NodePredicate`\ s should compare non-equal to one another
    (if the are *attrs* classes, set *eq=False*).
    """

    @abstractmethod
    def __call__(self, graph_node: PerceptionGraphNode) -> bool:
        """
        Determines whether a *graph_node* is matched by this predicate.
        """

    @abstractmethod
    def dot_label(self) -> str:
        """
        Node label to use when rendering patterns as graphs using *dot*.
        """

    @abstractmethod
    def matches_predicate(self, predicate_node: "NodePredicate") -> bool:
        """
        Determines whether a NodePredicate matches another Node Predicate
        """


@attrs(frozen=True, slots=True, eq=False)
class AnyNodePredicate(NodePredicate):
    """
    Matches any node whatsoever.
    """

    def __call__(self, graph_node: PerceptionGraphNode) -> bool:
        return True

    def dot_label(self) -> str:
        return "*"

    def matches_predicate(self, predicate_node: "NodePredicate") -> bool:
        return True


@attrs(frozen=True, slots=True, eq=False)
class AnyObjectPerception(NodePredicate):
    """
    Matches any `ObjectPerception` node.
    """

    debug_handle: Optional[str] = attrib(validator=optional(instance_of(str)))

    def __call__(self, graph_node: PerceptionGraphNode) -> bool:
        return isinstance(graph_node, ObjectPerception)

    def dot_label(self) -> str:
        if self.debug_handle is not None:
            debug_handle_str = f"[{self.debug_handle}]"
        else:
            debug_handle_str = ""
        return f"*obj{debug_handle_str}"

    def matches_predicate(self, predicate_node: "NodePredicate") -> bool:
        return isinstance(predicate_node, AnyObjectPerception)


@attrs(frozen=True, slots=True, eq=False)
class AxisPredicate(NodePredicate):
    """
    Represents constraints on an axis given in a `PerceptionGraphPattern`
    """

    curved: Optional[bool] = attrib(validator=optional(instance_of(bool)))
    directed: Optional[bool] = attrib(validator=optional(instance_of(bool)))
    aligned_to_gravitational: Optional[bool] = attrib(
        validator=optional(instance_of(bool))
    )

    def __call__(self, graph_node: PerceptionGraphNode) -> bool:
        # axes might be wrapped in tuples with their id()
        # in order to simulate comparison by object ID.
        if isinstance(graph_node, tuple):
            graph_node = graph_node[0]

        if isinstance(graph_node, GeonAxis):
            if self.curved is not None and self.curved != graph_node.curved:
                return False
            if self.directed is not None and self.directed != graph_node.directed:
                return False
            if (
                self.aligned_to_gravitational is not None
                and self.aligned_to_gravitational != graph_node.aligned_to_gravitational
            ):
                return False
            return True
        else:
            return False

    @staticmethod
    def from_axis(axis_to_match: GeonAxis) -> "AxisPredicate":
        return AxisPredicate(
            curved=axis_to_match.curved,
            directed=axis_to_match.directed,
            aligned_to_gravitational=axis_to_match.aligned_to_gravitational,
        )

    def dot_label(self) -> str:
        constraints = []
        if self.curved is not None:
            constraints.append(f"curved={self.curved}")
        if self.directed is not None:
            constraints.append(f"directed={self.directed}")
        if self.aligned_to_gravitational is not None:
            constraints.append(f"aligned_to_grav={self.aligned_to_gravitational}")

        return f"axis({', '.join(constraints)})"

    def matches_predicate(self, predicate_node: "NodePredicate") -> bool:
        if isinstance(predicate_node, AxisPredicate):
            if self.curved is not None and self.curved != predicate_node.curved:
                return False
            if self.directed is not None and self.directed != predicate_node.directed:
                return False
            if (
                self.aligned_to_gravitational is not None
                and self.aligned_to_gravitational
                != predicate_node.aligned_to_gravitational
            ):
                return False
            else:
                return True
        else:
            return False


@attrs(frozen=True, slots=True, eq=False)
class GeonPredicate(NodePredicate):
    """
    Represents constraints on a `Geon` given in a `PerceptionGraphPattern`
    """

    template_geon: Geon = attrib(validator=instance_of(Geon))

    def __call__(self, graph_node: PerceptionGraphNode) -> bool:
        # geons might be wrapped in tuples with their id()
        # in order to simulate comparison by object ID.
        if isinstance(graph_node, tuple):
            graph_node = graph_node[0]

        if isinstance(graph_node, Geon):
            return (
                self.template_geon.cross_section == graph_node.cross_section
                and self.template_geon.cross_section_size == graph_node.cross_section_size
            )
        else:
            return False

    def dot_label(self) -> str:
        return f"geon({self.template_geon})"

    @staticmethod
    def exactly_matching(geon: Geon) -> "GeonPredicate":
        return GeonPredicate(geon)

    def matches_predicate(self, predicate_node: "NodePredicate") -> bool:
        if isinstance(predicate_node, GeonPredicate):
            return (
                self.template_geon.cross_section
                == predicate_node.template_geon.cross_section
                and self.template_geon.cross_section_size
                == predicate_node.template_geon.cross_section_size
            )
        else:
            return False


@attrs(frozen=True, slots=True, eq=False)
class RegionPredicate(NodePredicate):
    """
    Represents constraints on a `Region` given in a `PerceptionGraphPattern`.
    """

    distance: Optional[Distance] = attrib(validator=optional(instance_of(Distance)))

    def __call__(self, graph_node: PerceptionGraphNode) -> bool:
        # regions might be wrapped in tuples with their id()
        # in order to simulate comparison by object ID.
        if isinstance(graph_node, tuple):
            graph_node = graph_node[0]
        return isinstance(graph_node, Region) and self.distance == graph_node.distance

    def dot_label(self) -> str:
        return f"dist({self.distance})"

    @staticmethod
    def matching_distance(region: Region[Any]) -> "RegionPredicate":
        return RegionPredicate(region.distance)

    def matches_predicate(self, predicate_node: "NodePredicate") -> bool:
        if isinstance(predicate_node, RegionPredicate):
            return predicate_node.distance == self.distance
        else:
            return False


@attrs(frozen=True, slots=True, eq=False)
class IsOntologyNodePredicate(NodePredicate):
    property_value: OntologyNode = attrib(validator=instance_of(OntologyNode))

    def __call__(self, graph_node: PerceptionGraphNode) -> bool:
        return graph_node == self.property_value

    def dot_label(self) -> str:
        return f"prop({self.property_value.handle})"

    def matches_predicate(self, predicate_node: "NodePredicate") -> bool:
        if isinstance(predicate_node, IsOntologyNodePredicate):
            return predicate_node.property_value == self.property_value
        else:
            return False


@attrs(frozen=True, slots=True, eq=False)
class IsColorNodePredicate(NodePredicate):
    color: RgbColorPerception = attrib(validator=instance_of(RgbColorPerception))

    def __call__(self, graph_node: PerceptionGraphNode) -> bool:
        if isinstance(graph_node, RgbColorPerception):
            return (
                (graph_node.red == self.color.red)
                and (graph_node.blue == self.color.blue)
                and (graph_node.green == self.color.green)
            )
        return False

    def matches_predicate(self, predicate_node: "NodePredicate") -> bool:
        if isinstance(predicate_node, IsColorNodePredicate):
            return (
                (predicate_node.color.red == self.color.red)
                and (predicate_node.color.blue == self.color.blue)
                and (predicate_node.color.green == self.color.green)
            )
        else:
            return False

    def dot_label(self) -> str:
        return f"prop({self.color.hex})"


@attrs(frozen=True, slots=True, eq=False)
class AndNodePredicate(NodePredicate):
    """
    `NodePredicate` which matches if all its *sub_predicates* match.
    """

    sub_predicates: Tuple[NodePredicate, ...] = attrib(converter=_to_tuple)

    def __call__(self, graph_node: PerceptionGraphNode) -> bool:
        return all(sub_predicate(graph_node) for sub_predicate in self.sub_predicates)

    def dot_label(self) -> str:
        return " & ".join(sub_pred.dot_label() for sub_pred in self.sub_predicates)

    def matches_predicate(self, predicate_node: "NodePredicate") -> bool:
        return all(
            sub_predicate.matches_predicate(predicate_node)
            for sub_predicate in self.sub_predicates
        )


@attrs(frozen=True, slots=True, eq=False)
class MatchedObjectPerceptionPredicate(NodePredicate):
    """
    `NodePredicate` which matches if the node is of this type
    """

    def __call__(self, graph_node: PerceptionGraphNode) -> bool:
        return isinstance(graph_node, MatchedObjectNode)

    def dot_label(self) -> str:
        return "matched-object-perception"

    def matches_predicate(self, predicate_node: "NodePredicate") -> bool:
        return isinstance(predicate_node, MatchedObjectPerceptionPredicate)


class EdgePredicate(ABC):
    r"""
    Super-class for pattern graph edges.
    """

    @abstractmethod
    def __call__(
        self,
        source_object_perception: PerceptionGraphNode,
        edge_label: PerceptionGraphEdgeLabel,
        dest_object_percption: PerceptionGraphNode,
    ) -> bool:
        """
        Returns whether this predicate matches the edge
        from *source_object* to *dest_object* with label *edge_label*.
        """

    @abstractmethod
    def dot_label(self) -> str:
        """
        Edge label to use when rendering patterns as graphs using *dot*.
        """

    def reverse_in_dot_graph(self) -> bool:
        """
        In the dot graph, should this edge be treated as reversed for layout purposes?
        (dot tries to put the sources of edges to the left of the destinations)
        """
        return False

    @abstractmethod
    def matches_predicate(
        self,
        source_predicate_node: NodePredicate,
        edge_label: "EdgePredicate",
        dest_predicate_node: NodePredicate,
    ) -> bool:
        """
        Returns wheether this predicate matches the edge from *source_predicate_node* to *dest_predicate_node* with label *edge_label*
        """


@attrs(frozen=True, slots=True)
class AnyEdgePredicate(EdgePredicate):
    """
    `EdgePredicate` which matches any edge.
    """

    def __call__(
        self,
        source_object_perception: PerceptionGraphNode,
        edge_label: PerceptionGraphEdgeLabel,
        dest_object_percption: PerceptionGraphNode,
    ) -> bool:
        return True

    def dot_label(self) -> str:
        return "*"

    def matches_predicate(
        self,
        source_predicate_node: NodePredicate,
        edge_label: "EdgePredicate",
        dest_predicate_node: NodePredicate,
    ) -> bool:
        return True


@attrs(frozen=True, slots=True)
class RelationTypeIsPredicate(EdgePredicate):
    """
    `EdgePredicate` which matches a relation of the given type.
    """

    relation_type: OntologyNode = attrib(validator=instance_of(OntologyNode))

    def __call__(
        self,
        source_object_perception: PerceptionGraphNode,
        edge_label: PerceptionGraphEdgeLabel,
        dest_object_percption: PerceptionGraphNode,
    ) -> bool:
        return edge_label == self.relation_type

    def dot_label(self) -> str:
        return f"rel({self.relation_type})"

    def reverse_in_dot_graph(self) -> bool:
        return self.relation_type == PART_OF

    def matches_predicate(
        self,
        source_predicate_node: NodePredicate,
        edge_label: "EdgePredicate",
        dest_predicate_node: NodePredicate,
    ) -> bool:
        return edge_label == self.relation_type


@attrs(frozen=True, slots=True)
class DirectionPredicate(EdgePredicate):
    """
    `EdgePredicate` which matches a `Direction` object
    annotating an edge between a `Region` and an `ObjectPerception`.
    """

    reference_direction: Direction[Any] = attrib(validator=instance_of(Direction))

    def __call__(
        self,
        source_object_perception: PerceptionGraphNode,
        edge_label: PerceptionGraphEdgeLabel,
        dest_object_percption: PerceptionGraphNode,
    ) -> bool:
        return (
            isinstance(edge_label, Direction)
            and edge_label.positive == self.reference_direction.positive
        )

    def dot_label(self) -> str:
        return f"dir(positive={self.reference_direction.positive})"

    @staticmethod
    def exactly_matching(direction: Direction[Any]) -> "DirectionPredicate":
        return DirectionPredicate(direction)

    def matches_predicate(
        self,
        source_predicate_node: NodePredicate,
        edge_label: "EdgePredicate",
        dest_predicate_node: NodePredicate,
    ) -> bool:
        if isinstance(edge_label, DirectionPredicate):
            return (
                edge_label.reference_direction.positive
                == self.reference_direction.positive
            )
        else:
            return False


# Graph translation code shared between perception graph construction
# and pattern construction

_AxisMapper = Callable[[GeonAxis], Any]
_EdgeMapper = Callable[[Any], Mapping[str, Any]]


def _add_labelled_edge(
    graph: DiGraph,
    source: Any,
    target: Any,
    unmapped_label: Any,
    *,
    map_edge: _EdgeMapper,
):
    graph.add_edge(source, target)
    mapped_edge = map_edge(unmapped_label)
    graph.edges[source, target].update(mapped_edge)


def _translate_axes(
    graph: DiGraph,
    owner: HasAxes,
    mapped_owner: Any,
    map_axis: _AxisMapper,
    map_edge: _EdgeMapper,
) -> None:
    mapped_primary_axis = map_axis(owner.axes.primary_axis)
    graph.add_node(mapped_primary_axis)
    _add_labelled_edge(
        graph, mapped_owner, mapped_primary_axis, PRIMARY_AXIS_LABEL, map_edge=map_edge
    )
    graph.add_nodes_from(
        map_axis(orienting_axis) for orienting_axis in owner.axes.orienting_axes
    )
    for orienting_axis in owner.axes.orienting_axes:
        mapped_axis = map_axis(orienting_axis)
        _add_labelled_edge(
            graph, mapped_owner, mapped_axis, HAS_AXIS_LABEL, map_edge=map_edge
        )

    # the relations between those axes becomes edges
    for axis_relation in owner.axes.axis_relations:
        mapped_arg1 = map_axis(axis_relation.first_slot)
        mapped_arg2 = map_axis(axis_relation.second_slot)
        _add_labelled_edge(
            graph,
            mapped_arg1,
            mapped_arg2,
            axis_relation.relation_type,
            map_edge=map_edge,
        )


def _translate_geon(
    graph: DiGraph,
    owner: MaybeHasGeon,
    *,
    mapped_owner: Any,
    map_geon: Callable[[Geon], Any],
    map_axis: _AxisMapper,
    map_edge: _EdgeMapper,
) -> None:
    if owner.geon:
        mapped_geon = map_geon(owner.geon)
        graph.add_node(mapped_geon)
        _add_labelled_edge(
            graph, mapped_owner, mapped_geon, HAS_GEON_LABEL, map_edge=map_edge
        )
        _translate_axes(graph, owner.geon, mapped_geon, map_axis, map_edge)
        mapped_generating_axis = map_axis(owner.geon.generating_axis)
        _add_labelled_edge(
            graph,
            mapped_geon,
            mapped_generating_axis,
            GENERATING_AXIS_LABEL,
            map_edge=map_edge,
        )


def _translate_region(
    graph: DiGraph,
    region: Region[Any],
    *,
    map_node: Callable[[Any], Any],
    map_edge: _EdgeMapper,
    axes_info: Optional[AxesInfo[Any]] = None,
) -> None:
    mapped_region = map_node(region)
    mapped_reference_object = map_node(region.reference_object)
    _add_labelled_edge(
        graph,
        mapped_region,
        mapped_reference_object,
        REFERENCE_OBJECT_LABEL,
        map_edge=map_edge,
    )
    if region.direction:
        axis_relative_to = region.direction.relative_to_axis.to_concrete_axis(axes_info)
        mapped_axis_relative_to = map_node(axis_relative_to)
        _add_labelled_edge(
            graph,
            mapped_region,
            mapped_axis_relative_to,
            region.direction,
            map_edge=map_edge,
        )


# This is used to control the order in which pattern nodes are matched,
# which can have a significant impact on match speed.
# We try to match the most restrictive nodes first.
_PATTERN_PREDICATE_NODE_ORDER = [
    # If we have matchedObjects in the pattern we want to try and find these first.
    MatchedObjectPerceptionPredicate,
    # properties and colors tend to be highlight restrictive, so let's match them first
    IsOntologyNodePredicate,
    IsColorNodePredicate,
    AnyObjectPerception,
    GeonPredicate,
    RegionPredicate,
    # the matcher tends to get bogged down when dealing with axes,
    # so we search those last one the other nodes have established the skeleton of a match.
    AxisPredicate,
]


def _pattern_matching_node_order(node_node_data_tuple) -> int:
    (node, _) = node_node_data_tuple
    return _PATTERN_PREDICATE_NODE_ORDER.index(node.__class__)


# This is used to control the order in which pattern nodes are matched,
# which can have a significant impact on match speed.
# This should match _PATTERN_PREDICATE_NODE_ORDER above.
_GRAPH_NODE_ORDER: List[  # type: ignore
    Type[
        Union[
            MatchedObjectNode,
            OntologyNode,
            RgbColorPerception,
            ObjectPerception,
            Geon,
            Region,
            GeonAxis,
        ]
    ]
] = [
    MatchedObjectNode,
    OntologyNode,
    RgbColorPerception,
    ObjectPerception,
    Geon,
    Region,
    GeonAxis,
]


def _graph_node_order(node_node_data_tuple) -> int:
    (node, _) = node_node_data_tuple
    if isinstance(node, tuple):
        # some node types are wrapped in tuples with unique ids to keep them distinct
        # (e.g. otherwise identical Geon objects).
        # We need to unwrap these before comparing types.
        node = node[0]

    return _GRAPH_NODE_ORDER.index(node.__class__)


@attrs(frozen=True, slots=True, eq=False)
class PrepositionPattern:
    graph_pattern: PerceptionGraphPattern = attrib(
        validator=instance_of(PerceptionGraphPattern), kw_only=True
    )
    object_map: Mapping[str, MatchedObjectPerceptionPredicate] = attrib(
        converter=_to_immutabledict, kw_only=True
    )

    def intersection(self, pattern: "PrepositionPattern") -> "PrepositionPattern":
        graph_pattern = self.graph_pattern.intersection(pattern.graph_pattern)
        mapping_builder = []
        items_to_iterate: List[Tuple[str, MatchedObjectPerceptionPredicate]] = []
        items_to_iterate.extend(self.object_map.items())
        items_to_iterate.extend(pattern.object_map.items())
        for name, pattern_node in immutableset(items_to_iterate):
            if (
                pattern_node
                in graph_pattern._graph.nodes  # pylint:disable=protected-access
            ):
                mapping_builder.append((name, pattern_node))

        return PrepositionPattern(graph_pattern=graph_pattern, object_map=mapping_builder)


_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


def _invert_to_immutabledict(mapping: Mapping[_KT, _VT]) -> ImmutableDict[_VT, _KT]:
    return immutabledict((v, k) for (k, v) in mapping.items())
