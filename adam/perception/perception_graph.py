r"""
Code for representing `DevelopmentalPrimitivePerceptionFrame`\ s as directed graphs
and for matching patterns against such graph.
Such patterns could be used to implement object recognition,
among other things.

This file first defines `PerceptionGraph`\ s,
then defines `PerceptionGraphPattern`\ s to match them.

The MatchedObjectNode is defined at the top of this module as it is needed prior
to defining the type of Nodes in our `PerceptionGraph`\ s readers should start with
`PerceptionGraphProtocol`, `PerceptionGraph`, and `PerceptionGraphPattern` before
reading other parts of this module.
"""
import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from time import process_time
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Sized,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)
from uuid import uuid4

import graphviz
from attr.validators import instance_of, optional
from immutablecollections import ImmutableDict, ImmutableSet, immutabledict, immutableset
from immutablecollections.converter_utils import _to_immutabledict, _to_tuple
from more_itertools import first
from networkx import DiGraph, connected_components, is_isomorphic, set_node_attributes
from typing_extensions import Protocol
from vistautils.misc_utils import str_list_limited
from vistautils.preconditions import check_arg
from vistautils.range import Range

from adam.axes import AxesInfo, HasAxes
from adam.axis import GeonAxis
from adam.geon import Geon, MaybeHasGeon
from adam.ontology import OntologyNode
from adam.ontology.ontology import Ontology
from adam.ontology.phase1_ontology import GAILA_PHASE_1_ONTOLOGY, PART_OF, COLOR
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
from adam.utils.networkx_utils import copy_digraph, digraph_with_nodes_sorted_by, subgraph
from attr import attrib, attrs


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
    has been identified in the graph
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
FACING_OBJECT_LABEL = OntologyNode("facing-axis")
"""
Edge label in a `PerceptionGraph` linking an `Axis` to a `ObjectPerception` it is facing
"""


class PerceptionGraphProtocol(Protocol):
    _graph: DiGraph = attrib(validator=instance_of(DiGraph))

    def copy_as_digraph(self) -> DiGraph:
        return self._graph.copy()

    def render_to_file(
        self,
        graph_name: str,
        output_file: Path,
        *,
        match_correspondence_ids: Mapping[Any, str] = immutabledict(),
        robust=True,
    ) -> None:
        """
        Debugging tool to render the graph to PDF using *dot*.
        """


@attrs(frozen=True, repr=False)
class PerceptionGraph(PerceptionGraphProtocol):
    r"""
    Represents a `DevelopmentalPrimitivePerceptionFrame` as a directed graph.

    `ObjectPerception`\ s, properties, `Geon`\ s, `GeonAxis`\ s, and `Region`\ s are nodes.

    These can be matched against by `PerceptionGraphPattern`\ s.
    """
    _graph: DiGraph = attrib(validator=instance_of(DiGraph), converter=copy_digraph)

    @staticmethod
    def from_frame(frame: DevelopmentalPrimitivePerceptionFrame) -> "PerceptionGraph":
        """
        Gets the `PerceptionGraph` corresponding to a `DevelopmentalPrimitivePerceptionFrame`.
        """
        graph = DiGraph()

        def map_node(obj: Any, *, force_unique_counter: Optional[int] = None):
            # in some cases, a special index will be given to force
            # otherwise identical objects to be treated separately.
            # We do this for properties, for example, so that if two things
            # are both animate, they end up with distinct animacy nodes in the graph
            # which could be e.g. treated differently during pattern relaxation.
            if force_unique_counter is not None:
                return (obj, force_unique_counter)
            # Regions and Geons are normally treated as value objects,
            # but we want to maintain their distinctness in the perceptual graph
            # for the purpose of matching patterns, so we make their corresponding
            # graph nods compare by identity.
            elif isinstance(obj, (Region, Geon)):
                return (obj, id(obj))
            else:
                return obj

        def map_edge(label: Any):
            return {"label": label}

        # see force_unique_counter in map_node above
        property_index = 0

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
            property_index += 1
            source_node = map_node(property_.perceived_object)
            if isinstance(property_, HasBinaryProperty):
                dest_node = map_node(
                    property_.binary_property, force_unique_counter=property_index
                )
            elif isinstance(property_, HasColor):
                dest_node = map_node(property_.color, force_unique_counter=property_index)
            else:
                raise RuntimeError(f"Don't know how to translate property {property_}")
            graph.add_edge(source_node, dest_node, label=HAS_PROPERTY_LABEL)

        if frame.axis_info:
            for (object_, axis) in frame.axis_info.axes_facing.items():
                graph.add_edge(axis, object_, label=FACING_OBJECT_LABEL)

        return PerceptionGraph(graph)

    def render_to_file(  # pragma: no cover
        self,
        graph_name: str,
        output_file: Path,
        *,
        match_correspondence_ids: Mapping[Any, str] = immutabledict(),
        robust=True,
    ) -> None:
        """
        Debugging tool to render the graph to PDF using *dot*.

        If this graph has been matched against a pattern,
        the matched nodes can be highlighted and given labels which show
        what the correspond to in a pattern by supplying
        *match_correspondence_ids* which maps graph nodes to
        the desired correspondence labels.

        If *robust* is *True* (the default), then this will suppress crashes on failures.
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

        try:
            dot_graph.render(str(output_file))
        except Exception as e:  # pylint:disable=broad-except
            if robust:
                logging.warning("Error during dot rendering: %s", e)
            else:
                raise

    def _to_dot_node(
        self,
        dot_graph: graphviz.Digraph,
        perception_node: PerceptionGraphNode,
        next_node_id: Incrementer,
        match_correspondence_ids: Mapping[Any, str],
    ) -> str:
        if isinstance(perception_node, tuple):
            unwrapped_perception_node = perception_node[0]
        else:
            unwrapped_perception_node = perception_node

        # object perceptions have no content, so they are blank nodes
        if isinstance(unwrapped_perception_node, ObjectPerception):
            label = unwrapped_perception_node.debug_handle
        # regions do have content but we express those as edges to other nodes
        elif isinstance(unwrapped_perception_node, Region):
            label = f"reg:{unwrapped_perception_node}"
        elif isinstance(unwrapped_perception_node, GeonAxis):
            label = f"axis:{unwrapped_perception_node.debug_name}"
        elif isinstance(unwrapped_perception_node, RgbColorPerception):
            label = unwrapped_perception_node.hex
        elif isinstance(unwrapped_perception_node, OntologyNode):
            label = unwrapped_perception_node.handle
        elif isinstance(unwrapped_perception_node, Geon):
            label = str(unwrapped_perception_node.cross_section) + str(
                unwrapped_perception_node.cross_section_size
            )
        elif isinstance(unwrapped_perception_node, MatchedObjectNode):
            label = " ".join(unwrapped_perception_node.name)
        else:
            raise RuntimeError(
                f"Do not know how to perception node render node "
                f"{unwrapped_perception_node} with dot"
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

    def copy_as_digraph(self) -> DiGraph:
        return self._graph.copy()

    def __repr__(self) -> str:
        return (
            f"PerceptionGraph(nodes={str_list_limited(self._graph.nodes, 10)}, edges="
            f"{str_list_limited(self._graph.edges(data='label'), 15)})"
        )


@attrs(frozen=True, slots=True, repr=False)
class PerceptionGraphPattern(PerceptionGraphProtocol, Sized):
    r"""
    A pattern which can match `PerceptionGraph`\ s.

    Such patterns could be used, for example, to represent a learner's
    knowledge of an object for object recognition.
    """

    _graph: DiGraph = attrib(validator=instance_of(DiGraph), converter=copy_digraph)

    def _node_match(
        self, node1: Dict[str, "NodePredicate"], node2: Dict[str, "NodePredicate"]
    ) -> bool:
        return node1["node"].is_equivalent(node2["node"])
        # TODO: Compare node predicates with isEquivalent

    def _edge_match(
        self, edge1: Dict[str, "EdgePredicate"], edge2: Dict[str, "EdgePredicate"]
    ) -> bool:
        return edge1["predicate"].dot_label() == edge2["predicate"].dot_label()

    def check_isomorphism(self, other_graph: "PerceptionGraphPattern") -> bool:
        """
        Compares two pattern graphs and returns true if they are isomorphic, including edges and node attributes.
        """
        return is_isomorphic(
            self._graph,
            other_graph.copy_as_digraph(),
            node_match=self._node_match,
            edge_match=self._edge_match,
        )

    def matcher(
        self,
        graph_to_match_against: PerceptionGraphProtocol,
        *,
        debug_callback: Optional[DebugCallableType] = None,
        matching_objects: bool,
    ) -> "PatternMatching":
        """
        Creates an object representing an attempt to match this pattern
        against *graph_to_match_against*.
        """
        return PatternMatching(
            pattern=self,
            graph_to_match_against=graph_to_match_against,
            debug_callback=debug_callback,
            matching_objects=matching_objects,
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
            situation, chooser=RandomChooser.for_seed(0), include_ground=False
        )
        perception_graph = PerceptionGraph.from_frame(
            first(perception.frames)
        ).copy_as_digraph()

        # We remove color and properties that are added by situation generation,
        # but are not in schemas.
        nodes_to_remove = [
            node
            for node in perception_graph
            # Perception generation wraps properties in tuples, so we need to unwrap them.
            if isinstance(node, tuple)
            and (
                isinstance(node[0], RgbColorPerception)
                or isinstance(node[0], OntologyNode)
            )
        ]
        perception_graph.remove_nodes_from(nodes_to_remove)

        # Finally, we convert the PerceptionGraph DiGraph representation to a PerceptionGraphPattern
        return PerceptionGraphPattern.from_graph(
            perception_graph=perception_graph
        ).perception_graph_pattern

    @staticmethod
    def from_graph(
        perception_graph: Union[DiGraph, PerceptionGraph]
    ) -> "PerceptionGraphPatternFromGraph":
        """
        Creates a pattern for recognizing an object based on its *perception_graph*.
        """
        if isinstance(perception_graph, PerceptionGraph):
            perception_graph = perception_graph._graph  # pylint:disable=protected-access

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

    def __len__(self) -> int:
        return len(self._graph)

    def copy_as_digraph(self) -> DiGraph:
        return self._graph.copy()

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
                # first unwrap any nodes which have been tuple-ized to force distinctness.
                if isinstance(node, tuple):
                    node = node[0]

                if isinstance(node, Geon):
                    perception_node_to_pattern_node[key] = GeonPredicate.exactly_matching(
                        node
                    )
                elif isinstance(node, Region):
                    perception_node_to_pattern_node[
                        key
                    ] = RegionPredicate.matching_distance(node)
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

        # We add every node in the source graph, after translating their type with map_node .
        for original_node in perception_graph.nodes:
            # Add each node
            pattern_node = map_node(original_node)
            pattern_graph.add_node(pattern_node)
            set_node_attributes(pattern_graph, {pattern_node: {"node": pattern_node}})

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
        robust=True,
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

        try:
            dot_graph.render(str(output_file))
        except Exception as e:  # pylint:disable=broad-except
            if robust:
                logging.warning("Error during dot rendering: %s", e)
            else:
                raise

    def intersection(
        self,
        graph_pattern: "PerceptionGraphPattern",
        *,
        debug_callback: Optional[DebugCallableType] = None,
        graph_logger: Optional["GraphLogger"] = None,
        ontology: Ontology,
    ) -> Optional["PerceptionGraphPattern"]:
        """
        Determine the largest partial match between two `PerceptionGraphPattern`s

        The algorithm used is approximate and is not guaranteed to return the largest
        possible match.
        """
        matcher = PatternMatching(
            pattern=graph_pattern,
            graph_to_match_against=self,
            debug_callback=debug_callback,
            matching_pattern_against_pattern=True,
            matching_objects=False,
        )
        attempted_match = matcher.relax_pattern_until_it_matches(
            graph_logger=graph_logger, ontology=ontology
        )
        if attempted_match:
            return attempted_match
        else:
            return None

    def __repr__(self) -> str:
        return (
            f"PerceptionGraphPattern(nodes={str_list_limited(self._graph.nodes, 10)}, "
            f"edges="
            f"{str_list_limited(self._graph.edges(data='predicate'), 10)})"
        )


class DumpPartialMatchCallback:
    """
        Helper callable class for debugging purposes. An instance of this object can be provided as the `debug_callback` argument of `GraphMatching.match` to render
        the match search process at every 100 time steps. We start rendering after the first 60 seconds.
    """

    def __init__(
        self,
        render_path,
        seconds_to_wait_before_rendering: int = 60,
        dump_every_x_calls: int = 100,
    ) -> None:
        self.render_path = render_path
        self.calls_to_match_counter = 0
        self.start_time = process_time()
        self.seconds_to_wait_before_rendering = seconds_to_wait_before_rendering
        self.dump_every_x_calls = dump_every_x_calls

    def __call__(
        self, graph: DiGraph, graph_node_to_pattern_node: Dict[Any, Any]
    ) -> None:
        self.calls_to_match_counter += 1
        current_time = process_time()
        if (
            self.calls_to_match_counter % self.dump_every_x_calls == 0
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


@attrs(slots=True, eq=False)
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
    matching_pattern_against_pattern: bool = attrib()
    # the attrs mypy plugin complains for the below
    # "Non-default attributes not allowed after default attributes."
    # But that doesn't seem to be our situation? And it works fine?
    _matching_objects: bool = attrib(  # type: ignore
        validator=instance_of(bool), kw_only=True
    )

    # Callable object for debugging purposes. We use this to track the number of calls to match and render the graphs.
    debug_callback: Optional[DebugCallableType] = attrib(default=None, kw_only=True)

    @attrs(frozen=True, kw_only=True, auto_attribs=True)
    class MatchFailure:
        """
        Indicates a failed attempt at matching a `PerceptionGraphPattern`.

        *pattern_node_to_graph_node_for_largest_match* indicates the partial match found
        with the largest number of nodes.
        Note that this is not necessarily the largest possible partial match,
        just the largest one encountered during the search process
        before the algorithm decided that a full match was not possible.

        *last_failed_pattern_node* is that the last pattern node attempted to be matched
        at the point the algorithm decided a match was impossible.
        Note that this is not necessarily the node responsible for the match failure;
        it could fail due to the edge predicate on the connecting edge,
        or it could fail due to there being no proper match for a pattern node
        which had no good alignment earlier in the match process.
        """

        pattern: PerceptionGraphPattern
        graph: PerceptionGraphProtocol
        pattern_node_to_graph_node_for_largest_match: Dict[Any, Any]
        last_failed_pattern_node: "NodePredicate"
        largest_match_pattern_subgraph: PerceptionGraphPattern = attrib()
        # TODO: the below is just a DiGraph because these is currently overloaded
        # to return match failures for both pattern-perception graph
        # and pattern-pattern matches.
        # It can be made a PerceptionGraph or PerceptionGraphPattern pending
        # https://github.com/isi-vista/adam/issues/489
        largest_match_graph_subgraph: DiGraph = attrib()

        def __attrs_post_init__(self) -> None:
            if (
                self.last_failed_pattern_node
                not in self.pattern._graph.nodes  # pylint:disable=protected-access
            ):

                raise RuntimeError(
                    f"Something has gone wrong: the pattern "
                    f"does not contain the failed node:"
                    f"{self.last_failed_pattern_node}"
                )

            if set(self.pattern_node_to_graph_node_for_largest_match.keys()) != set(
                self.largest_match_pattern_subgraph._graph.nodes  # pylint:disable=protected-access
            ):
                raise RuntimeError(
                    "Mismatch between node alignment and largest partial " "pattern match"
                )

            if set(self.pattern_node_to_graph_node_for_largest_match.values()) != set(
                self.largest_match_graph_subgraph.nodes
            ):
                raise RuntimeError(
                    "Mismatch between node alignment and largest partial graph " "match"
                )

        @largest_match_pattern_subgraph.default  # noqa: F821
        def _matched_pattern_subgraph_default(self) -> DiGraph:
            return PerceptionGraphPattern(
                self.pattern._graph.subgraph(  # pylint:disable=protected-access
                    self.pattern_node_to_graph_node_for_largest_match.keys()
                )
            )

        @largest_match_graph_subgraph.default  # noqa: F821
        def _matched_graph_subgraph_default(self) -> DiGraph:
            return self.graph._graph.subgraph(  # pylint:disable=protected-access
                immutableset(self.pattern_node_to_graph_node_for_largest_match.values())
            )

    def matches(
        self,
        *,
        use_lookahead_pruning: bool,
        suppress_multiple_alignments_to_same_nodes: bool = True,
        initial_partial_match: Mapping[Any, Any] = immutabledict(),
        graph_logger: Optional["GraphLogger"] = None,
    ) -> Iterable["PerceptionGraphPatternMatch"]:
        """
        Attempt the matching and returns a generator over the set of possible matches.

        If *suppress_multiple_alignments_to_same_nodes* is *True* (default *True*),
        then only the first alignment encountered for a given set of nodes will be returned.
        This prevents you from e.g. getting multiple matches for different ways
        of aligning axes for symmetric objects.
        The cost is that we need to keep around a memory of previous node matches.

        *matching_pattern_against_pattern* indicates
        you are matching one pattern against another,
        rather than against a perception graph.
        This should get split off into its own distinct method:
        https://github.com/isi-vista/adam/issues/487
        """
        for match in self._internal_matches(
            graph_to_match_against=self.graph_to_match_against,
            pattern=self.pattern,
            debug_callback=self.debug_callback,
            use_lookahead_pruning=use_lookahead_pruning,
            suppress_multiple_alignments_to_same_nodes=suppress_multiple_alignments_to_same_nodes,
            initial_partial_match=initial_partial_match,
            graph_logger=graph_logger,
        ):
            # we want to ignore the failure objects returned for the use of
            # first_match_or_failure_info
            if isinstance(match, PerceptionGraphPatternMatch):
                yield match

    def first_match_or_failure_info(
        self,
        *,
        initial_partial_match: Mapping[Any, Any] = immutabledict(),
        graph_logger: Optional["GraphLogger"] = None,
    ) -> Union["PerceptionGraphPatternMatch", "PatternMatching.MatchFailure"]:
        """
        Gets the first match encountered of the pattern against the graph
        (which one is first is deterministic but undefined)
        or a `PatternMatching.MatchFailure` giving debugging information
        about a failed match attempt.
        """
        return first(
            self._internal_matches(
                graph_to_match_against=self.graph_to_match_against,
                pattern=self.pattern,
                debug_callback=self.debug_callback,
                # lookahead pruning messes up the debug information by making the
                # cause of failures appear "earlier" in the pattern than they really are.
                use_lookahead_pruning=False,
                suppress_multiple_alignments_to_same_nodes=True,
                initial_partial_match=initial_partial_match,
                graph_logger=graph_logger,
            )
        )

    def relax_pattern_until_it_matches(
        self,
        *,
        graph_logger: Optional["GraphLogger"] = None,
        ontology: Ontology,
        min_ratio: Optional[float] = None,
    ) -> Optional[PerceptionGraphPattern]:
        """
        Prunes or relaxes the *pattern* for this matching until it successfully matches
        using heuristic rules.

        If a matching relaxed `PerceptionGraphPattern` can be found, it is returned.
        Otherwise, *None* is returned.
        """

        min_num_nodes_to_continue = 1
        if min_ratio:
            check_arg(min_ratio in Range.open_closed(0.0, 1.0))
            min_num_nodes_to_continue = int(min_ratio * len(self.pattern))

        # We start with the original pattern and attempt to match progressively relaxed versions.
        cur_pattern: Optional[PerceptionGraphPattern] = self.pattern
        # When we try to match relaxed patterns, we will remember how much we could match
        # last time so we don't start from scratch.
        partial_match: Mapping[Any, Any] = {}

        relaxation_step = 0
        while (
            cur_pattern and len(cur_pattern) >= min_num_nodes_to_continue
        ):  # pylint:disable=len-as-condition
            match_attempt = first(
                self._internal_matches(
                    graph_to_match_against=self.graph_to_match_against,
                    pattern=cur_pattern,
                    debug_callback=None,
                    # Using lookahead pruning would make our guess at the "cause"
                    # of the match failure be too "early" in the pattern graph search.
                    use_lookahead_pruning=False,
                    suppress_multiple_alignments_to_same_nodes=True,
                    initial_partial_match=partial_match,
                )
            )
            if isinstance(match_attempt, PerceptionGraphPatternMatch):
                return cur_pattern
            else:
                relaxation_step += 1
                # If we couldn't successfully match the current part of the pattern,
                # chop off the node which we think might have caused the match to fail.
                # Why is this cast necessary? mypy should be able to infer this...
                cur_pattern = self._relax_pattern(
                    match_attempt, graph_logger=graph_logger, ontology=ontology
                )
                if graph_logger and cur_pattern:
                    graph_logger.log_graph(
                        cur_pattern,
                        logging.INFO,
                        "Relaxation step %s, cur pattern size %s",
                        relaxation_step,
                        len(cur_pattern),
                    )
        # no relaxation could successfully match
        return None

    def _internal_matches(
        self,
        *,
        graph_to_match_against: PerceptionGraphProtocol,
        pattern: PerceptionGraphPattern,
        debug_callback: Optional[Callable[[Any, Any], None]],
        use_lookahead_pruning: bool,
        suppress_multiple_alignments_to_same_nodes: bool = True,
        collect_debug_statistics: bool = False,
        initial_partial_match: Mapping[Any, Any] = immutabledict(),
        graph_logger: Optional["GraphLogger"] = None,
    ) -> Iterable[Union["PerceptionGraphPatternMatch", "PatternMatching.MatchFailure"]]:
        # Controlling the iteration order of the graphs
        # controls the order in which nodes are matched.
        # This has a significant, benchmark-confirmed impact on performance.
        sorted_graph_to_match_against = digraph_with_nodes_sorted_by(
            graph_to_match_against._graph,  # pylint: disable=W0212
            _graph_node_order
            if not self.matching_pattern_against_pattern
            else _pattern_matching_node_order,
        )
        sorted_pattern = digraph_with_nodes_sorted_by(
            pattern._graph, _pattern_matching_node_order  # pylint: disable=W0212
        )

        matching = GraphMatching(
            sorted_graph_to_match_against,
            sorted_pattern,
            use_lookahead_pruning=use_lookahead_pruning,
            matching_pattern_against_pattern=self.matching_pattern_against_pattern,
            matching_objects=self._matching_objects,
        )

        sets_of_nodes_matched: Set[ImmutableSet[PerceptionGraphNode]] = set()

        got_a_match = False
        if debug_callback:
            self.debug_callback = debug_callback
        for graph_node_to_matching_pattern_node in matching.subgraph_isomorphisms_iter(
            collect_debug_statistics=collect_debug_statistics,
            debug_callback=debug_callback,
            initial_partial_match=initial_partial_match,
        ):
            matched_graph_nodes: ImmutableSet[PerceptionGraphNode] = immutableset(
                graph_node_to_matching_pattern_node
            )
            if (
                matched_graph_nodes not in sets_of_nodes_matched
                or not suppress_multiple_alignments_to_same_nodes
            ):
                got_a_match = True
                yield PerceptionGraphPatternMatch(
                    graph_matched_against=graph_to_match_against,
                    matched_pattern=pattern,
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
        if not got_a_match:
            # mypy doesn't like the assignment to the pattern_node_to_graph_node_for_largest_match
            # argument for reasons which are unclear to me. It works fine, though.
            match_failure = PatternMatching.MatchFailure(  # type: ignore
                pattern=pattern,
                graph=graph_to_match_against,
                pattern_node_to_graph_node_for_largest_match=immutabledict(
                    matching.debug_largest_match
                ),
                last_failed_pattern_node=cast(
                    NodePredicate, matching.failing_pattern_node_for_deepest_match
                ),
            )
            if graph_logger:
                graph_logger.log_match_failure(
                    match_failure, logging.INFO, "Match failure"
                )
            yield match_failure

    def _relax_pattern(
        self,
        match_failure: "PatternMatching.MatchFailure",
        ontology: Ontology,
        *,
        graph_logger: Optional["GraphLogger"] = None,
    ) -> Optional[PerceptionGraphPattern]:
        """
        Attempts to produce a "relaxed" version of pattern
        which has a better chance of matching,
        given the clue that *last_failed_pattern_node* was the last pattern node
        which could not be aligned in the previous pattern alignment attempt.

        This is to support `relax_pattern_until_it_matches`.
        """
        if len(match_failure.pattern) == 1:
            # It's the end of the line: we've pruned out all the pattern nodes.
            return None

        pattern_as_digraph = match_failure.pattern.copy_as_digraph()

        # first, we delete the last_failed_pattern_node.
        # If that node was an object, we also recursively delete its sub-objects
        # and any Regions it or they belonged to.
        nodes_to_delete_directly: List[NodePredicate] = []

        RELATION_EDGES_TO_FOLLOW_WHEN_DELETING = {  # pylint:disable=invalid-name
            PART_OF,
            # IN_REGION,
        }

        def gather_nodes_to_excise(focus_node: NodePredicate) -> None:
            if focus_node in match_failure.pattern_node_to_graph_node_for_largest_match:
                # don't delete or continue deleting through a node which successfully matched
                return
            nodes_to_delete_directly.append(focus_node)

            # If this is an object, also excise any sub-objects.
            for predecessor in pattern_as_digraph.pred[focus_node]:
                edge_label = pattern_as_digraph.edges[predecessor, focus_node][
                    "predicate"
                ]
                if (
                    isinstance(edge_label, RelationTypeIsPredicate)
                    and edge_label.relation_type in RELATION_EDGES_TO_FOLLOW_WHEN_DELETING
                ):
                    gather_nodes_to_excise(predecessor)

        last_failed_node = match_failure.last_failed_pattern_node
        logging.info("Relaxation: last failed pattern node is %s", last_failed_node)

        if last_failed_node in match_failure.pattern_node_to_graph_node_for_largest_match:
            # This means the supposed "last_failed_node" was in fact matched successfully,
            # so the match failure must be because all alignments found were judged
            # illegal. In that case, no relaxation can fix things.
            logging.info("Match found is illegal; no relaxation can help")
            return None

        gather_nodes_to_excise(last_failed_node)

        same_color_nodes: List[NodePredicate]

        if isinstance(last_failed_node, IsColorNodePredicate):
            # We treat colors as a special case.
            # In a complex object (e.g. a dog), an object and a large number of its
            # sub-components will share the same color.
            # Usually if the color is irrelevant to a pattern in one place,
            # it is irrelevant in many other places.
            # Relaxing the pattern by peeling the color predicates off one by one can be very slow,
            # especially since such large patterns are slow to match in the first place.
            # Therefore, if we remove a color predicate in one place, we remove it
            # from the entire pattern.
            color = last_failed_node.color

            same_color_nodes = [
                node
                for node in pattern_as_digraph.nodes
                if isinstance(node, IsColorNodePredicate) and node.color == color
            ]
        elif isinstance(
            last_failed_node, IsOntologyNodePredicate
        ) and ontology.is_subtype_of(last_failed_node.property_value, COLOR):
            # same as above, but for the case where we are perceiving colors discretely.
            discrete_color = last_failed_node.property_value

            same_color_nodes = [
                node
                for node in pattern_as_digraph.nodes
                if isinstance(node, IsOntologyNodePredicate)
                and node.property_value == discrete_color
            ]
        else:
            same_color_nodes = []

        if same_color_nodes:
            logging.info("Deleting extra color nodes: %s", same_color_nodes)
        nodes_to_delete_directly.extend(same_color_nodes)

        logging.info("Nodes to delete directly: %s", nodes_to_delete_directly)

        pattern_as_digraph.remove_nodes_from(immutableset(nodes_to_delete_directly))

        if len(pattern_as_digraph) == 0:  # pylint:disable=len-as-condition
            # We deleted the whole graph, so no relaxation is possible.
            return None

        # The deletions above may have left disconncted "islands" in the pattern.
        # We remove these as well by deleting all connected components except the one
        # containing the successful portion of the pattern match.

        pattern_as_undirected_graph = pattern_as_digraph.to_undirected(as_view=True)

        if (
            len(  # pylint:disable=len-as-condition
                match_failure.largest_match_pattern_subgraph
            )
            > 0
        ):
            # We just need any node which did match so we know which connected component
            # to keep from the post-deletion graph
            probe_node = first(
                match_failure.largest_match_pattern_subgraph._graph.nodes  # pylint:disable=protected-access
            )

            to_delete_due_to_disconnection: List[NodePredicate] = []
            connected_components_containing_successful_pattern_matches = 0
            for connected_component in connected_components(pattern_as_undirected_graph):
                if probe_node not in connected_component:
                    # this is a component which is now disconnected from the successful partial match
                    to_delete_due_to_disconnection.extend(connected_component)
                else:
                    connected_components_containing_successful_pattern_matches += 1

            if connected_components_containing_successful_pattern_matches != 1:
                if graph_logger:
                    graph_logger.log_match_failure(
                        match_failure, logging.INFO, "Pattern match component failure"
                    )
                raise RuntimeError(
                    f"Expected the successfully matching portion of the pattern"
                    f" to belong to a single connected component, but it was in "
                    f"{connected_components_containing_successful_pattern_matches}"
                )

            logging.info(
                "Relaxation: deleted due to disconnection: %s",
                to_delete_due_to_disconnection,
            )
            pattern_as_digraph.remove_nodes_from(to_delete_due_to_disconnection)
        else:
            # If nothing was successfully matched, it is less clear how to choose what portion
            # of the pattern to keep. For now, we are going to keep the largest connected component.
            # BEWARE: we have not confirmed the order of connected components is deterministic.
            components = sorted(
                list(connected_components(pattern_as_undirected_graph)),
                key=lambda x: len(x),  # pylint:disable=unnecessary-lambda
            )

            # We know there is some component because otherwise we would have bailed out
            # after we did the first deletion pass above.
            biggest_component = first(components)
            for component in components:
                if component != biggest_component:
                    logging.info(
                        "Relaxation: deleted due to being in smaller component: %s",
                        component,
                    )
                    pattern_as_digraph.remove_nodes_from(component)

        return PerceptionGraphPattern(pattern_as_digraph.copy())

    @matching_pattern_against_pattern.default
    def _matching_pattern_against_pattern_default(self) -> bool:
        if isinstance(self.graph_to_match_against, PerceptionGraphPattern):
            return True
        elif isinstance(self.graph_to_match_against, PerceptionGraph):
            return False
        else:
            raise RuntimeError(
                f"Don't know how to match against: {self.graph_to_match_against}"
            )


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
    def is_equivalent(self, other: "NodePredicate") -> bool:
        """
        Compares two predicates and return true if they are equivalent
        """

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

    def is_equivalent(self, other) -> bool:
        return isinstance(other, AndNodePredicate)

    def matches_predicate(self, predicate_node: "NodePredicate") -> bool:
        return isinstance(predicate_node, AnyNodePredicate)


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

    def is_equivalent(self, other) -> bool:
        return isinstance(other, AnyObjectPerception)

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

    def is_equivalent(self, other) -> bool:
        if isinstance(other, AxisPredicate):
            return (
                self.aligned_to_gravitational == other.aligned_to_gravitational
                and self.curved == other.curved
                and self.directed == other.directed
            )
        return False

    def matches_predicate(self, predicate_node: "NodePredicate") -> bool:
        if isinstance(predicate_node, AxisPredicate):
            if self.curved != predicate_node.curved:
                return False
            if self.directed != predicate_node.directed:
                return False
            if self.aligned_to_gravitational != predicate_node.aligned_to_gravitational:
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

    def is_equivalent(self, other) -> bool:
        if isinstance(other, GeonPredicate):
            return (
                self.template_geon.axes.axis_relations
                == other.template_geon.axes.axis_relations
                and self.template_geon.axes.orienting_axes
                == other.template_geon.axes.orienting_axes
                and self.template_geon.axes.primary_axis
                == other.template_geon.axes.primary_axis
                and self.template_geon.cross_section.curved
                == other.template_geon.cross_section.curved
                and self.template_geon.cross_section.has_reflective_symmetry
                == other.template_geon.cross_section.has_reflective_symmetry
                and self.template_geon.cross_section.has_rotational_symmetry
                == other.template_geon.cross_section.has_rotational_symmetry
                and self.template_geon.cross_section_size.name
                == other.template_geon.cross_section_size.name
                and self.template_geon.generating_axis.curved
                == other.template_geon.generating_axis.curved
                and self.template_geon.generating_axis.directed
                == other.template_geon.generating_axis.directed
                and self.template_geon.generating_axis.aligned_to_gravitational
                == other.template_geon.generating_axis.aligned_to_gravitational
            )
        return False

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

    def is_equivalent(self, other) -> bool:
        if isinstance(other, RegionPredicate):
            if self.distance is None and other.distance is None:
                return True
            elif self.distance is not None and other.distance is not None:
                return self.distance.name == other.distance.name
        return False

    def matches_predicate(self, predicate_node: "NodePredicate") -> bool:
        if isinstance(predicate_node, RegionPredicate):
            return predicate_node.distance == self.distance
        else:
            return False


@attrs(frozen=True, slots=True, eq=False)
class IsOntologyNodePredicate(NodePredicate):
    property_value: OntologyNode = attrib(validator=instance_of(OntologyNode))

    def __call__(self, graph_node: PerceptionGraphNode) -> bool:
        # ontology nodes might be wrapped in tuples with their id()
        # in order to simulate comparison by object ID.
        if isinstance(graph_node, tuple):
            graph_node = graph_node[0]
        return graph_node == self.property_value

    def dot_label(self) -> str:
        return f"prop({self.property_value.handle})"

    def is_equivalent(self, other) -> bool:
        if isinstance(other, IsOntologyNodePredicate):
            return self.property_value == other.property_value
        return False

    def matches_predicate(self, predicate_node: "NodePredicate") -> bool:
        if isinstance(predicate_node, IsOntologyNodePredicate):
            return predicate_node.property_value == self.property_value
        return False


@attrs(frozen=True, slots=True, eq=False)
class IsColorNodePredicate(NodePredicate):
    color: RgbColorPerception = attrib(validator=instance_of(RgbColorPerception))

    def __call__(self, graph_node: PerceptionGraphNode) -> bool:
        # color nodes might be wrapped in tuples with their id()
        # in order to simulate comparison by object ID.
        if isinstance(graph_node, tuple):
            graph_node = graph_node[0]

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
        return False

    def dot_label(self) -> str:
        return f"prop({self.color.hex})"

    def is_equivalent(self, other) -> bool:
        if isinstance(other, IsColorNodePredicate):
            return self.color.hex == other.color.hex
        return False


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

    def is_equivalent(self, other) -> bool:
        if isinstance(other, AndNodePredicate) and len(self.sub_predicates) == len(
            other.sub_predicates
        ):
            return all(
                any(pred1.is_equivalent(pred2) for pred1 in self.sub_predicates)
                for pred2 in other.sub_predicates
            )
        return False

    def matches_predicate(self, predicate_node: "NodePredicate") -> bool:
        raise NotImplementedError(
            f"Matches Predicate between AndNodePredicate " f"is not yet implemented"
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

    def is_equivalent(self, other) -> bool:
        return isinstance(other, MatchedObjectPerceptionPredicate)


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
    def matches_predicate(self, edge_label: "EdgePredicate") -> bool:
        """
        Returns whether *edge_label* matches *self*
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

    def matches_predicate(self, edge_label: "EdgePredicate") -> bool:
        return isinstance(edge_label, AnyEdgePredicate)


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

    def matches_predicate(self, edge_label: "EdgePredicate") -> bool:
        return (
            isinstance(edge_label, RelationTypeIsPredicate)
            and edge_label.relation_type == self.relation_type
        )


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

    def matches_predicate(self, edge_label: "EdgePredicate") -> bool:
        return isinstance(edge_label, DirectionPredicate) and (
            edge_label.reference_direction.positive == self.reference_direction.positive
        )


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


GOVERNED = OntologyNode("governed")
"""
An object match governed in a preposition relationship
"""
MODIFIED = OntologyNode("modified")
"""
An object match modified in a preposition relationship
"""


_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


def _invert_to_immutabledict(mapping: Mapping[_KT, _VT]) -> ImmutableDict[_VT, _KT]:
    return immutabledict((v, k) for (k, v) in mapping.items())


@attrs
class GraphLogger:
    log_directory: Path = attrib(validator=instance_of(Path))
    enable_graph_rendering: bool = attrib(validator=instance_of(bool))
    serialize_graphs: bool = attrib(validator=instance_of(bool), default=False)
    call_count: int = attrib(init=False, default=0)

    def log_graph(
        self,
        graph: PerceptionGraphProtocol,
        level,
        msg: str,
        *args,
        match_correspondence_ids: Mapping[Any, str] = immutabledict(),
        graph_name: Optional[str] = None,
    ) -> None:
        self.call_count += 1
        if self.enable_graph_rendering:
            if not graph_name:
                graph_name = str(uuid4())

            filename = self.log_directory / f"{graph_name}"
            graph.render_to_file(
                graph_name, filename, match_correspondence_ids=match_correspondence_ids
            )
            logging.log(
                level, f"[{self.call_count}] Rendered to {filename}.pdf\n{msg}", *args
            )
            if self.serialize_graphs:
                serialized_file = self.log_directory / f"{graph_name}.serialized"
                logging.info("Serializing to %s", serialized_file)
                with open(str(serialized_file), "wb") as out:
                    pickle.dump(graph, out)
        else:
            logging.log(level, msg, *args)

    def log_match_failure(
        self,
        match_failure: PatternMatching.MatchFailure,
        level,
        msg: str,
        *args,
        graph_name: Optional[str] = None,
    ) -> None:
        if not graph_name:
            graph_name = str(uuid4())

        if self.enable_graph_rendering:
            graph_correspondence_ids: Mapping[Any, str] = immutabledict(
                (graph_node, str(i))
                for (i, (_, graph_node)) in enumerate(
                    match_failure.pattern_node_to_graph_node_for_largest_match.items()
                )
            )
            self.log_graph(
                match_failure.graph,
                level,
                msg + " [graph] ",
                *args,
                match_correspondence_ids=graph_correspondence_ids,
                graph_name=f"{graph_name}-graph",
            )

            pattern_correspondence_ids: Mapping[Any, str] = immutabledict(
                (pattern_node, str(i))
                for (i, (pattern_node, _)) in enumerate(
                    match_failure.pattern_node_to_graph_node_for_largest_match.items()
                )
            )
            self.log_graph(
                match_failure.pattern,
                level,
                msg + " [pattern] ",
                *args,
                match_correspondence_ids=pattern_correspondence_ids,
                graph_name=f"{graph_name}-pattern",
            )
        else:
            logging.log(level, msg, *args)

    def log_pattern_match(
        self,
        pattern_match: PerceptionGraphPatternMatch,
        level,
        msg: str,
        *args,
        graph_name: Optional[str] = None,
    ) -> None:
        if not graph_name:
            graph_name = str(uuid4())

        if self.enable_graph_rendering:
            graph_correspondence_ids: Mapping[Any, str] = immutabledict(
                (graph_node, str(i))
                for (i, (_, graph_node)) in enumerate(
                    pattern_match.pattern_node_to_matched_graph_node.items()
                )
            )
            self.log_graph(
                pattern_match.graph_matched_against,
                level,
                msg + " [graph] ",
                *args,
                match_correspondence_ids=graph_correspondence_ids,
                graph_name=f"{graph_name}-graph",
            )

            pattern_correspondence_ids: Mapping[Any, str] = immutabledict(
                (pattern_node, str(i))
                for (i, (pattern_node, _)) in enumerate(
                    pattern_match.pattern_node_to_matched_graph_node.items()
                )
            )
            self.log_graph(
                pattern_match.matched_pattern,
                level,
                msg + " [pattern] ",
                *args,
                match_correspondence_ids=pattern_correspondence_ids,
                graph_name=f"{graph_name}-pattern",
            )
        else:
            logging.log(level, msg, *args)
