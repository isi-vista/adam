import random as r
from itertools import chain
from typing import cast

import pytest
from more_itertools import first, only
from networkx import DiGraph

from adam.curriculum.curriculum_utils import (
    PHASE1_CHOOSER_FACTORY,
    phase1_instances,
    standard_object,
)
from adam.learner import graph_without_learner
from adam.ontology import IN_REGION, OntologyNode
from adam.ontology.phase1_ontology import (
    BIRD,
    BOX,
    GAILA_PHASE_1_ONTOLOGY,
    GROUND,
    HOUSE,
    INANIMATE_OBJECT,
    IS_BODY_PART,
    LIQUID,
    PART_OF,
    TABLE,
    _BALL_SCHEMA,
    _HOUSE_SCHEMA,
    _TABLE_SCHEMA,
    above,
    bigger_than,
    on,
)
from adam.ontology.phase1_spatial_relations import DISTAL, EXTERIOR_BUT_IN_CONTACT, Region
from adam.ontology.structural_schema import ObjectStructuralSchema
from adam.perception import ObjectPerception, PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
    RgbColorPerception,
)
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    GAILA_PHASE_1_PERCEPTION_GENERATOR,
)
from adam.perception.perception_graph import (
    IsColorNodePredicate,
    PatternMatching,
    PerceptionGraph,
    PerceptionGraphPattern,
    PerceptionGraphPatternMatch,
    TemporalScope,
    TemporallyScopedEdgeLabel,
    HoldsAtTemporalScopePredicate,
    AnyEdgePredicate,
    AnyObjectPerception,
    PerceptionGraphNode,
)
from adam.random_utils import RandomChooser
from adam.relation import Relation
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    all_possible,
    color_variable,
    object_variable,
)
from adam_test_utils import all_possible_test
from immutablecollections import immutableset, immutablesetmultidict

from tests.perception import MatchMode

r.seed(0)


def test_temporally_scoped_attribute():
    before = TemporallyScopedEdgeLabel.for_dynamic_perception(
        "foo", when=TemporalScope.BEFORE
    )
    assert before.attribute == "foo"
    assert before.temporal_specifiers == immutableset([TemporalScope.BEFORE])

    before_and_after = TemporallyScopedEdgeLabel.for_dynamic_perception(
        "foo", when=[TemporalScope.BEFORE, TemporalScope.AFTER]
    )
    assert before_and_after.attribute == "foo"
    assert before_and_after.temporal_specifiers == immutableset(
        [TemporalScope.BEFORE, TemporalScope.AFTER]
    )

    # You have to give a TemporalSpecifier
    with pytest.raises(RuntimeError):
        TemporallyScopedEdgeLabel.for_dynamic_perception("foo", when=[])


def test_house_on_table():
    assert do_object_on_table_test(HOUSE, _HOUSE_SCHEMA, BIRD)


def test_objects_individually():
    for object_ in GAILA_PHASE_1_ONTOLOGY.nodes_with_properties(
        INANIMATE_OBJECT, banned_properties=[LIQUID, IS_BODY_PART]
    ):
        if object_ not in [BIRD, TABLE, GROUND]:
            schemata = GAILA_PHASE_1_ONTOLOGY.structural_schemata(object_)
            if len(schemata) == 1:
                print(f"Matching {object_}")
                assert do_object_on_table_test(object_, first(schemata), BIRD)


def test_inanimate_objects():
    """
    Tests whether several inanimate objects can be matched.
    """
    for object_ in GAILA_PHASE_1_ONTOLOGY.nodes_with_properties(
        INANIMATE_OBJECT, banned_properties=[LIQUID, IS_BODY_PART]
    ):
        if object_ != BIRD and object_ != TABLE:
            schemata = GAILA_PHASE_1_ONTOLOGY.structural_schemata(object_)
            if len(schemata) == 1 and object_ != GROUND:
                print(f"Matching {object_}")
                if do_object_on_table_test(object_, first(schemata), BIRD):
                    print(f"{object_} passed")
                else:
                    print(f"{object_} failed")


def do_object_on_table_test(
    object_type_to_match: OntologyNode,
    object_schema: ObjectStructuralSchema,
    negative_object_ontology_node: OntologyNode,
):
    """
    Tests the `PerceptionGraphMatcher` can match simple objects.
    """
    # we create four situations:
    # a object_to_match above or under a table with color red or blue
    color = color_variable("color")
    object_to_match = object_variable(
        debug_handle=object_type_to_match.handle,
        root_node=object_type_to_match,
        added_properties=[color],
    )
    table = standard_object("table_0", TABLE)

    object_on_table_template = Phase1SituationTemplate(
        "object_to_match-on-table",
        salient_object_variables=[object_to_match, table],
        asserted_always_relations=[
            bigger_than(table, object_to_match),
            on(object_to_match, table),
        ],
    )

    object_under_table_template = Phase1SituationTemplate(
        "object_to_match-under-table",
        salient_object_variables=[object_to_match, table],
        asserted_always_relations=[
            bigger_than(table, object_to_match),
            above(table, object_to_match),
        ],
    )

    # We test that a perceptual pattern for "object_to_match" matches in all four cases.
    object_to_match_pattern = PerceptionGraphPattern.from_schema(object_schema)

    situations_with_object_to_match = chain(
        all_possible_test(object_on_table_template),
        all_possible_test(object_under_table_template),
    )

    for (_, situation_with_object) in enumerate(situations_with_object_to_match):
        perception = GAILA_PHASE_1_PERCEPTION_GENERATOR.generate_perception(
            situation_with_object, chooser=RandomChooser.for_seed(0)
        )
        perception_graph = PerceptionGraph.from_frame(perception.frames[0])
        # perception_graph.render_to_file(f"object_to_match {idx}", out_dir / f"object_to_match
        # -{idx}.pdf")
        # object_to_match_pattern.render_to_file(f"object_to_match pattern", out_dir /
        # "object_to_match_pattern.pdf")
        matcher = object_to_match_pattern.matcher(
            perception_graph, match_mode=MatchMode.OBJECT
        )
        # debug_matching = matcher.debug_matching(
        #    use_lookahead_pruning=False, render_match_to=Path("/Users/gabbard/tmp")
        # )
        result = any(matcher.matches(use_lookahead_pruning=False))
        if not result:
            return False

    # Now let's create the same situations, but substitute a negative_object for a object_to_match.
    negative_object = object_variable(
        debug_handle=negative_object_ontology_node.handle,
        root_node=negative_object_ontology_node,
        added_properties=[color],
    )
    negative_object_on_table_template = Phase1SituationTemplate(
        "negative_object-on-table",
        salient_object_variables=[negative_object, table],
        asserted_always_relations=[
            bigger_than(table, negative_object),
            on(negative_object, table),
        ],
    )

    negative_object_under_table_template = Phase1SituationTemplate(
        "negative_object-under-table",
        salient_object_variables=[negative_object, table],
        asserted_always_relations=[
            bigger_than(table, negative_object),
            above(table, negative_object),
        ],
    )

    situations_with_negative_object = chain(
        all_possible_test(negative_object_on_table_template),
        all_possible_test(negative_object_under_table_template),
    )

    # The pattern should now fail to match.
    for situation_with_negative_object in situations_with_negative_object:
        perception = GAILA_PHASE_1_PERCEPTION_GENERATOR.generate_perception(
            situation_with_negative_object, chooser=RandomChooser.for_seed(0)
        )
        perception_graph = PerceptionGraph.from_frame(perception.frames[0])
        if any(
            object_to_match_pattern.matcher(
                perception_graph, match_mode=MatchMode.OBJECT
            ).matches(use_lookahead_pruning=True)
        ):
            return False
    return True


def test_last_failed_pattern_node():
    """
    Tests whether `MatchFailure` can find the correct node.
    """

    target_object = BOX
    # Create train and test templates for the target objects
    train_obj_object = object_variable("obj-with-color", target_object)
    obj_template = Phase1SituationTemplate(
        "colored-obj-object", salient_object_variables=[train_obj_object]
    )
    template = all_possible(
        obj_template, chooser=PHASE1_CHOOSER_FACTORY(), ontology=GAILA_PHASE_1_ONTOLOGY
    )

    train_curriculum = phase1_instances("all obj situations", situations=template)

    for (_, _, perceptual_representation) in train_curriculum.instances():
        # Original perception graph
        perception = graph_without_learner(
            PerceptionGraph.from_frame(perceptual_representation.frames[0])
        )

        # Original perception pattern
        whole_perception_pattern = PerceptionGraphPattern.from_graph(
            perception
        ).perception_graph_pattern
        # Create an altered perception graph we replace the color node
        altered_perception_digraph = perception.copy_as_digraph()
        nodes_to_remove = []
        edges = []
        different_nodes = []
        for node in perception.copy_as_digraph().nodes:
            # If we find a color node, we make it black
            if isinstance(node, tuple) and isinstance(node[0], RgbColorPerception):
                new_node = (RgbColorPerception(0, 0, 0), 42)
                # Get edge information
                for edge in perception.copy_as_digraph().edges(data=True):
                    if edge[0] == node:
                        edges.append((new_node, edge[1], edge[2]))
                    if edge[1] == node:
                        edges.append((edge[0], new_node, edge[2]))
                nodes_to_remove.append(node)
                different_nodes.append(new_node)

        # add new nodes
        for node in different_nodes:
            altered_perception_digraph.add_node(node)
        # add edge information
        for edge in edges:
            altered_perception_digraph.add_edge(edge[0], edge[1])
            for k, v in edge[2].items():
                altered_perception_digraph[edge[0]][edge[1]][k] = v
        # remove original node
        altered_perception_digraph.remove_nodes_from(nodes_to_remove)

        # Start the matching process
        matcher = whole_perception_pattern.matcher(
            PerceptionGraph(altered_perception_digraph), match_mode=MatchMode.NON_OBJECT
        )
        match_or_failure = matcher.first_match_or_failure_info()
        assert isinstance(match_or_failure, PatternMatching.MatchFailure)
        assert isinstance(match_or_failure.last_failed_pattern_node, IsColorNodePredicate)


def test_successfully_extending_partial_match():
    """
    Tests whether we can match a perception pattern against a perception graph
    when initializing the search from a partial match.
    """

    target_object = BOX
    # Create train and test templates for the target objects
    train_obj_object = object_variable("obj-with-color", target_object)

    obj_template = Phase1SituationTemplate(
        "colored-obj-object", salient_object_variables=[train_obj_object]
    )
    template = all_possible(
        obj_template, chooser=PHASE1_CHOOSER_FACTORY(), ontology=GAILA_PHASE_1_ONTOLOGY
    )

    train_curriculum = phase1_instances("all obj situations", situations=template)

    perceptual_representation = only(train_curriculum.instances())[2]

    # Original perception graph
    perception = PerceptionGraph.from_frame(perceptual_representation.frames[0])

    # Create a perception pattern for the whole thing
    # and also a perception pattern for a subset of the whole pattern
    whole_perception_pattern = PerceptionGraphPattern.from_graph(
        perception
    ).perception_graph_pattern

    partial_digraph = whole_perception_pattern.copy_as_digraph()
    partial_digraph.remove_nodes_from(
        [node for node in partial_digraph.nodes if isinstance(node, IsColorNodePredicate)]
    )
    partial_perception_pattern = PerceptionGraphPattern(partial_digraph)

    # get our initial match by matching the partial pattern
    matcher = partial_perception_pattern.matcher(
        perception, match_mode=MatchMode.NON_OBJECT
    )

    partial_match: PerceptionGraphPatternMatch = first(
        matcher.matches(use_lookahead_pruning=True)
    )
    partial_mapping = partial_match.pattern_node_to_matched_graph_node

    # Try to extend the partial mapping, to create a complete mapping
    matcher_2 = whole_perception_pattern.matcher(
        perception, match_mode=MatchMode.NON_OBJECT
    )
    complete_match: PerceptionGraphPatternMatch = first(
        matcher_2.matches(
            initial_partial_match=partial_mapping, use_lookahead_pruning=True
        ),
        None,
    )
    complete_mapping = complete_match.pattern_node_to_matched_graph_node
    assert len(complete_mapping) == len(perception.copy_as_digraph().nodes)
    assert len(complete_mapping) == len(whole_perception_pattern.copy_as_digraph().nodes)


def test_semantically_infeasible_partial_match():
    """
    Tests whether semantic feasibility works as intended
    """

    target_object = BOX
    # Create train and test templates for the target objects
    train_obj_object = object_variable("obj-with-color", target_object)
    obj_template = Phase1SituationTemplate(
        "colored-obj-object", salient_object_variables=[train_obj_object]
    )
    template = all_possible(
        obj_template, chooser=PHASE1_CHOOSER_FACTORY(), ontology=GAILA_PHASE_1_ONTOLOGY
    )

    train_curriculum = phase1_instances("all obj situations", situations=template)

    perceptual_representation = only(train_curriculum.instances())[2]
    # Original perception graph
    perception = graph_without_learner(
        PerceptionGraph.from_frame(perceptual_representation.frames[0])
    )
    whole_perception_pattern = PerceptionGraphPattern.from_graph(
        perception
    ).perception_graph_pattern

    # Create an altered perception graph we remove the color node
    altered_perception_digraph = perception.copy_as_digraph()
    nodes_to_remove = []
    edges = []
    different_nodes = []
    for node in perception.copy_as_digraph().nodes:
        # If we find a color node, we make it black
        if isinstance(node, tuple) and isinstance(node[0], RgbColorPerception):
            new_node = (RgbColorPerception(0, 0, 0), node[1])
            # Get edge information
            for edge in perception.copy_as_digraph().edges(data=True):
                if edge[0] == node:
                    edges.append((new_node, edge[1], edge[2]))
                if edge[1] == node:
                    edges.append((edge[0], new_node, edge[2]))
            nodes_to_remove.append(node)
            different_nodes.append(new_node)

    # remove original node
    altered_perception_digraph.remove_nodes_from(nodes_to_remove)

    # add new nodes
    for node in different_nodes:
        altered_perception_digraph.add_node(node)
    # add edge information
    for edge in edges:
        altered_perception_digraph.add_edge(edge[0], edge[1])
        for k, v in edge[2].items():
            altered_perception_digraph[edge[0]][edge[1]][k] = v

    altered_perception_pattern = PerceptionGraphPattern.from_graph(
        PerceptionGraph(altered_perception_digraph)
    ).perception_graph_pattern

    partial_digraph = altered_perception_pattern.copy_as_digraph()
    partial_digraph.remove_nodes_from(
        [node for node in partial_digraph.nodes if isinstance(node, IsColorNodePredicate)]
    )

    # Start the matching process, get a partial match
    matcher = whole_perception_pattern.matcher(perception, match_mode=MatchMode.OBJECT)
    partial_match: PerceptionGraphPatternMatch = first(
        matcher.matches(use_lookahead_pruning=True)
    )
    partial_mapping = partial_match.pattern_node_to_matched_graph_node

    # Try to extend the partial mapping, we expect a semantic infeasibility runtime error
    second_matcher = whole_perception_pattern.matcher(
        PerceptionGraph(altered_perception_digraph), match_mode=MatchMode.OBJECT
    )
    # The partial mapping (obtained from first matcher with original perception graph)
    # semantically doesn't match the one in the altered version (second matcher with altered graph)
    with pytest.raises(RuntimeError):
        first(
            second_matcher.matches(
                initial_partial_match=partial_mapping, use_lookahead_pruning=True
            ),
            None,
        )


def test_syntactically_infeasible_partial_match():
    """
    Tests whether syntactic feasibility works as intended
    """

    # We use a situation to generate the perceptual representation
    # for a box with color.
    target_object = BOX
    train_obj_object = object_variable("obj-with-color", target_object)
    obj_template = Phase1SituationTemplate(
        "colored-obj-object", salient_object_variables=[train_obj_object]
    )
    template = all_possible(
        obj_template, chooser=PHASE1_CHOOSER_FACTORY(), ontology=GAILA_PHASE_1_ONTOLOGY
    )

    train_curriculum = phase1_instances("all obj situations", situations=template)

    perceptual_representation = only(train_curriculum.instances())[2]

    # Original perception graph
    perception = graph_without_learner(
        PerceptionGraph.from_frame(perceptual_representation.frames[0])
    )
    whole_perception_pattern = PerceptionGraphPattern.from_graph(
        perception
    ).perception_graph_pattern

    # Create an altered perception graph we remove the color node
    altered_perception_digraph = perception.copy_as_digraph()
    nodes = []
    for node in perception.copy_as_digraph().nodes:
        # If we find a color node, we add an extra edge to it
        if isinstance(node, tuple) and isinstance(node[0], RgbColorPerception):
            nodes.append(node)

    # change edge information
    for node in nodes:
        random_node = r.choice(list(altered_perception_digraph.nodes))
        altered_perception_digraph.add_edge(node, random_node, label=PART_OF)
        random_node_2 = r.choice(list(altered_perception_digraph.nodes))
        altered_perception_digraph.add_edge(random_node_2, node, label=PART_OF)

    # Start the matching process, get a partial match
    first_matcher = whole_perception_pattern.matcher(
        perception, match_mode=MatchMode.OBJECT
    )
    partial_match: PerceptionGraphPatternMatch = first(
        first_matcher.matches(use_lookahead_pruning=True), None
    )
    partial_mapping = partial_match.pattern_node_to_matched_graph_node
    # Try to extend the partial mapping, we expect a semantic infeasibility runtime error
    second_matcher = whole_perception_pattern.matcher(
        PerceptionGraph(altered_perception_digraph), match_mode=MatchMode.OBJECT
    )
    # The partial mapping (obtained from first matcher with original perception graph)
    # syntactically doesn't match the one in the altered version (second matcher with altered graph)
    with pytest.raises(RuntimeError):
        first(
            second_matcher.matches(
                initial_partial_match=partial_mapping, use_lookahead_pruning=True
            ),
            None,
        )


def test_allowed_matches_with_bad_partial_match():
    """
    Tests whether PatternMarching's allowed_matches functionality works as intended when a bad
    partial match is specified.
    """
    target_object = BOX
    train_obj_object = object_variable("obj-with-color", target_object)
    obj_template = Phase1SituationTemplate(
        "colored-obj-object", salient_object_variables=[train_obj_object]
    )
    template = all_possible(
        obj_template, chooser=PHASE1_CHOOSER_FACTORY(), ontology=GAILA_PHASE_1_ONTOLOGY
    )

    train_curriculum = phase1_instances("all obj situations", situations=template)

    perceptual_representation = only(train_curriculum.instances())[2]
    perception = graph_without_learner(
        PerceptionGraph.from_frame(perceptual_representation.frames[0])
    )

    pattern1: PerceptionGraphPattern = PerceptionGraphPattern.from_graph(
        perception.subgraph_by_nodes(
            {
                cast(PerceptionGraphNode, node)
                for node in perception._graph.nodes  # pylint: disable=protected-access
                if getattr(node, "debug_handle", None) == "box_0"
            }
        )
    ).perception_graph_pattern

    pattern2: PerceptionGraphPattern = PerceptionGraphPattern.from_graph(
        perception.subgraph_by_nodes(
            {
                cast(PerceptionGraphNode, node)
                for node in perception._graph.nodes  # pylint: disable=protected-access
                if getattr(node, "debug_handle", None) in {"box_0", "the ground"}
            }
        )
    ).perception_graph_pattern

    pattern1_box: AnyObjectPerception = cast(
        AnyObjectPerception,
        only(
            node
            for node in pattern1._graph  # pylint: disable=protected-access
            if getattr(node, "debug_handle", None) == "box_0"
        ),
    )
    pattern2_box: AnyObjectPerception = cast(
        AnyObjectPerception,
        only(
            node
            for node in pattern2._graph  # pylint: disable=protected-access
            if getattr(node, "debug_handle", None) == "box_0"
        ),
    )
    pattern2_ground: AnyObjectPerception = cast(
        AnyObjectPerception,
        only(
            node
            for node in pattern2._graph  # pylint: disable=protected-access
            if getattr(node, "debug_handle", None) == "the ground"
        ),
    )

    matcher = PatternMatching(
        pattern=pattern1,
        graph_to_match_against=pattern2,
        matching_pattern_against_pattern=True,
        matching_objects=True,
        allowed_matches=immutablesetmultidict([(pattern1_box, pattern2_box)]),
    )
    with pytest.raises(RuntimeError):
        first(
            matcher.matches(
                initial_partial_match={pattern1_box: pattern2_ground},
                use_lookahead_pruning=True,
            ),
            None,
        )


def test_dynamic_perception_graph_instantiation():
    ball = ObjectPerception("ball", _BALL_SCHEMA.geon.copy())
    table = ObjectPerception("table", axes=_TABLE_SCHEMA.axes.copy())

    first_frame = DevelopmentalPrimitivePerceptionFrame(
        perceived_objects=[ball, table],
        relations=[
            above(ball, table),
            Relation(IN_REGION, ball, Region(table, distance=EXTERIOR_BUT_IN_CONTACT)),
            Relation(IN_REGION, table, Region(ball, distance=EXTERIOR_BUT_IN_CONTACT)),
        ],
    )

    second_frame = DevelopmentalPrimitivePerceptionFrame(
        perceived_objects=[ball, table],
        relations=[Relation(IN_REGION, ball, Region(table, distance=DISTAL))],
    )

    perception_graph = PerceptionGraph.from_dynamic_perceptual_representation(
        PerceptualRepresentation(frames=[first_frame, second_frame])
    )
    assert perception_graph.dynamic

    # Ensure we don't attempt to handle more than two frames yet.
    with pytest.raises(ValueError):
        PerceptionGraph.from_dynamic_perceptual_representation(
            PerceptualRepresentation(frames=[first_frame, second_frame, second_frame])
        )


def test_cannot_make_dynamic_copy_of_a_dynamic_graph():
    graph = PerceptionGraph(graph=DiGraph(), dynamic=True)
    with pytest.raises(RuntimeError):
        graph.copy_with_temporal_scopes([TemporalScope.BEFORE])


def test_copy_with_temporal_scopes_content():
    """
    Tests whether copy_with_temporal_scopes converts graphs to be dynamic as intended
    """

    # We use a situation to generate the perceptual representation
    # for a box with color.
    target_object = BOX
    train_obj_object = object_variable("obj-with-color", target_object)
    obj_template = Phase1SituationTemplate(
        "colored-obj-object", salient_object_variables=[train_obj_object]
    )
    template = all_possible(
        obj_template, chooser=PHASE1_CHOOSER_FACTORY(), ontology=GAILA_PHASE_1_ONTOLOGY
    )

    train_curriculum = phase1_instances("all obj situations", situations=template)

    perceptual_representation = only(train_curriculum.instances())[2]

    perception_graph = graph_without_learner(
        PerceptionGraph.from_frame(perceptual_representation.frames[0])
    )
    temporal_perception_graph = perception_graph.copy_with_temporal_scopes(
        temporal_scopes=[TemporalScope.AFTER]
    )
    for (source, target) in perception_graph.copy_as_digraph().edges():
        assert not isinstance(
            perception_graph.copy_as_digraph()[source][target]["label"],
            TemporallyScopedEdgeLabel,
        )
    for (source, target) in temporal_perception_graph.copy_as_digraph().edges():
        # Check type, and then the content
        label = temporal_perception_graph.copy_as_digraph()[source][target]["label"]
        assert isinstance(label, TemporallyScopedEdgeLabel)
        assert (
            label.attribute == perception_graph.copy_as_digraph()[source][target]["label"]
        )
        assert all(
            specifier in [TemporalScope.AFTER] for specifier in label.temporal_specifiers
        )


def test_perception_graph_post_init_edge_cases():
    target_object = BOX
    train_obj_object = object_variable("obj-with-color", target_object)
    obj_template = Phase1SituationTemplate(
        "colored-obj-object", salient_object_variables=[train_obj_object]
    )
    template = all_possible(
        obj_template, chooser=PHASE1_CHOOSER_FACTORY(), ontology=GAILA_PHASE_1_ONTOLOGY
    )
    train_curriculum = phase1_instances("all obj situations", situations=template)
    perceptual_representation = only(train_curriculum.instances())[2]
    perception_graph = graph_without_learner(
        PerceptionGraph.from_frame(perceptual_representation.frames[0])
    )
    temporal_perception_graph = perception_graph.copy_with_temporal_scopes(
        temporal_scopes=[TemporalScope.AFTER]
    )
    temporal_digraph = temporal_perception_graph.copy_as_digraph()
    # Test valid edge label
    # The only feasible test seems to be the instation, since creating a corrupt instance throws the same RuntimeError
    with pytest.raises(RuntimeError):
        TemporallyScopedEdgeLabel(None)

    # In a dynamic graph, all edge labels must be wrapped in TemporallyScopedEdgeLabel
    new_graph = DiGraph()
    for (source, target) in temporal_digraph.edges():
        new_graph.add_edge(source, target)
        new_graph[source][target]["label"] = None
    with pytest.raises(RuntimeError):
        PerceptionGraph(new_graph, dynamic=True)

    # TemporallyScopedEdgeLabels may not appear in a static graph
    new_graph = DiGraph()
    for (source, target) in temporal_digraph.edges():
        new_graph.add_edge(source, target)
        new_graph[source][target]["label"] = TemporallyScopedEdgeLabel(
            "attribute", [TemporalScope.AFTER]
        )
    with pytest.raises(RuntimeError):
        PerceptionGraph(new_graph)

    # Every edge in a PerceptionGraph must have a 'label
    new_graph = DiGraph()
    for (source, target) in temporal_digraph.edges():
        new_graph.add_edge(source, target)
    with pytest.raises(RuntimeError):
        PerceptionGraph(new_graph)


def test_matching_static_vs_dynamic_graphs():
    target_object = BOX
    train_obj_object = object_variable("obj-with-color", target_object)
    obj_template = Phase1SituationTemplate(
        "colored-obj-object", salient_object_variables=[train_obj_object]
    )
    template = all_possible(
        obj_template, chooser=PHASE1_CHOOSER_FACTORY(), ontology=GAILA_PHASE_1_ONTOLOGY
    )
    train_curriculum = phase1_instances("all obj situations", situations=template)
    perceptual_representation = only(train_curriculum.instances())[2]

    perception_graph = graph_without_learner(
        PerceptionGraph.from_frame(perceptual_representation.frames[0])
    )
    temporal_perception_graph = perception_graph.copy_with_temporal_scopes(
        temporal_scopes=[TemporalScope.AFTER]
    )

    perception_pattern = PerceptionGraphPattern.from_graph(
        perception_graph
    ).perception_graph_pattern

    temporal_perception_pattern = perception_pattern.copy_with_temporal_scopes(
        required_temporal_scopes=[TemporalScope.AFTER]
    )

    # Test runtime error for matching static pattern against dynamic graph and vice versa

    with pytest.raises(RuntimeError):
        perception_pattern.matcher(
            temporal_perception_graph, match_mode=MatchMode.NON_OBJECT
        )

    with pytest.raises(RuntimeError):
        temporal_perception_pattern.matcher(
            perception_graph, match_mode=MatchMode.NON_OBJECT
        )


def test_copy_with_temporal_scope_pattern_content():
    """
    Tests whether copy_with_temporal_scope converts patterns to be dynamic as intended
    """

    # We use a situation to generate the perceptual representation
    # for a box with color.
    target_object = BOX
    train_obj_object = object_variable("obj-with-color", target_object)
    obj_template = Phase1SituationTemplate(
        "colored-obj-object", salient_object_variables=[train_obj_object]
    )
    template = all_possible(
        obj_template, chooser=PHASE1_CHOOSER_FACTORY(), ontology=GAILA_PHASE_1_ONTOLOGY
    )

    train_curriculum = phase1_instances("all obj situations", situations=template)

    perceptual_representation = only(train_curriculum.instances())[2]

    perception_graph = graph_without_learner(
        PerceptionGraph.from_frame(perceptual_representation.frames[0])
    )

    perception_pattern = PerceptionGraphPattern.from_graph(
        perception_graph
    ).perception_graph_pattern

    temporal_perception_graph = perception_graph.copy_with_temporal_scopes(
        temporal_scopes=[TemporalScope.AFTER]
    )
    temporal_perception_pattern = perception_pattern.copy_with_temporal_scopes(
        required_temporal_scopes=TemporalScope.AFTER
    )

    # Exception while applying to dynamic pattern
    with pytest.raises(RuntimeError):
        temporal_perception_pattern.copy_with_temporal_scopes(
            required_temporal_scopes=TemporalScope.AFTER
        )

    for (source, target) in perception_pattern.copy_as_digraph().edges():
        assert not isinstance(
            perception_pattern.copy_as_digraph()[source][target]["predicate"],
            HoldsAtTemporalScopePredicate,
        )
    for (source, target) in temporal_perception_pattern.copy_as_digraph().edges():
        # Check type, and then the content
        predicate = temporal_perception_pattern.copy_as_digraph()[source][target][
            "predicate"
        ]
        # Test HoldsAtTemporalScope dot label, matches predicate
        assert isinstance(predicate.dot_label(), str)
        assert predicate.matches_predicate(
            HoldsAtTemporalScopePredicate(
                predicate.wrapped_edge_predicate, predicate.temporal_scopes
            )
        )
        assert not predicate.matches_predicate(
            HoldsAtTemporalScopePredicate(
                predicate.wrapped_edge_predicate, [TemporalScope.BEFORE]
            )
        )
        assert isinstance(predicate, HoldsAtTemporalScopePredicate)
        assert (
            predicate.wrapped_edge_predicate
            == perception_pattern.copy_as_digraph()[source][target]["predicate"]
        )
        assert len(predicate.temporal_scopes) == 1
        assert only(predicate.temporal_scopes) == TemporalScope.AFTER

    # Test normal matching behavior
    temporal_matcher = temporal_perception_pattern.matcher(
        temporal_perception_graph, match_mode=MatchMode.NON_OBJECT
    )
    first(temporal_matcher.matches(use_lookahead_pruning=True))

    # Test HoldsAtTemporalScopePredicate
    for (source, target) in perception_graph.copy_as_digraph().edges():
        label = "test edge label"
        edge_predicate = AnyEdgePredicate()
        temporal_predicate = HoldsAtTemporalScopePredicate(
            edge_predicate, [TemporalScope.AFTER]
        )

        temporal_edge_label = TemporallyScopedEdgeLabel(label, [TemporalScope.AFTER])
        assert temporal_predicate(source, temporal_edge_label, target)
        # Non temporal edge exception
        with pytest.raises(RuntimeError):
            temporal_predicate(source, label, target)
