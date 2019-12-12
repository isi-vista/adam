from itertools import chain

from adam_test_utils import all_possible_test
from more_itertools import first

from adam.curriculum.curriculum_utils import standard_object, PHASE1_CHOOSER, phase1_instances
from adam.learner.subset import graph_without_learner
from adam.ontology import OntologyNode
from adam.ontology.phase1_ontology import (
    BIRD,
    GAILA_PHASE_1_ONTOLOGY,
    GROUND,
    HOUSE,
    INANIMATE_OBJECT,
    IS_BODY_PART,
    LIQUID,
    TABLE,
    _HOUSE_SCHEMA,
    above,
    bigger_than,
    on,
    BOX)
from adam.ontology.structural_schema import ObjectStructuralSchema
from adam.perception.developmental_primitive_perception import RgbColorPerception
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    GAILA_PHASE_1_PERCEPTION_GENERATOR,
)
from adam.perception.perception_graph import PerceptionGraph, PerceptionGraphPattern, PerceptionGraphProtocol, \
    PatternMatching, IsColorNodePredicate
from adam.random_utils import RandomChooser
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    color_variable,
    object_variable,
    all_possible)


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
        matcher = object_to_match_pattern.matcher(perception_graph)
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
        if any(object_to_match_pattern.matcher(perception_graph).matches()):
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
        "colored-obj-object", salient_object_variables=[train_obj_object],
    )
    template = all_possible(
                    obj_template,
                    chooser=PHASE1_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                )

    train_curriculum = phase1_instances(
        "all obj situations", situations=template
    )

    for (_, _, perceptual_representation) in train_curriculum.instances():
        perception = graph_without_learner(
            PerceptionGraph.from_frame(
                perceptual_representation.frames[0]
            ).copy_as_digraph()
        )
        whole_perception_pattern = PerceptionGraphPattern.from_graph(
            perception.copy_as_digraph()
        ).perception_graph_pattern
        print('OUT:', perception.copy_as_digraph().nodes)
        copy_of_perception_digraph = perception.copy_as_digraph()
        nodes_to_remove = []
        edges = []
        different_nodes = []
        for node in perception.copy_as_digraph().nodes:
            if isinstance(node, RgbColorPerception):
                new_node = RgbColorPerception(0, 0, 0)
                for edge in perception.copy_as_digraph().edges:
                    if len(edge) > 2:
                        if edge[0] == node:
                            edges.append((new_node, edge[1], edge[2]))
                        if edge[1] == node:
                            edges.append((edge[0], new_node, edge[2]))
                    else:
                        if edge[0] == node:
                            edges.append((new_node, edge[1]))
                        if edge[1] == node:
                            edges.append((edge[0], new_node))
                nodes_to_remove.append(node)
                different_nodes.append(new_node)
        for edge in edges:
            copy_of_perception_digraph.add_edge(*edge)
        copy_of_perception_digraph.remove_nodes_from(nodes_to_remove)

        matcher = whole_perception_pattern.matcher(
            PerceptionGraph(copy_of_perception_digraph)
        )
        matches = matcher.matches()
        assert len(matches) > 0
        for match in matcher.matches():
            assert isinstance(match, PatternMatching.MatchFailure) and \
                   isinstance(match.last_failed_pattern_node, IsColorNodePredicate)
