from itertools import chain
from pathlib import Path

from adam.curriculum.phase1_curriculum import _standard_object
from adam.ontology.phase1_ontology import (
    BIRD,
    TABLE,
    TRUCK,
    _TRUCK_SCHEMA,
    above,
    bigger_than,
    on,
)
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    GAILA_PHASE_1_PERCEPTION_GENERATOR,
)
from adam.perception.perception_graph import PerceptionGraphPattern, to_perception_graph
from adam.random_utils import RandomChooser
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    color_variable,
    object_variable,
)
from adam_test_utils import all_possible_test


def test_match_truck_and_table():
    """
    Tests the `PerceptionGraphMatcher` can match simple objects.
    """
    # we create four situations:
    # a truck above or under a table with color red or blue
    color = color_variable("color")
    truck = object_variable(
        debug_handle="truck_0", root_node=TRUCK, added_properties=[color]
    )
    table = _standard_object("table_0", TABLE)

    truck_on_table_template = Phase1SituationTemplate(
        "truck-on-table",
        salient_object_variables=[truck, table],
        asserted_always_relations=[bigger_than(table, truck), on(truck, table)],
    )

    truck_under_table_template = Phase1SituationTemplate(
        "truck-under-table",
        salient_object_variables=[truck, table],
        asserted_always_relations=[bigger_than(table, truck), above(table, truck)],
    )

    # We test that a perceptual pattern for "truck" matches in all four cases.
    truck_pattern = PerceptionGraphPattern.from_schema(_TRUCK_SCHEMA)

    situations_with_truck = chain(
        all_possible_test(truck_on_table_template),
        all_possible_test(truck_under_table_template),
    )

    out_dir = Path("/Users/gabbard/tmp")

    for (idx, situation_with_truck) in enumerate(situations_with_truck):
        perception = GAILA_PHASE_1_PERCEPTION_GENERATOR.generate_perception(
            situation_with_truck, chooser=RandomChooser.for_seed(0)
        )
        perception_graph = to_perception_graph(perception.frames[0])
        perception_graph.render_to_file(f"truck {idx}", out_dir / f"truck-{idx}.pdf")
        truck_pattern.render_to_file(f"truck pattern", out_dir / "truck_pattern.pdf")
        assert any(truck_pattern.matcher(perception_graph).matches())

    # Now let's create the same situations, but substitute a bird for a truck.
    bird = object_variable(
        debug_handle="bird_0", root_node=BIRD, added_properties=[color]
    )
    bird_on_table_template = Phase1SituationTemplate(
        "bird-on-table",
        salient_object_variables=[bird, table],
        asserted_always_relations=[bigger_than(table, bird), on(bird, table)],
    )

    bird_under_table_template = Phase1SituationTemplate(
        "bird-under-table",
        salient_object_variables=[bird, table],
        asserted_always_relations=[bigger_than(table, bird), above(table, bird)],
    )

    situations_with_bird = chain(
        all_possible_test(bird_on_table_template),
        all_possible_test(bird_under_table_template),
    )

    # The pattern should now fail to match.
    for situation_with_bird in situations_with_bird:
        perception = GAILA_PHASE_1_PERCEPTION_GENERATOR.generate_perception(
            situation_with_bird
        )
        perception_graph = to_perception_graph(perception.frames[0])
        assert not any(truck_pattern.matcher(perception_graph).matches())
