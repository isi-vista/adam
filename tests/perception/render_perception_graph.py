from adam.curriculum.curriculum_utils import PHASE1_CHOOSER
from adam.ontology.phase1_ontology import GAILA_PHASE_1_ONTOLOGY, BALL
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    GAILA_PHASE_1_PERCEPTION_GENERATOR,
)
from adam.perception.perception_graph import PerceptionGraph
from adam.random_utils import RandomChooser
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    all_possible,
    object_variable,
)


def render_perception_graph():
    object_to_match = object_variable(debug_handle="ball_handle", root_node=BALL)

    object_on_ground_template = Phase1SituationTemplate(
        "object_to_match-on-table", salient_object_variables=[object_to_match]
    )
    for i, sit in enumerate(
        all_possible(
            object_on_ground_template,
            ontology=GAILA_PHASE_1_ONTOLOGY,
            chooser=PHASE1_CHOOSER,
        )
    ):
        perception = GAILA_PHASE_1_PERCEPTION_GENERATOR.generate_perception(
            sit, chooser=RandomChooser.for_seed(0)
        )
        perception_graph = PerceptionGraph.from_frame(perception.frames[0])
        perception_graph.render_to_file(
            graph_name="name", output_file=f"../outputs/out_{i}"
        )


if __name__ == "__main__":
    render_perception_graph()
