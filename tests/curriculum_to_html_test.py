from typing import Tuple

from immutablecollections import ImmutableSet

from adam import curriculum_to_html
from adam.experiment.instance_group import ExplicitWithSituationInstanceGroup
from adam.language.dependency import LinearizedDependencyTree
from adam.ontology.phase1_ontology import GAILA_PHASE_1_ONTOLOGY, BALL, TABLE
from adam.perception import PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.situation import HighLevelSemanticsSituation, SituationObject


def test_simple_curriculum_html():
    instances = []
    ball = SituationObject(BALL)
    table = SituationObject(TABLE)
    group = ExplicitWithSituationInstanceGroup(
        name="Test Group",
        instances=Tuple[
            Tuple[
                HighLevelSemanticsSituation(
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    objects=ImmutableSet[ball, table],
                    # actions=ImmutableSet[],
                    # relations=ImmutableSet[]
                ),
                LinearizedDependencyTree(),
                PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame](),
            ]
        ],
    )
    instances.append(group)
    htmlExporter = curriculum_to_html.CurriculumToHtml()
    htmlExporter.generate(instances, "./", overwrite=True, title="Test Objects")
