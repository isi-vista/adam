from typing import Iterable
from immutablecollections import (
    immutableset,
    ImmutableSetMultiDict,
    immutablesetmultidict,
)
from networkx import DiGraph

from adam.ontology import (
    OntologyNode,
    REQUIRED_ONTOLOGY_NODES,
    PROPERTY,
    THING,
    CAN_FILL_TEMPLATE_SLOT,
)
from adam.ontology.structural_schema import ObjectStructuralSchema
from adam.ontology.ontology import Ontology
from adam.random_utils import RandomChooser
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    object_variable,
    all_possible,
)

_TESTING_ONTOLOGY_GRAPH = DiGraph()


def _subtype(sub: OntologyNode, _super: OntologyNode) -> None:
    _TESTING_ONTOLOGY_GRAPH.add_edge(sub, _super)


for required_ontology_node in REQUIRED_ONTOLOGY_NODES:
    _TESTING_ONTOLOGY_GRAPH.add_node(required_ontology_node)

# normally you would make toy_vehicle a node in the object ontology
# and have nodes inherit from it, but we want to test property selection,
# so here we do it as a property
_TOY_VEHICLE = OntologyNode("toy_vehicle")
_subtype(_TOY_VEHICLE, PROPERTY)

_TOY = OntologyNode("toy")
_subtype(_TOY, THING)
_BALL = OntologyNode("ball", [CAN_FILL_TEMPLATE_SLOT])
_subtype(_BALL, _TOY)
_TRUCK = OntologyNode(
    "toy_truck", inheritable_properties=[_TOY_VEHICLE, CAN_FILL_TEMPLATE_SLOT]
)
_subtype(_TRUCK, _TOY)
_CAR = OntologyNode(
    "toy_car", inheritable_properties=[_TOY_VEHICLE, CAN_FILL_TEMPLATE_SLOT]
)
_subtype(_CAR, _TOY)

_PERSON = OntologyNode("person")
_subtype(_PERSON, THING)
_MOM = OntologyNode("mom", [CAN_FILL_TEMPLATE_SLOT])
_subtype(_MOM, _PERSON)
_DAD = OntologyNode("dad", [CAN_FILL_TEMPLATE_SLOT])
_subtype(_DAD, _PERSON)


def _testing_schemata(
    nodes: Iterable[OntologyNode]
) -> ImmutableSetMultiDict[OntologyNode, ObjectStructuralSchema]:
    return immutablesetmultidict((node, ObjectStructuralSchema(node)) for node in nodes)


_TESTING_ONTOLOGY = Ontology(
    _TESTING_ONTOLOGY_GRAPH,
    structural_schemata=_testing_schemata([_MOM, _DAD, _BALL, _TRUCK, _CAR]),
)


def test_two_objects():
    two_object_template = Phase1SituationTemplate(
        object_variables=[
            object_variable("person", root_node=_PERSON),
            object_variable("toy_vehicle", required_properties=[_TOY_VEHICLE]),
        ]
    )

    reference_object_sets = {
        immutableset(["mom", "toy_truck"]),
        immutableset(["dad", "toy_truck"]),
        immutableset(["mom", "toy_car"]),
        immutableset(["dad", "toy_car"]),
    }

    generated_object_sets = set(
        immutableset(
            situation_object.ontology_node.handle
            for situation_object in situation.objects
        )
        for situation in all_possible(
            two_object_template,
            ontology=_TESTING_ONTOLOGY,
            chooser=RandomChooser.for_seed(0),
        )
    )

    assert generated_object_sets == reference_object_sets
