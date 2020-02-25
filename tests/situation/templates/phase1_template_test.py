from typing import Iterable

from immutablecollections import (
    ImmutableSetMultiDict,
    immutableset,
    immutablesetmultidict,
)

from adam.ontology import (
    CAN_FILL_TEMPLATE_SLOT,
    OntologyNode,
    PROPERTY,
    REQUIRED_ONTOLOGY_NODES,
    THING,
    minimal_ontology_graph,
    IS_ADDRESSEE,
)
from adam.ontology.ontology import Ontology
from adam.ontology.phase1_ontology import (
    RECOGNIZED_PARTICULAR_PROPERTY,
    LEARNER,
    BALL,
    near,
    GAILA_PHASE_1_ONTOLOGY,
    MOM,
    GROUND,
    BOX,
    ROLL,
    AGENT,
    ROLL_SURFACE_AUXILIARY,
    on,
    far,
)
from adam.ontology.structural_schema import ObjectStructuralSchema
from adam.random_utils import RandomChooser
from adam.relation import flatten_relations
from adam.situation import Action
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    all_possible,
    object_variable,
    sampled,
)
from adam.axes import WORLD_AXES

_TESTING_ONTOLOGY_GRAPH = minimal_ontology_graph()
_TESTING_ONTOLOGY_GRAPH.add_node(RECOGNIZED_PARTICULAR_PROPERTY)


def _subtype(sub: OntologyNode, _super: OntologyNode) -> None:
    _TESTING_ONTOLOGY_GRAPH.add_edge(sub, _super)


for required_ontology_node in REQUIRED_ONTOLOGY_NODES:
    _TESTING_ONTOLOGY_GRAPH.add_node(required_ontology_node)

# normally you would make toy_vehicle a node in the object ontology
# and have nodes inherit from it, but we want to test property selection,
# so here we do it as a property
_TOY_VEHICLE = OntologyNode("toy_vehicle")
_subtype(_TOY_VEHICLE, PROPERTY)
_subtype(IS_ADDRESSEE, PROPERTY)

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
_LEARNER = OntologyNode("learner", [CAN_FILL_TEMPLATE_SLOT])
_subtype(_LEARNER, _PERSON)


def _testing_schemata(
    nodes: Iterable[OntologyNode]
) -> ImmutableSetMultiDict[OntologyNode, ObjectStructuralSchema]:
    return immutablesetmultidict(
        (node, ObjectStructuralSchema(node, axes=WORLD_AXES)) for node in nodes
    )


_TESTING_ONTOLOGY = Ontology(
    "test-ontology",
    _TESTING_ONTOLOGY_GRAPH,
    structural_schemata=_testing_schemata([_MOM, _DAD, _BALL, _TRUCK, _CAR, _LEARNER]),
)


def test_two_objects():
    two_object_template = Phase1SituationTemplate(
        "two-objects",
        salient_object_variables=[
            object_variable("person", root_node=_PERSON),
            object_variable("toy_vehicle", required_properties=[_TOY_VEHICLE]),
        ],
    )

    reference_object_sets = {
        immutableset(["mom", "toy_truck"]),
        immutableset(["dad", "toy_truck"]),
        immutableset(["learner", "toy_truck"]),
        immutableset(["mom", "toy_car"]),
        immutableset(["dad", "toy_car"]),
        immutableset(["learner", "toy_car"]),
    }

    generated_object_sets = set(
        immutableset(
            situation_object.ontology_node.handle
            for situation_object in situation.salient_objects
        )
        for situation in all_possible(
            two_object_template,
            ontology=_TESTING_ONTOLOGY,
            chooser=RandomChooser.for_seed(0),
            default_addressee_node=_LEARNER,
        )
    )

    assert generated_object_sets == reference_object_sets


def test_learner_as_default_addressee():
    learner = object_variable("learner", root_node=LEARNER)
    ball = object_variable("ball", root_node=BALL)
    template_with_learner = Phase1SituationTemplate(
        "template with learner",
        salient_object_variables=[learner, ball],
        asserted_always_relations=[near(learner, ball)],
    )

    template_with_out_learner = Phase1SituationTemplate(
        "template with out learner",
        salient_object_variables=[object_variable("ball", root_node=BALL)],
    )

    template_with_addressee = Phase1SituationTemplate(
        "template with addressee",
        salient_object_variables=[
            object_variable("mom", root_node=MOM, added_properties=[IS_ADDRESSEE])
        ],
    )

    situation_with_learner = sampled(
        template_with_learner,
        ontology=GAILA_PHASE_1_ONTOLOGY,
        chooser=RandomChooser.for_seed(0),
        max_to_sample=1,
    )

    situation_with_out_learner = sampled(
        template_with_out_learner,
        ontology=GAILA_PHASE_1_ONTOLOGY,
        chooser=RandomChooser.for_seed(0),
        max_to_sample=1,
    )

    situation_with_addressee = sampled(
        template_with_addressee,
        ontology=GAILA_PHASE_1_ONTOLOGY,
        chooser=RandomChooser.for_seed(0),
        max_to_sample=1,
    )

    for object_ in situation_with_learner[0].all_objects:
        if object_.ontology_node == LEARNER:
            assert IS_ADDRESSEE in object_.properties
            break

    assert situation_with_learner[0].axis_info
    assert situation_with_learner[0].axis_info.addressee

    assert len(situation_with_out_learner[0].all_objects) == 2

    for object_ in situation_with_out_learner[0].all_objects:
        if object_.ontology_node == LEARNER:
            assert IS_ADDRESSEE in object_.properties
            break

    assert situation_with_out_learner[0].axis_info
    assert situation_with_out_learner[0].axis_info.addressee

    for object_ in situation_with_addressee[0].all_objects:
        if object_.ontology_node == LEARNER:
            assert False

    assert situation_with_addressee[0].axis_info
    assert situation_with_addressee[0].axis_info.addressee


def test_before_after_relations_asserted():
    ball = object_variable("ball", root_node=BALL)
    box = object_variable("box", root_node=BOX)
    ground = object_variable("ground", root_node=GROUND)

    template_action = Phase1SituationTemplate(
        "Before/After Relation",
        salient_object_variables=[ball, box],
        background_object_variables=[ground],
        actions=[
            Action(
                ROLL,
                argument_roles_to_fillers=[(AGENT, ball)],
                auxiliary_variable_bindings=[(ROLL_SURFACE_AUXILIARY, ground)],
            )
        ],
        before_action_relations=flatten_relations([on(ball, box)]),
        after_action_relations=flatten_relations([far(ball, box)]),
    )

    situation_with_relations = sampled(
        template_action,
        ontology=GAILA_PHASE_1_ONTOLOGY,
        chooser=RandomChooser.for_seed(0),
        max_to_sample=1,
    )

    assert situation_with_relations[0].before_action_relations
    assert situation_with_relations[0].after_action_relations
