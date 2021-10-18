from immutablecollections import immutableset

from adam.axes import Axes, directed, straight_up
from adam.geon import Geon, IRREGULAR, CONSTANT
from adam.ontology import (
    OntologyNode,
    CAN_FILL_TEMPLATE_SLOT,
    META_PROPERTY,
)
from adam.ontology.integrated_learner_experiement_ontology import (
    INTEGRATED_EXPERIMENT_STRUCTURAL_SCHEMATA,
)
from adam.ontology.ontology import Ontology
from adam.ontology.phase1_ontology import (
    subtype,
    _ontology_graph,
    FRUIT,
    FURNITURE,
    TOY,
    BALL,
    BOOK,
    TABLE,
    CHAIR,
    BOX,
    CUP,
    PAPER,
    PHASE_3_CONCEPT,
    _BALL_SCHEMA,
    _BOOK_SCHEMA,
    _TABLE_SCHEMA,
    _CHAIR_SCHEMA,
    _BOX_SCHEMA,
    _CUP_SCHEMA,
    PERSON_CAN_HAVE,
    MATERIAL,
    SHAPE_PROPERTY_DESCRIPTION,
)
from adam.ontology.structural_schema import ObjectStructuralSchema

# Phase 3 Properties
# Property Types used to express increased control over simulation generation
# Material Properties
PLASTIC = OntologyNode("plastic", [CAN_FILL_TEMPLATE_SLOT])
subtype(PLASTIC, MATERIAL)
WOOD = OntologyNode("wood", [CAN_FILL_TEMPLATE_SLOT])
subtype(WOOD, MATERIAL)

# Shape Properties
TRIANGULAR = OntologyNode("triangular", [CAN_FILL_TEMPLATE_SLOT])
subtype(TRIANGULAR, SHAPE_PROPERTY_DESCRIPTION)
CUBIC = OntologyNode("cubic", [CAN_FILL_TEMPLATE_SLOT])
subtype(CUBIC, SHAPE_PROPERTY_DESCRIPTION)
SPHERICAL = OntologyNode("spherical", [CAN_FILL_TEMPLATE_SLOT])
subtype(SPHERICAL, SHAPE_PROPERTY_DESCRIPTION)

# Phase 3 Curriculum Objects
# Base Property Objects
BLOCK = OntologyNode(
    "block",
    [CAN_FILL_TEMPLATE_SLOT, PERSON_CAN_HAVE],
    non_inheritable_properties=[PHASE_3_CONCEPT],
)
subtype(BLOCK, TOY)

# Sub-Object Components of Curriculum Objects -- Needed for ADAM's symbolic representation if desired

# Curriculum Objects
APPLE = OntologyNode(
    "apple",
    [CAN_FILL_TEMPLATE_SLOT, PERSON_CAN_HAVE],
    non_inheritable_properties=[PHASE_3_CONCEPT],
)
subtype(APPLE, FRUIT)
ORANGE = OntologyNode(
    "orange",
    [CAN_FILL_TEMPLATE_SLOT, PERSON_CAN_HAVE],
    non_inheritable_properties=[PHASE_3_CONCEPT],
)
subtype(ORANGE, FRUIT)
BANANA = OntologyNode(
    "banana",
    [CAN_FILL_TEMPLATE_SLOT, PERSON_CAN_HAVE],
    non_inheritable_properties=[PHASE_3_CONCEPT],
)
subtype(BANANA, FRUIT)

SOFA = OntologyNode(
    "sofa", [CAN_FILL_TEMPLATE_SLOT], non_inheritable_properties=[PHASE_3_CONCEPT]
)
subtype(SOFA, FURNITURE)
DESK = OntologyNode(
    "desk", [CAN_FILL_TEMPLATE_SLOT], non_inheritable_properties=[PHASE_3_CONCEPT]
)
subtype(DESK, FURNITURE)

FLOOR = OntologyNode(
    "floor", [CAN_FILL_TEMPLATE_SLOT], non_inheritable_properties=[PHASE_3_CONCEPT]
)
subtype(FLOOR, FURNITURE)
WINDOW = OntologyNode(
    "window", [CAN_FILL_TEMPLATE_SLOT], non_inheritable_properties=[PHASE_3_CONCEPT]
)
subtype(WINDOW, FURNITURE)
TOY_TRUCK = OntologyNode(
    "toy_truck",
    [CAN_FILL_TEMPLATE_SLOT, PERSON_CAN_HAVE],
    non_inheritable_properties=[PHASE_3_CONCEPT],
)
subtype(TOY_TRUCK, TOY)
TOY_SEDAN = OntologyNode(
    "toy_sedan",
    [CAN_FILL_TEMPLATE_SLOT, PERSON_CAN_HAVE],
    non_inheritable_properties=[PHASE_3_CONCEPT],
)
subtype(TOY_SEDAN, TOY)

NULL_NODE = OntologyNode("null")
subtype(NULL_NODE, META_PROPERTY)

PHASE_3_CURRICULUM_OBJECTS = immutableset(
    [
        APPLE,
        ORANGE,
        BANANA,
        BALL,
        BOOK,
        BLOCK,  # This is intended to have a shape modifier (Sphere, Cube, Triangular)
        TABLE,
        CHAIR,
        SOFA,
        BOX,
        CUP,
        FLOOR,
        WINDOW,
        PAPER,
        DESK,
        TOY_TRUCK,
        TOY_SEDAN,
    ]
)

# TODO: Implement P1/P2 Structural Schema for P3 Objects
# https://github.com/isi-vista/adam/issues/1048
_NULL_SCHEMA = ObjectStructuralSchema(
    ontology_node=NULL_NODE,
    geon=Geon(
        cross_section=IRREGULAR,
        cross_section_size=CONSTANT,
        axes=Axes(
            primary_axis=straight_up("bottom-to-top"),
            orienting_axes=[directed("front-to-back"), directed("side-to-side")],
        ),
    ),
)

GAILA_PHASE_3_STRUCUTRAL_SCHEMATA = [
    value for value in INTEGRATED_EXPERIMENT_STRUCTURAL_SCHEMATA
]
GAILA_PHASE_3_STRUCUTRAL_SCHEMATA.extend(
    [
        (APPLE, _NULL_SCHEMA),
        (ORANGE, _NULL_SCHEMA),
        (BANANA, _NULL_SCHEMA),
        (BALL, _BALL_SCHEMA),
        (BOOK, _BOOK_SCHEMA),
        (BLOCK, _NULL_SCHEMA),
        (TABLE, _TABLE_SCHEMA),
        (CHAIR, _CHAIR_SCHEMA),
        (SOFA, _NULL_SCHEMA),
        (BOX, _BOX_SCHEMA),
        (CUP, _CUP_SCHEMA),
        (FLOOR, _NULL_SCHEMA),
        (WINDOW, _NULL_SCHEMA),
        (PAPER, _NULL_SCHEMA),
        (DESK, _NULL_SCHEMA),
        (TOY_TRUCK, _NULL_SCHEMA),
        (TOY_SEDAN, _NULL_SCHEMA),
    ]
)

GAILA_PHASE_3_ONTOLOGY = Ontology(
    "gaila-phase-3",
    _ontology_graph,
    structural_schemata=GAILA_PHASE_3_STRUCUTRAL_SCHEMATA,
)
