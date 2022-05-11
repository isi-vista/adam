from itertools import chain

from immutablecollections import immutableset

from adam.axes import Axes, directed, straight_up
from adam.geon import Geon, IRREGULAR, CONSTANT
from adam.ontology import (
    ACTION,
    OntologyNode,
    CAN_FILL_TEMPLATE_SLOT,
    META_PROPERTY,
)
from adam.ontology.action_description import (
    ActionDescription,
    ActionDescriptionFrame,
    ActionDescriptionVariable,
)
from adam.ontology.integrated_learner_experiement_ontology import (
    INTEGRATED_EXPERIMENT_STRUCTURAL_SCHEMATA,
)
from adam.ontology.ontology import Ontology
from adam.ontology.phase1_ontology import (
    _EAT_ACTION_DESCRIPTION,
    _FALL_ACTION_DESCRIPTION,
    _GIVE_ACTION_DESCRIPTION,
    _PUT_ACTION_DESCRIPTION,
    _TAKE_ACTION_DESCRIPTION,
    _WALK_ACTION_DESCRIPTION,
    AGENT,
    DRINK,
    EAT,
    FALL,
    GIVE,
    GO,
    GOAL,
    JUMP,
    PUT,
    RUN,
    SHOVE,
    SIT,
    SPIN,
    TAKE,
    THEME,
    THROW,
    WALK,
    _make_drink_description,
    _make_go_description,
    _make_jump_description,
    _make_push_descriptions,
    _make_sit_action_descriptions,
    _make_spin_descriptions,
    _make_throw_descriptions,
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
    CONTAINER,
    _PAPER_SCHEMA,
    PHASE_3_M4_CORE_CONCEPT,
    PHASE_3_M4_STRETCH_CONCEPT,
    PERSON,
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
    [PERSON_CAN_HAVE],
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

CUBE_BLOCK = OntologyNode(
    "cube_block",
    [CAN_FILL_TEMPLATE_SLOT, PERSON_CAN_HAVE, CUBIC],
    non_inheritable_properties=[
        PHASE_3_CONCEPT,
        PHASE_3_M4_CORE_CONCEPT,
        PHASE_3_M4_STRETCH_CONCEPT,
    ],
)
subtype(CUBE_BLOCK, BLOCK)
PYRAMID_BLOCK = OntologyNode(
    "pyramid_block",
    [CAN_FILL_TEMPLATE_SLOT, PERSON_CAN_HAVE, TRIANGULAR],
    non_inheritable_properties=[
        PHASE_3_CONCEPT,
        PHASE_3_M4_CORE_CONCEPT,
        PHASE_3_M4_STRETCH_CONCEPT,
    ],
)
subtype(PYRAMID_BLOCK, BLOCK)
SPHERICAL_BLOCK = OntologyNode(
    "sphere_block",
    [CAN_FILL_TEMPLATE_SLOT, PERSON_CAN_HAVE, SPHERICAL],
    non_inheritable_properties=[
        PHASE_3_CONCEPT,
        PHASE_3_M4_CORE_CONCEPT,
        PHASE_3_M4_STRETCH_CONCEPT,
    ],
)
subtype(SPHERICAL_BLOCK, BLOCK)

MUG = OntologyNode(
    "mug",
    [CAN_FILL_TEMPLATE_SLOT, PERSON_CAN_HAVE],
    non_inheritable_properties=[PHASE_3_CONCEPT],
)
subtype(MUG, CONTAINER)

NULL_NODE = OntologyNode("null")
subtype(NULL_NODE, META_PROPERTY)


PHASE_3_CURRICULUM_OBJECTS = immutableset(
    [
        APPLE,
        ORANGE,
        BANANA,
        BALL,
        BOOK,
        PYRAMID_BLOCK,
        SPHERICAL_BLOCK,
        CUBE_BLOCK,
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


PHASE_3_DECODE_OBJECTS = immutableset(chain(PHASE_3_CURRICULUM_OBJECTS, [PERSON]))

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
        (PYRAMID_BLOCK, _NULL_SCHEMA),
        (CUBE_BLOCK, _NULL_SCHEMA),
        (SPHERICAL_BLOCK, _NULL_SCHEMA),
        (TABLE, _TABLE_SCHEMA),
        (CHAIR, _CHAIR_SCHEMA),
        (SOFA, _NULL_SCHEMA),
        (BOX, _BOX_SCHEMA),
        (CUP, _CUP_SCHEMA),
        (MUG, _NULL_SCHEMA),
        (FLOOR, _NULL_SCHEMA),
        (WINDOW, _NULL_SCHEMA),
        (PAPER, _PAPER_SCHEMA),
        (DESK, _NULL_SCHEMA),
        (TOY_TRUCK, _NULL_SCHEMA),
        (TOY_SEDAN, _NULL_SCHEMA),
    ]
)


OPEN = OntologyNode("open")
subtype(OPEN, ACTION)
CLOSE = OntologyNode("close")
subtype(CLOSE, ACTION)
WRITING = OntologyNode("writing")
subtype(WRITING, ACTION)
STACK = OntologyNode("stack")
subtype(STACK, ACTION)
SHAKE = OntologyNode("shake")
subtype(SHAKE, ACTION)
SPILL = OntologyNode("spill")
subtype(SPILL, ACTION)


PHASE_3_CURRICULUM_ACTIONS = immutableset(
    [
        TAKE,
        SIT,
        RUN,
        EAT,
        PUT,
        OPEN,
        JUMP,
        DRINK,
        GO,
        CLOSE,
        FALL,
        WRITING,
        STACK,
        THROW,
        SHAKE,
        GIVE,
        SPIN,
        WALK,
        SPILL,
        SHOVE,
    ]
)

_NULL_ACTION_VARIABLE = ActionDescriptionVariable(NULL_NODE)
_NULL_ACTION_DESCRIPTION = ActionDescription(
    frame=ActionDescriptionFrame(
        {
            AGENT: _NULL_ACTION_VARIABLE,
            THEME: _NULL_ACTION_VARIABLE,
            GOAL: _NULL_ACTION_VARIABLE,
        }
    ),
    enduring_conditions=[],
    preconditions=[],
    postconditions=[],
    asserted_properties=[],
)

_ACTIONS_TO_DESCRIPTIONS = [
    (TAKE, _TAKE_ACTION_DESCRIPTION),
    (PUT, _PUT_ACTION_DESCRIPTION),
    (GIVE, _GIVE_ACTION_DESCRIPTION),
    (EAT, _EAT_ACTION_DESCRIPTION),
    (FALL, _FALL_ACTION_DESCRIPTION),
    (WALK, _WALK_ACTION_DESCRIPTION),
    (RUN, _WALK_ACTION_DESCRIPTION),
    (OPEN, _NULL_ACTION_DESCRIPTION),
    (CLOSE, _NULL_ACTION_DESCRIPTION),
    (WRITING, _NULL_ACTION_DESCRIPTION),
    (STACK, _NULL_ACTION_DESCRIPTION),
    (SHAKE, _NULL_ACTION_DESCRIPTION),
    (SPILL, _NULL_ACTION_DESCRIPTION),
    (SHOVE, list(_make_push_descriptions())[0][1]),
]

_ACTIONS_TO_DESCRIPTIONS.extend(_make_jump_description())
_ACTIONS_TO_DESCRIPTIONS.extend(_make_drink_description())
_ACTIONS_TO_DESCRIPTIONS.extend(_make_sit_action_descriptions())
_ACTIONS_TO_DESCRIPTIONS.extend(_make_spin_descriptions())
_ACTIONS_TO_DESCRIPTIONS.extend(_make_go_description())
_ACTIONS_TO_DESCRIPTIONS.extend(_make_throw_descriptions())


GAILA_PHASE_3_ONTOLOGY = Ontology(
    "gaila-phase-3",
    _ontology_graph,
    structural_schemata=GAILA_PHASE_3_STRUCUTRAL_SCHEMATA,
    action_to_description=_ACTIONS_TO_DESCRIPTIONS,
)
