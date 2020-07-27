from adam.geon import CONSTANT, LARGE_TO_SMALL, SMALL_TO_LARGE_TO_SMALL
from adam.ontology import CAN_FILL_TEMPLATE_SLOT, OntologyNode
from adam.ontology.ontology import Ontology
from adam.ontology.phase1_ontology import (
    BIGGER_THAN,
    BLUE,
    CAN_BE_SAT_ON_BY_PEOPLE,
    CAN_HAVE_THINGS_RESTING_ON_THEM,
    CHAIR,
    DARK_BROWN,
    GAILA_PHASE_1_ONTOLOGY,
    GAILA_PHASE_1_SIZE_GRADES,
    GREEN,
    HOLLOW,
    INANIMATE_OBJECT,
    LIGHT_BROWN,
    PERSON_CAN_HAVE,
    RED,
    SMALLER_THAN,
    TRANSPARENT,
    _ACTIONS_TO_DESCRIPTIONS,
    _CHAIR_LEGS,
    _CHAIR_SCHEMA_BACK,
    _CHAIR_SCHEMA_BACK_LEFT_LEG,
    _CHAIR_SCHEMA_BACK_RIGHT_LEG,
    _CHAIR_SCHEMA_FRONT_LEFT_LEG,
    _CHAIR_SCHEMA_FRONT_RIGHT_LEG,
    _CHAIR_SCHEMA_SEAT,
    _CHAIR_SCHEMA_SQUARE_SEAT,
    _CHAIR_THREE_LEGS,
    _make_cup_schema,
    _ontology_graph,
    above,
    contacts,
    subtype,
)
from adam.ontology.phase1_size_relationships import build_size_relationships
from adam.ontology.structural_schema import ObjectStructuralSchema
from adam.relation import flatten_relations

CHAIR_2 = OntologyNode(
    "chair-2",
    [
        CAN_FILL_TEMPLATE_SLOT,
        CAN_HAVE_THINGS_RESTING_ON_THEM,
        CAN_BE_SAT_ON_BY_PEOPLE,
        LIGHT_BROWN,
        DARK_BROWN,
    ],
)
subtype(CHAIR_2, INANIMATE_OBJECT)
CHAIR_3 = OntologyNode(
    "chair-3",
    [
        CAN_FILL_TEMPLATE_SLOT,
        CAN_HAVE_THINGS_RESTING_ON_THEM,
        CAN_BE_SAT_ON_BY_PEOPLE,
        LIGHT_BROWN,
        DARK_BROWN,
    ],
)
subtype(CHAIR_3, INANIMATE_OBJECT)
CHAIR_4 = OntologyNode(
    "chair-4",
    [
        CAN_FILL_TEMPLATE_SLOT,
        CAN_HAVE_THINGS_RESTING_ON_THEM,
        CAN_BE_SAT_ON_BY_PEOPLE,
        LIGHT_BROWN,
        DARK_BROWN,
    ],
)
subtype(CHAIR_4, INANIMATE_OBJECT)
CHAIR_5 = OntologyNode(
    "chair-5",
    [
        CAN_FILL_TEMPLATE_SLOT,
        CAN_HAVE_THINGS_RESTING_ON_THEM,
        CAN_BE_SAT_ON_BY_PEOPLE,
        LIGHT_BROWN,
        DARK_BROWN,
    ],
)
subtype(CHAIR_5, INANIMATE_OBJECT)

CUP_2 = OntologyNode(
    "cup-2",
    [HOLLOW, CAN_FILL_TEMPLATE_SLOT, PERSON_CAN_HAVE, RED, BLUE, GREEN, TRANSPARENT],
)
subtype(CUP_2, INANIMATE_OBJECT)
CUP_3 = OntologyNode(
    "cup-3",
    [HOLLOW, CAN_FILL_TEMPLATE_SLOT, PERSON_CAN_HAVE, RED, BLUE, GREEN, TRANSPARENT],
)
subtype(CUP_3, INANIMATE_OBJECT)
CUP_4 = OntologyNode(
    "cup-4",
    [HOLLOW, CAN_FILL_TEMPLATE_SLOT, PERSON_CAN_HAVE, RED, BLUE, GREEN, TRANSPARENT],
)
subtype(CUP_4, INANIMATE_OBJECT)

_CUP_2_SCHEMA = _make_cup_schema(cross_section_size=CONSTANT)
_CUP_3_SCHEMA = _make_cup_schema(cross_section_size=LARGE_TO_SMALL)
_CUP_4_SCHEMA = _make_cup_schema(cross_section_size=SMALL_TO_LARGE_TO_SMALL)


# 2 with square seat
_CHAIR_2_SCHEMA = ObjectStructuralSchema(
    CHAIR,
    sub_objects=[
        _CHAIR_SCHEMA_BACK,
        _CHAIR_SCHEMA_SQUARE_SEAT,
        _CHAIR_SCHEMA_FRONT_LEFT_LEG,
        _CHAIR_SCHEMA_FRONT_RIGHT_LEG,
        _CHAIR_SCHEMA_BACK_LEFT_LEG,
        _CHAIR_SCHEMA_BACK_RIGHT_LEG,
    ],
    sub_object_relations=flatten_relations(
        [
            contacts(_CHAIR_LEGS, _CHAIR_SCHEMA_SQUARE_SEAT),
            above(_CHAIR_SCHEMA_SQUARE_SEAT, _CHAIR_LEGS),
            contacts(_CHAIR_SCHEMA_BACK, _CHAIR_SCHEMA_SQUARE_SEAT),
            above(_CHAIR_SCHEMA_BACK, _CHAIR_SCHEMA_SQUARE_SEAT),
        ]
    ),
    axes=_CHAIR_SCHEMA_BACK.schema.axes.copy(),
)
# 3 with no back
_CHAIR_3_SCHEMA = ObjectStructuralSchema(
    CHAIR,
    sub_objects=[
        _CHAIR_SCHEMA_SEAT,
        _CHAIR_SCHEMA_FRONT_LEFT_LEG,
        _CHAIR_SCHEMA_FRONT_RIGHT_LEG,
        _CHAIR_SCHEMA_BACK_LEFT_LEG,
        _CHAIR_SCHEMA_BACK_RIGHT_LEG,
    ],
    sub_object_relations=flatten_relations(
        [
            contacts(_CHAIR_LEGS, _CHAIR_SCHEMA_SEAT),
            above(_CHAIR_SCHEMA_SEAT, _CHAIR_LEGS),
        ]
    ),
    axes=_CHAIR_SCHEMA_BACK.schema.axes.copy(),
)
# 4 with square seat and no back
_CHAIR_4_SCHEMA = ObjectStructuralSchema(
    CHAIR,
    sub_objects=[
        _CHAIR_SCHEMA_SQUARE_SEAT,
        _CHAIR_SCHEMA_FRONT_LEFT_LEG,
        _CHAIR_SCHEMA_FRONT_RIGHT_LEG,
        _CHAIR_SCHEMA_BACK_LEFT_LEG,
        _CHAIR_SCHEMA_BACK_RIGHT_LEG,
    ],
    sub_object_relations=flatten_relations(
        [
            contacts(_CHAIR_LEGS, _CHAIR_SCHEMA_SQUARE_SEAT),
            above(_CHAIR_SCHEMA_SQUARE_SEAT, _CHAIR_LEGS),
        ]
    ),
    axes=_CHAIR_SCHEMA_BACK.schema.axes.copy(),
)
# 5 with three legs
_CHAIR_5_SCHEMA = ObjectStructuralSchema(
    CHAIR,
    sub_objects=[
        _CHAIR_SCHEMA_BACK,
        _CHAIR_SCHEMA_SEAT,
        _CHAIR_SCHEMA_FRONT_LEFT_LEG,
        _CHAIR_SCHEMA_FRONT_RIGHT_LEG,
        _CHAIR_SCHEMA_BACK_LEFT_LEG,
    ],
    sub_object_relations=flatten_relations(
        [
            contacts(_CHAIR_THREE_LEGS, _CHAIR_SCHEMA_SEAT),
            above(_CHAIR_SCHEMA_SEAT, _CHAIR_THREE_LEGS),
            contacts(_CHAIR_SCHEMA_BACK, _CHAIR_SCHEMA_SEAT),
            above(_CHAIR_SCHEMA_BACK, _CHAIR_SCHEMA_SEAT),
        ]
    ),
    axes=_CHAIR_SCHEMA_BACK.schema.axes.copy(),
)


GAILA_PHASE_2_ONTOLOGY = Ontology(
    "gaila-phase-2",
    _ontology_graph,
    structural_schemata=[
        schemata
        for schemata in GAILA_PHASE_1_ONTOLOGY._structural_schemata.items()  # pylint: disable=protected-access
    ]
    + [
        (CHAIR_2, _CHAIR_2_SCHEMA),
        (CHAIR_3, _CHAIR_3_SCHEMA),
        (CHAIR_4, _CHAIR_4_SCHEMA),
        (CHAIR_5, _CHAIR_5_SCHEMA),
        (CUP_2, _CUP_2_SCHEMA),
        (CUP_3, _CUP_3_SCHEMA),
        (CUP_4, _CUP_4_SCHEMA),
    ],
    action_to_description=_ACTIONS_TO_DESCRIPTIONS,
    relations=build_size_relationships(
        GAILA_PHASE_1_SIZE_GRADES, relation_type=BIGGER_THAN, opposite_type=SMALLER_THAN
    ),
)
