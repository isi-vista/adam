from typing import Tuple

from adam.geon import LARGE_TO_SMALL, CONSTANT, SMALL_TO_LARGE_TO_SMALL
from adam.ontology import OntologyNode, CAN_FILL_TEMPLATE_SLOT
from adam.ontology.ontology import Ontology
from adam.ontology.phase1_ontology import (
    MUCH_BIGGER_THAN,
    MUCH_SMALLER_THAN,
    _make_cup_schema,
    _CHAIR_SCHEMA_BACK,
    _CHAIR_SCHEMA_SQUARE_SEAT,
    _CHAIR_SCHEMA_FRONT_LEFT_LEG,
    _CHAIR_SCHEMA_FRONT_RIGHT_LEG,
    _CHAIR_SCHEMA_BACK_LEFT_LEG,
    _CHAIR_SCHEMA_BACK_RIGHT_LEG,
    _CHAIR_LEGS,
    _CHAIR_SCHEMA_SEAT,
    _CHAIR_THREE_LEGS,
    _ontology_graph,
    _ACTIONS_TO_DESCRIPTIONS,
    CAN_HAVE_THINGS_RESTING_ON_THEM,
    CAN_BE_SAT_ON_BY_PEOPLE,
    LIGHT_BROWN,
    DARK_BROWN,
    INANIMATE_OBJECT,
    subtype,
    HOLLOW,
    PERSON_CAN_HAVE,
    RED,
    BLUE,
    GREEN,
    TRANSPARENT,
    CHAIR,
    contacts,
    above,
    GAILA_PHASE_1_ONTOLOGY,
    SMALLER_THAN,
    BIGGER_THAN,
    PERCEIVABLE_PROPERTY,
    _TIRE,
    BOX,
    DOG,
    _WING,
    _TAIL,
    MILK,
    WATER,
    JUICE,
    HAT,
    CUP,
    COOKIE,
    BOOK,
    BIRD,
    BALL,
    _FOOT,
    _LEG_SEGMENT,
    _ARM_SEGMENT,
    HEAD,
    HAND,
    WATERMELON,
    _INANIMATE_LEG,
    _ANIMAL_LEG,
    _ARM,
    _CHAIR_SEAT,
    _CHAIR_BACK,
    _TORSO,
    _BODY,
    BABY,
    PERSON,
    DAD,
    MOM,
    _TABLETOP,
    DOOR,
    TABLE,
    _TRUCK_CAB,
    _TRAILER,
    _FLATBED,
    TRUCK,
    CAR,
    _WALL,
    _ROOF,
    HOUSE,
)
from adam.ontology.phase1_size_relationships import build_size_relationships
from adam.ontology.structural_schema import ObjectStructuralSchema
from adam.relation import flatten_relations

TWO = OntologyNode("two")
subtype(TWO, PERCEIVABLE_PROPERTY)
MANY = OntologyNode("many")
subtype(TWO, PERCEIVABLE_PROPERTY)
HAS_COUNT = OntologyNode("has-count")

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

GAILA_PHASE_2_SIZE_GRADES: Tuple[Tuple[OntologyNode, ...], ...] = (
    (HOUSE,),
    (_ROOF, _WALL),
    (CAR, TRUCK),
    (_TRAILER, _FLATBED),
    (_TRUCK_CAB,),
    (TABLE, DOOR),
    (_TABLETOP,),
    (MOM, DAD, PERSON),
    (DOG, BOX, CHAIR, CHAIR_2, CHAIR_3, CHAIR_4, CHAIR_5, _TIRE),
    (BABY,),
    (_BODY,),
    (_TORSO, _CHAIR_BACK, _CHAIR_SEAT),
    (_ARM, _ANIMAL_LEG, _INANIMATE_LEG),
    (WATERMELON, HAND, HEAD, _ARM_SEGMENT, _LEG_SEGMENT, _FOOT),
    (BALL, BIRD, BOOK, COOKIE, CUP, CUP_2, CUP_3, CUP_4, HAT, JUICE, WATER, MILK),
    (_TAIL, _WING),
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
        GAILA_PHASE_2_SIZE_GRADES, relation_type=BIGGER_THAN, opposite_type=SMALLER_THAN
    ),
)


def gravitationally_aligned_axis_is_largest(
    ontology_node: OntologyNode, ontology: Ontology
) -> bool:
    schemata = list(ontology.structural_schemata(ontology_node))
    if not schemata or len(schemata) != 1:
        return False
    gravitational = schemata[0].axes.gravitationally_aligned_axis
    relations = schemata[0].axes.axis_relations
    if not gravitational or not relations:
        return False
    return (
        any(
            r.first_slot == gravitational
            and r.relation_type in [BIGGER_THAN, MUCH_BIGGER_THAN]
            for r in relations
        )
        and not any(
            r.first_slot == gravitational
            and r.relation_type in [SMALLER_THAN, MUCH_SMALLER_THAN]
            for r in relations
        )
        and not any(
            r.second_slot == gravitational
            and r.relation_type in [BIGGER_THAN, MUCH_BIGGER_THAN]
            for r in relations
        )
    )
