from immutablecollections import immutableset
from typing import Tuple

from adam.ontology import OntologyNode, CAN_FILL_TEMPLATE_SLOT
from adam.ontology.ontology import Ontology
from adam.ontology.phase1_ontology import (
    BIGGER_THAN,
    SMALLER_THAN,
    _ACTIONS_TO_DESCRIPTIONS,
    _ontology_graph,
    HOUSE,
    _ROOF,
    _WALL,
    CAR,
    TRUCK,
    _TRAILER,
    _FLATBED,
    _TRUCK_CAB,
    TABLE,
    DOOR,
    _TABLETOP,
    MOM,
    DAD,
    PERSON,
    DOG,
    BOX,
    CHAIR,
    _TIRE,
    BABY,
    _BODY,
    _TORSO,
    _CHAIR_BACK,
    _CHAIR_SEAT,
    _ARM,
    _ANIMAL_LEG,
    _INANIMATE_LEG,
    WATERMELON,
    HAND,
    HEAD,
    _ARM_SEGMENT,
    _LEG_SEGMENT,
    _FOOT,
    BALL,
    BIRD,
    BOOK,
    COOKIE,
    CUP,
    HAT,
    JUICE,
    WATER,
    MILK,
    _TAIL,
    _WING,
    HOLLOW,
    CAN_HAVE_THINGS_RESTING_ON_THEM,
    PERSON_CAN_HAVE,
    RED,
    subtype,
    INANIMATE_OBJECT,
    BLUE,
    GREEN,
    ROLLABLE,
    CAT,
    _BALL_SCHEMA,
    _BOX_SCHEMA,
    INTEGRATED_EXPERIMENT_PROP,
    BEAR,
)
from adam.ontology.phase1_size_relationships import build_size_relationships
from adam.ontology.phase2_ontology import (
    GAILA_PHASE_2_ONTOLOGY,
    CHAIR_2,
    CHAIR_3,
    CHAIR_4,
    CHAIR_5,
    CUP_2,
    CUP_3,
    CUP_4,
)

MAWG = OntologyNode(
    "mawg",
    [
        HOLLOW,
        CAN_FILL_TEMPLATE_SLOT,
        CAN_HAVE_THINGS_RESTING_ON_THEM,
        PERSON_CAN_HAVE,
        RED,
        INTEGRATED_EXPERIMENT_PROP,
    ],
)
subtype(MAWG, INANIMATE_OBJECT)
TOMBUR = OntologyNode(
    "tombur",
    [
        HOLLOW,
        CAN_FILL_TEMPLATE_SLOT,
        CAN_HAVE_THINGS_RESTING_ON_THEM,
        PERSON_CAN_HAVE,
        BLUE,
        INTEGRATED_EXPERIMENT_PROP,
    ],
)
subtype(TOMBUR, INANIMATE_OBJECT)
GLIM = OntologyNode(
    "glim",
    [
        HOLLOW,
        CAN_FILL_TEMPLATE_SLOT,
        CAN_HAVE_THINGS_RESTING_ON_THEM,
        PERSON_CAN_HAVE,
        GREEN,
        INTEGRATED_EXPERIMENT_PROP,
    ],
)
subtype(GLIM, INANIMATE_OBJECT)
ZUP = OntologyNode(
    "zup",
    [CAN_FILL_TEMPLATE_SLOT, PERSON_CAN_HAVE, ROLLABLE, RED, INTEGRATED_EXPERIMENT_PROP],
)
subtype(ZUP, INANIMATE_OBJECT)
SPAD = OntologyNode(
    "spad",
    [CAN_FILL_TEMPLATE_SLOT, PERSON_CAN_HAVE, ROLLABLE, BLUE, INTEGRATED_EXPERIMENT_PROP],
)
subtype(SPAD, INANIMATE_OBJECT)
DAYGIN = OntologyNode(
    "daygin",
    [
        CAN_FILL_TEMPLATE_SLOT,
        PERSON_CAN_HAVE,
        ROLLABLE,
        GREEN,
        INTEGRATED_EXPERIMENT_PROP,
    ],
)
subtype(DAYGIN, INANIMATE_OBJECT)

INTEGRATED_EXPERIMENT_CURRICULUM_OBJECTS = immutableset(
    [MAWG, TOMBUR, GLIM, ZUP, SPAD, DAYGIN, CAT, DOG, BIRD, BEAR, CUP, BOOK]
)

INTEGRATED_EXPERIMENT_SIZE_GRADES: Tuple[Tuple[OntologyNode, ...], ...] = (
    (HOUSE,),
    (_ROOF, _WALL),
    (CAR, TRUCK),
    (_TRAILER, _FLATBED),
    (_TRUCK_CAB,),
    (TABLE, DOOR),
    (_TABLETOP,),
    (MOM, DAD, PERSON),
    (DOG, BOX, MAWG, TOMBUR, GLIM, CHAIR, CHAIR_2, CHAIR_3, CHAIR_4, CHAIR_5, _TIRE),
    (BABY,),
    (_BODY,),
    (_TORSO, _CHAIR_BACK, _CHAIR_SEAT),
    (_ARM, _ANIMAL_LEG, _INANIMATE_LEG),
    (WATERMELON, HAND, HEAD, _ARM_SEGMENT, _LEG_SEGMENT, _FOOT),
    (
        BALL,
        ZUP,
        SPAD,
        DAYGIN,
        BIRD,
        BOOK,
        COOKIE,
        CUP,
        CUP_2,
        CUP_3,
        CUP_4,
        HAT,
        JUICE,
        WATER,
        MILK,
    ),
    (_TAIL, _WING),
)

INTEGRATED_EXPERIMENT_ONTOLOGY = Ontology(
    "integrated_experiment_ontology",
    _ontology_graph,
    structural_schemata=[
        schemata
        for schemata in GAILA_PHASE_2_ONTOLOGY._structural_schemata.items()  # pylint: disable=protected-access
    ]
    + [
        (ZUP, _BALL_SCHEMA),
        (SPAD, _BALL_SCHEMA),
        (DAYGIN, _BALL_SCHEMA),
        (MAWG, _BOX_SCHEMA),
        (TOMBUR, _BOX_SCHEMA),
        (GLIM, _BOX_SCHEMA),
    ],
    action_to_description=_ACTIONS_TO_DESCRIPTIONS,
    relations=build_size_relationships(
        INTEGRATED_EXPERIMENT_SIZE_GRADES,
        relation_type=BIGGER_THAN,
        opposite_type=SMALLER_THAN,
    ),
)
