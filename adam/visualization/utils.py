# pragma: no cover
import enum
from typing import Optional, List, Tuple
from immutablecollections import immutableset, immutabledict, ImmutableDict
from panda3d.core import NodePath  # pylint: disable=no-name-in-module
from adam.perception import ObjectPerception
from adam.geon import CrossSection
from adam.ontology.phase1_ontology import (
    PHASE_1_CURRICULUM_OBJECTS,
    OntologyNode,
    GAILA_PHASE_1_SIZE_GRADES,
    MOM,
    DAD,
    PERSON,
)

from adam.ontology.phase1_ontology import (
    _ARM,
    _TORSO,
    _ANIMAL_LEG,
    _INANIMATE_LEG,
    _CHAIR_BACK,
    _CHAIR_SEAT,
    _TABLETOP,
    _TAIL,
    _WING,
    _ARM_SEGMENT,
    _WALL,
    _ROOF,
    _TIRE,
    _TRUCK_CAB,
    _TRAILER,
    _FLATBED,
    _BODY,
    _DOG_HEAD,
    _BIRD_HEAD,
    _LEG_SEGMENT,
    _FOOT,
    HEAD,
    HAND,
)

OBJECT_NAMES_TO_EXCLUDE = immutableset(["the ground", "learner"])

_SUBOBJECTS = immutableset(
    [
        _ARM,
        _TORSO,
        _ANIMAL_LEG,
        _INANIMATE_LEG,
        _CHAIR_BACK,
        _CHAIR_SEAT,
        _TABLETOP,
        _TAIL,
        _WING,
        _ARM_SEGMENT,
        _WALL,
        _ROOF,
        _TIRE,
        _TRUCK_CAB,
        _TRAILER,
        _FLATBED,
        _BODY,
        _DOG_HEAD,
        _BIRD_HEAD,
        _LEG_SEGMENT,
        _FOOT,
        HEAD,
        HAND,
    ]
)


class Shape(enum.Enum):
    CIRCULAR = "CIRCULAR"
    SQUARE = "SQUARE"
    OVALISH = "OVALISH"
    RECTANGULAR = "RECTANGULAR"
    IRREGULAR = "IRREGULAR"


def cross_section_to_geon(cs: CrossSection) -> Shape:
    """
    Converts a cross section into a geon type, based on the properties of the cross section
    Args:
        cs: CrossSection to be mapped to a Geon type

    Returns: Shape: a convenience enum mapping to a file name for the to-be-rendered geometry

    """
    if cs.has_rotational_symmetry and cs.has_reflective_symmetry and cs.curved:
        return Shape("CIRCULAR")
    elif cs.has_rotational_symmetry and cs.has_reflective_symmetry and not cs.curved:
        return Shape("SQUARE")
    elif not cs.has_rotational_symmetry and cs.has_reflective_symmetry and cs.curved:
        return Shape("OVALISH")

    elif not cs.has_rotational_symmetry and cs.has_reflective_symmetry and not cs.curved:
        return Shape("RECTANGULAR")
    elif (
        not cs.has_rotational_symmetry
        and not cs.has_reflective_symmetry
        and not cs.curved
    ):
        return Shape("IRREGULAR")
    else:
        raise ValueError("Unknown Geon composition")


# currently supported shapes and models
GEON_SHAPES = [Shape.SQUARE, Shape.CIRCULAR, Shape.OVALISH, Shape.RECTANGULAR]
MODEL_NAMES = [
    "ball",
    "hat",
    "cup",
    "table",
    "door",
    "book",
    "car",
    "bird",
    "chair",
    "dog",
    "cookie",
    "water",
    "juice",
    "milk",
    "person",
    "house",
    "truck",
    "baby",
    "Dad",
    "Mom",
]


NAME_TO_ONTOLOGY_NODE: ImmutableDict[str, OntologyNode] = immutabledict(
    (node.handle, node) for node in PHASE_1_CURRICULUM_OBJECTS
)

_PART_NAME_TO_ONTOLOGY_NODE: ImmutableDict[str, OntologyNode] = immutabledict(
    (node.handle, node) for node in _SUBOBJECTS
)

# lookup for how many instances of a subpart should exist
_PART_CARDINALITY: ImmutableDict[str, ImmutableDict[str, int]] = immutabledict(
    [
        ("table", immutabledict([("tabletop", 1), ("(furniture) leg", 4)])),
        (
            "chair",
            immutabledict([("chairback", 1), ("chairseat", 1), ("(furniture) leg", 4)]),
        ),
        (
            "dog",
            immutabledict(
                (
                    [
                        ("dog-head", 1),
                        ("foot", 4),
                        ("tail", 1),
                        ("torso", 1),
                        ("leg-segment", 8),
                    ]
                )
            ),
        ),
        ("car", immutabledict([("tire", 4), ("body", 1)])),
        (
            "person",
            immutabledict(
                [
                    ("head", 1),
                    ("torso", 1),
                    ("armsegment", 4),
                    ("leg-segment", 4),
                    ("hand", 2),
                    ("foot", 2),
                ]
            ),
        ),
        (
            "Mom",
            immutabledict(
                [
                    ("head", 1),
                    ("torso", 1),
                    ("armsegment", 4),
                    ("leg-segment", 4),
                    ("hand", 2),
                    ("foot", 2),
                ]
            ),
        ),
        (
            "Dad",
            immutabledict(
                [
                    ("head", 1),
                    ("torso", 1),
                    ("armsegment", 4),
                    ("leg-segment", 4),
                    ("hand", 2),
                    ("foot", 2),
                ]
            ),
        ),
        (
            "baby",
            immutabledict(
                [
                    ("head", 1),
                    ("torso", 1),
                    ("armsegment", 4),
                    ("leg-segment", 4),
                    ("hand", 2),
                    ("foot", 2),
                ]
            ),
        ),
        (
            "bird",
            immutabledict(
                [
                    ("bird-head", 1),
                    ("torso", 1),
                    ("leg-segment", 4),
                    ("foot", 2),
                    ("wing", 2),
                    ("tail", 1),
                ]
            ),
        ),
        ("truck", immutabledict([("tire", 8), ("body", 1), ("flatbed", 1)])),
        ("house", immutabledict([("roof", 1), ("wall", 1)])),
    ]
)

# we may need to assemble the objects that have PART_OF relations with our top level set of objects


def model_lookup(
    object_percept: ObjectPerception, parent: Optional[NodePath] = None
) -> str:
    """
    Utility function to find a model name from an ObjectPerception
    Args:
        object_percept:

    Returns: string of the name of this object percept's geon or specific model

    """
    # if the object has a specific (atomic) model, return that

    name = object_percept.debug_handle.split("_")[0]
    if name in MODEL_NAMES or object_percept.geon is None:
        return name

    # if this is a sub object, see if the part name is supported for the parent object
    if parent and name in _PART_NAME_TO_ONTOLOGY_NODE:

        # get the top level parent of this object
        while parent.parent is not None and parent.parent.name != "render":
            parent = parent.parent

        parent_name = parent.name.split("_")[0]
        # print(f"parent name: {parent_name}")
        # convert unique sub-object index into a prototypical index
        if parent_name in _PART_CARDINALITY:
            sub_object_cardinality = (
                int(object_percept.debug_handle.split("_")[1])
                % _PART_CARDINALITY[parent_name][name]
            )
            return f"{parent_name}-{name}_{sub_object_cardinality}"

    # fallback: if this object at least has a geon, we will render that instead
    if object_percept.geon:
        shape = cross_section_to_geon(object_percept.geon.cross_section)
        if shape in GEON_SHAPES:
            return shape.name

    return name


def _create_object_scale_multiplier_mapping() -> ImmutableDict[str, float]:

    # get the index in the size grades corresponding to people, as a reference point
    human_index = GAILA_PHASE_1_SIZE_GRADES.index((MOM, DAD, PERSON))

    # two different scales are operating here (linearly): things bigger than adult humans, and things smaller
    # than adult humans.
    dict_entries: List[Tuple[str, float]] = []
    for i, size_grade in enumerate(GAILA_PHASE_1_SIZE_GRADES):
        if i < human_index:
            multiplier = 1 + 0.4 * (human_index - i)
        elif i == human_index:
            multiplier = 1.0
        else:
            multiplier = (len(GAILA_PHASE_1_SIZE_GRADES) - i + 2) / (
                len(GAILA_PHASE_1_SIZE_GRADES) - human_index + 2
            )
        for ontology_node in size_grade:
            dict_entries.append((ontology_node.handle, multiplier))

    return immutabledict(dict_entries)


OBJECT_SCALE_MULTIPLIER_MAP = _create_object_scale_multiplier_mapping()
