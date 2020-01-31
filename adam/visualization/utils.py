#pragma: no cover
import enum
from typing import Optional, List, Tuple
from immutablecollections import immutableset, immutabledict, ImmutableDict
from adam.perception import ObjectPerception
from adam.geon import CrossSection
from adam.ontology.phase1_ontology import (
    PHASE_1_CURRICULUM_OBJECTS,
    OntologyNode,
    GAILA_PHASE_1_SIZE_GRADES,
    MOM,
    DAD,
)

OBJECT_NAMES_TO_EXCLUDE = immutableset(["the ground", "learner"])


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
MODEL_NAMES = ["ball", "hat", "cup", "table", "door", "book", "car", "bird", "chair"]

NAME_TO_ONTOLOGY_NODE: ImmutableDict[str, OntologyNode] = immutabledict(
    (node.handle, node) for node in PHASE_1_CURRICULUM_OBJECTS
)


def model_lookup(object_percept: ObjectPerception) -> Optional[str]:
    """
    Utility function to find a model name from an ObjectPerception
    Args:
        object_percept:

    Returns: string of the name of this object percept's geon or specific model

    """
    # if the object has a specific model, return that

    name = object_percept.debug_handle.split("_")[0]
    if name in MODEL_NAMES:
        return name

    if object_percept.geon is None:
        return None

    # otherwise return its geon's name

    shape = cross_section_to_geon(object_percept.geon.cross_section)
    if shape in GEON_SHAPES:
        return shape.name

    return None


def _create_object_scale_multiplier_mapping() -> ImmutableDict[str, float]:

    # get the index in the size grades corresponding to people, as a reference point
    human_index = GAILA_PHASE_1_SIZE_GRADES.index((MOM, DAD))

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
