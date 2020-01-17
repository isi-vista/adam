import enum
from typing import Optional
from immutablecollections import immutableset
from adam.perception import ObjectPerception
from adam.geon import CrossSection

OBJECT_NAMES_TO_EXCLUDE = immutableset(["the ground", "learner"])


class Shape(enum.Enum):
    CIRCULAR = "CIRCULAR"
    SQUARE = "SQUARE"
    OVALISH = "OVALISH"
    RECTANGULAR = "RECTANGULAR"
    IRREGULAR = "IRREGULAR"


def cross_section_to_geo(cs: CrossSection) -> Shape:
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
MODEL_NAMES = ["ball", "hat", "cup", "table", "door", "book"]


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
    shape = cross_section_to_geo(object_percept.geon.cross_section)
    if shape in GEON_SHAPES:
        return shape.name

    return None
