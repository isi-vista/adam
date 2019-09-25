from more_itertools import first

from adam.geon import PrimaryAxisOfObject, AxesInfo
from adam.ontology.phase1_ontology import GAILA_PHASE_1_ONTOLOGY, HAND
from adam.perception import ObjectPerception


def test_primary_axis_function():
    hand_axes = first(GAILA_PHASE_1_ONTOLOGY.structural_schemata(HAND)).axes
    hand = ObjectPerception("hand", axes=hand_axes)
    hand_primary_axis = PrimaryAxisOfObject(hand).select_axis(
        AxesInfo(objects_to_axes=[])
    )
    assert hand_primary_axis == hand_axes.primary_axis
