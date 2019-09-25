from more_itertools import first

from adam.axes import AxesInfo, PrimaryAxisOfObject
from adam.ontology.phase1_ontology import GAILA_PHASE_1_ONTOLOGY, HAND
from adam.perception import ObjectPerception


def test_primary_axis_function():
    hand_axes = first(GAILA_PHASE_1_ONTOLOGY.structural_schemata(HAND)).axes
    hand = ObjectPerception("hand", axes=hand_axes)
    hand_primary_axis = PrimaryAxisOfObject(hand).to_concrete_axis(AxesInfo())
    assert hand_primary_axis == hand_axes.primary_axis
