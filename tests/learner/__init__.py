from typing import Mapping

from more_itertools import first

from adam.language_specific.english import ENGLISH_DETERMINERS
from adam.learner.object_recognizer import ObjectRecognizer

from adam.ontology.phase1_ontology import (
    GAILA_PHASE_1_ONTOLOGY,
    PHASE_1_CURRICULUM_OBJECTS,
)
from adam.perception.perception_graph import PerceptionGraphPattern
from immutablecollections import immutabledict

_TEST_OBJECTS: Mapping[str, PerceptionGraphPattern] = immutabledict(
    (
        node.handle,
        PerceptionGraphPattern.from_schema(
            first(GAILA_PHASE_1_ONTOLOGY.structural_schemata(node))
        ),
    )
    for node in PHASE_1_CURRICULUM_OBJECTS
    if node
    in GAILA_PHASE_1_ONTOLOGY._structural_schemata.keys()  # pylint:disable=protected-access
)
TEST_OBJECT_RECOGNIZER = ObjectRecognizer(_TEST_OBJECTS, determiners=ENGLISH_DETERMINERS)
