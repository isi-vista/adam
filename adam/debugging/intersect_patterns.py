# pylint:disable=invalid-name
r"""
A tool for debugging graph intersection by intersecting two serialized `PerceptionGraphPattern`\ s .

These can be serialized using a `GraphLogger` .
"""
import logging
import os
import pickle
import sys
from logging import INFO
from pathlib import Path

from adam.ontology.phase1_ontology import GAILA_PHASE_1_ONTOLOGY
from adam.perception.perception_graph import PerceptionGraphPattern, GraphLogger

if __name__ == "__main__":
    if len(sys.argv) == 3:
        first_pattern_file = sys.argv[1]
        second_pattern_file = sys.argv[2]
    else:
        raise RuntimeError("Expected two arguments, the serialized patterns to intersect")
    logging.getLogger().setLevel(logging.INFO)
    with open(first_pattern_file, "rb") as first_in:
        first_pattern: PerceptionGraphPattern = pickle.load(first_in)
    with open(second_pattern_file, "rb") as second_in:
        second_pattern: PerceptionGraphPattern = pickle.load(second_in)

    graph_logger = GraphLogger(Path(os.getcwd()), enable_graph_rendering=True)
    graph_logger.log_graph(first_pattern, INFO, "First pattern")
    graph_logger.log_graph(second_pattern, INFO, "Second pattern")

    intersection_forward = first_pattern.intersection(
        second_pattern, ontology=GAILA_PHASE_1_ONTOLOGY
    )
    if intersection_forward:
        graph_logger.log_graph(
            intersection_forward, INFO, "Intersected graph, first against second"
        )
    else:
        logging.info("Forward intersection is empty")

    intersection_backward = second_pattern.intersection(
        first_pattern, ontology=GAILA_PHASE_1_ONTOLOGY
    )
    if intersection_backward:
        graph_logger.log_graph(
            intersection_backward, INFO, "Intersected graph, second against first"
        )
    else:
        logging.info("Backward intersection is empty")
