"""
This is a simple ontology for use in testing.

This is for ADAM-internal use only.
"""
from networkx import DiGraph

from adam.ontology import OntologyNode, OntologyProperty, Ontology

ANIMATE = OntologyProperty("animate")
INANIMATE = OntologyProperty("inanimate")

_ontology_graph = DiGraph()  # pylint:disable=invalid-name

ACTION = OntologyNode("action")

OBJECT = OntologyNode("object")
ANIMATE_OBJECT = OntologyNode("animate object", [ANIMATE])
_ontology_graph.add_edge(ANIMATE_OBJECT, OBJECT)
INANIMATE_OBJECT = OntologyNode("inanimate object", [INANIMATE])
_ontology_graph.add_edge(INANIMATE_OBJECT, OBJECT)

TRUCK = OntologyNode("truck")
_ontology_graph.add_edge(TRUCK, INANIMATE_OBJECT)
BALL = OntologyNode("ball")
_ontology_graph.add_edge(BALL, INANIMATE_OBJECT)

PERSON = OntologyNode("person")
_ontology_graph.add_edge(PERSON, ANIMATE_OBJECT)
DOG = OntologyNode("dog")
_ontology_graph.add_edge(DOG, ANIMATE_OBJECT)

TESTING_ONTOLOGY = Ontology.from_directed_graph(_ontology_graph)
