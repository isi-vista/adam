"""
The `Ontology` for use in ISI's GAILA Phase 1 effort.

Note that this `Ontology` is only used for training and testing example generation; the learner
has no access to it.
"""
from networkx import DiGraph

from adam.ontology import OntologyProperty, OntologyNode, Ontology

ANIMATE = OntologyProperty("animate")
INANIMATE = OntologyProperty("inanimate")

RECOGNIZED_PARTICULAR = OntologyProperty("recognized-particular")
"""
Indicates that a node in the ontology corresponds to a particular (rather than a class)
which is assumed to be known to the `LanguageLearner`. The prototypical case here is *Mom*.
"""

_ontology_graph = DiGraph()  # pylint:disable=invalid-name


def subtype(sub: OntologyNode, _super: OntologyNode) -> None:
    _ontology_graph.add_edge(sub, _super)


OBJECT = OntologyNode("object")

INANIMATE_OBJECT = OntologyNode("inanimate-object", [INANIMATE])
subtype(INANIMATE_OBJECT, OBJECT)
TABLE = OntologyNode("table")
subtype(TABLE, INANIMATE_OBJECT)
BALL = OntologyNode("ball")
subtype(BALL, INANIMATE_OBJECT)

PERSON = OntologyNode("person", [ANIMATE])
subtype(PERSON, OBJECT)
MOM = OntologyNode("mom", [RECOGNIZED_PARTICULAR])
subtype(MOM, PERSON)
DAD = OntologyNode("dad", [RECOGNIZED_PARTICULAR])
subtype(DAD, PERSON)

ACTION = OntologyNode("action")
PUT = OntologyNode("put")
subtype(PUT, ACTION)

LOCATION = OntologyNode("location")

RELATION = OntologyNode("relation")
SPATIAL_RELATION = OntologyNode("spatial-relation")
subtype(RELATION, SPATIAL_RELATION)
# On is an English-specific bundle of semantics, but that's okay, because this is just for
# data generation, and it will get decomposed before being presented as perceptions to the
# learner.
ON = OntologyNode("on")
subtype(ON, SPATIAL_RELATION)

SEMANTIC_ROLE = OntologyNode("semantic-role")
AGENT = OntologyNode("agent")
subtype(AGENT, SEMANTIC_ROLE)
PATIENT = OntologyNode("patient")
subtype(PATIENT, SEMANTIC_ROLE)
THEME = OntologyNode("theme")
subtype(THEME, SEMANTIC_ROLE)
DESTINATION = OntologyNode("destination")
subtype(DESTINATION, SEMANTIC_ROLE)

GAILA_PHASE_1_ONTOLOGY = Ontology.from_directed_graph(_ontology_graph)
