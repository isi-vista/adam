"""
The `Ontology` for use in ISI's GAILA Phase 1 effort.

Note that this `Ontology` is only used for training and testing example generation; the learner
has no access to it.

The following will eventually end up here:

- Objects: mommy, daddy, baby, book, house, car, water, ball, juice, cup, box, chair, head,
  milk, hand, dog, truck, door, hat, table, cookie, bird
- Actions/Verbs: go, put, come, take, eat, give, turn, sit, drink, push, fall, throw, move, jump,
  has (possessive), give, roll, fly
- Relations, Modifiers, Function Words: basic color terms (red, blue, green, white, black…), one,
  two, I, me, my, you, your, to, in, on, [beside, behind, in front of, over, under], up, down
"""
from networkx import DiGraph

from adam.ontology import OntologyProperty, OntologyNode, Ontology

ANIMATE = OntologyProperty("animate")
INANIMATE = OntologyProperty("inanimate")

RECOGNIZED_PARTICULAR = OntologyProperty("recognized-particular")
"""
Indicates that a node in the ontology corresponds to a particular (rather than a class)
which is assumed to be known to the `LanguageLearner`. 
The prototypical cases here are *Mom* and *Dad*.
"""

_ontology_graph = DiGraph()  # pylint:disable=invalid-name


def subtype(sub: OntologyNode, _super: OntologyNode) -> None:
    _ontology_graph.add_edge(sub, _super)


PHYSICAL_OBJECT = OntologyNode("object")

INANIMATE_OBJECT = OntologyNode("inanimate-object", [INANIMATE])
subtype(INANIMATE_OBJECT, PHYSICAL_OBJECT)
TABLE = OntologyNode("table")
subtype(TABLE, INANIMATE_OBJECT)
BALL = OntologyNode("ball")
subtype(BALL, INANIMATE_OBJECT)
BOOK = OntologyNode("book")
subtype(BOOK, INANIMATE_OBJECT)
HOUSE = OntologyNode("house")
subtype(HOUSE, INANIMATE_OBJECT)
CAR = OntologyNode("car")
subtype(CAR, INANIMATE_OBJECT)
WATER = OntologyNode("water")
subtype(WATER, INANIMATE_OBJECT)
JUICE = OntologyNode("juice")
subtype(JUICE, INANIMATE_OBJECT)
CUP = OntologyNode("cup")
subtype(CUP, INANIMATE_OBJECT)
BOX = OntologyNode("box")
subtype(BOX, INANIMATE_OBJECT)
CHAIR = OntologyNode("chair")
subtype(CHAIR, INANIMATE_OBJECT)
HEAD = OntologyNode("head")
subtype(HEAD, INANIMATE_OBJECT)
MILK = OntologyNode("milk")
subtype(MILK, INANIMATE_OBJECT)
HAND = OntologyNode("hand")
subtype(HAND, INANIMATE_OBJECT)
TRUCK = OntologyNode("truck")
subtype(TRUCK, INANIMATE_OBJECT)
DOOR = OntologyNode("door")
subtype(DOOR, INANIMATE_OBJECT)
HAT = OntologyNode("hat")
subtype(HAT, INANIMATE_OBJECT)
COOKIE = OntologyNode("cookie")
subtype(COOKIE, INANIMATE_OBJECT)

PERSON = OntologyNode("person", [ANIMATE])
subtype(PERSON, PHYSICAL_OBJECT)
MOM = OntologyNode("mom", [RECOGNIZED_PARTICULAR])
subtype(MOM, PERSON)
DAD = OntologyNode("dad", [RECOGNIZED_PARTICULAR])
subtype(DAD, PERSON)
BABY = OntologyNode("baby")
subtype(BABY, PERSON)

NONHUMAN_ANIMAL = OntologyNode("animal", [ANIMATE])
subtype(NONHUMAN_ANIMAL, PHYSICAL_OBJECT)
DOG = OntologyNode("dog")
subtype(DOG, NONHUMAN_ANIMAL)
BIRD = OntologyNode("bird")
subtype(BIRD, NONHUMAN_ANIMAL)

ACTION = OntologyNode("action")
PUT = OntologyNode("put")
PUSH = OntologyNode("push")
subtype(PUT, ACTION)
subtype(PUSH, ACTION)

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
