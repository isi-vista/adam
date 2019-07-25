from adam.ontology.testing_ontology import TRUCK, INANIMATE, ANIMATE, PERSON
from adam.situation.templates import SimpleSituationTemplate


def test_objects_only_template():
    sit_b = SimpleSituationTemplate.Builder()
    # TODO: looser ontology constraints
    sit_b.object("truck", TRUCK, [INANIMATE])
    sit_b.object("person", PERSON, [ANIMATE])

    sit_b.build()
