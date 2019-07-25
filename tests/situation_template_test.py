from adam.ontology._testing_ontology import INANIMATE_OBJECT, ANIMATE_OBJECT
from adam.situation.templates import SimpleSituationTemplate


def test_objects_only_template():
    sit_b = SimpleSituationTemplate.Builder()
    sit_b.object("object1", INANIMATE_OBJECT)
    sit_b.object("person", ANIMATE_OBJECT)

    sit_b.build()
