from adam.ontology.phase1_ontology import (
    INANIMATE_OBJECT,
    ANIMATE,
    PERSON,
    INANIMATE,
    GAILA_PHASE_1_ONTOLOGY,
)
from adam.random_utils import FixedIndexChooser
from adam.situation.templates import (
    SimpleSituationTemplate,
    SimpleSituationTemplateProcessor,
)


def build_objects_only_template() -> SimpleSituationTemplate:
    sit_b = SimpleSituationTemplate.Builder()
    sit_b.object_variable("object1", INANIMATE_OBJECT)
    sit_b.object_variable("person", PERSON)

    return sit_b.build()


def test_objects_only_template():
    obj_only_template = build_objects_only_template()
    assert len(obj_only_template.objects) == 2
    possible_obj1s = [
        obj
        for obj in obj_only_template.objects
        if obj.handle == "object1"  # pylint:disable=protected-access
    ]
    assert len(possible_obj1s) == 1
    obj1 = possible_obj1s[0]
    assert obj_only_template.objects_to_ontology_types[obj1] == INANIMATE_OBJECT

    possible_persons = [
        obj
        for obj in obj_only_template.objects
        if obj.handle == "person"  # pylint:disable=protected-access
    ]
    assert len(possible_persons) == 1
    person = possible_persons[0]

    assert obj_only_template.objects_to_ontology_types[person] == PERSON


def test_simple_situation_generation():
    situation_processor = SimpleSituationTemplateProcessor(GAILA_PHASE_1_ONTOLOGY)
    situations = situation_processor.generate_situations(
        build_objects_only_template(), chooser=FixedIndexChooser(0)
    )
    assert len(situations) == 1
    situation = situations[0]
    assert len(situation.objects_to_locations) == 2
    # check the objects are placed in distinct locations
    assert len(set(situation.objects_to_locations.values())) == 2
    assert (
        len([obj for obj in situation.objects_to_locations if ANIMATE in obj.properties])
        == 1
    )
    assert (
        len(
            [obj for obj in situation.objects_to_locations if INANIMATE in obj.properties]
        )
        == 1
    )
