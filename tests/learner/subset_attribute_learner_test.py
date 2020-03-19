from itertools import chain

import pytest
from more_itertools import flatten

from adam.curriculum.curriculum_utils import (
    standard_object,
    phase1_instances,
    PHASE1_CHOOSER_FACTORY,
)
from adam.curriculum.phase1_curriculum import _object_with_color_template
from adam.learner import LearningExample
from adam.learner.attributes import SubsetAttributeLearner
from adam.ontology.phase1_ontology import (
    RED,
    WHITE,
    BLACK,
    GREEN,
    BLUE,
    BALL,
    BOOK,
    GAILA_PHASE_1_ONTOLOGY,
    CAR,
)
from adam.situation.templates.phase1_templates import property_variable, sampled
from tests.learner import TEST_OBJECT_RECOGNIZER

SUBSET_ATTRIBUTE_LEARNER_FACTORY = lambda: SubsetAttributeLearner(
    object_recognizer=TEST_OBJECT_RECOGNIZER, ontology=GAILA_PHASE_1_ONTOLOGY
)


@pytest.mark.parametrize(
    "color_node,object_0_node,object_1_node",
    [
        # (RED, BALL, BOOK),
        # (BLUE, BALL, BOOK),
        # (GREEN, BALL, BOOK),
        (BLACK, BALL, CAR),
        # (WHITE, BALL, CAR),
    ],
)
def test_subset_color_attribute_learner(color_node, object_0_node, object_1_node):
    color = property_variable(f"{color_node.handle}", color_node)
    object_0 = standard_object(
        f"{object_0_node.handle}", object_0_node, added_properties=[color]
    )
    object_1 = standard_object(
        f"{object_1_node.handle}", object_1_node, added_properties=[color]
    )

    color_object_template = _object_with_color_template(object_0)

    templates = [color_object_template, _object_with_color_template(object_1)]

    color_train_curriculum = phase1_instances(
        f"{color.handle} Color Train",
        situations=chain(
            *[
                flatten(
                    [
                        sampled(
                            template,
                            chooser=PHASE1_CHOOSER_FACTORY(),
                            ontology=GAILA_PHASE_1_ONTOLOGY,
                            max_to_sample=2,
                        )
                        for template in templates
                    ]
                )
            ]
        ),
    )

    color_test_curriculum = phase1_instances(
        f"{color.handle} Color Test",
        situations=sampled(
            color_object_template,
            chooser=PHASE1_CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
            max_to_sample=1,
        ),
    )

    learner = SUBSET_ATTRIBUTE_LEARNER_FACTORY()

    for (
        _,
        linguistic_description,
        perceptual_representation,
    ) in color_train_curriculum.instances():
        learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )

    for (
        _,
        test_lingustics_description,
        test_perceptual_representation,
    ) in color_test_curriculum.instances():
        descriptions_from_learner = learner.describe(test_perceptual_representation)
        gold = test_lingustics_description.as_token_sequence()
        assert descriptions_from_learner
        assert [desc.as_token_sequence() for desc in descriptions_from_learner][0] == gold
