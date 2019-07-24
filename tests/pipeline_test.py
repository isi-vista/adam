from random import Random

from vistautils.iter_utils import only

from adam.curriculum import CurriculumGenerator
from adam.learner import LearningExample, MemorizingLanguageLearner
from adam.linguistic_description import TokenSequenceLinguisticDescription
from adam.perception import (
    BagOfFeaturesPerceptualRepresentationFrame,
    PerceptualRepresentation,
)


def test_pipeline():
    curriculum_generator = CurriculumGenerator.create_always_generating(
        [
            LearningExample(
                perception=PerceptualRepresentation(
                    [BagOfFeaturesPerceptualRepresentationFrame(("red", "truck"))]
                ),
                linguistic_description=TokenSequenceLinguisticDescription(
                    ("red", "truck")
                ),
            )
        ]
    )

    learner: MemorizingLanguageLearner[
        BagOfFeaturesPerceptualRepresentationFrame, TokenSequenceLinguisticDescription
    ] = MemorizingLanguageLearner()

    for example in curriculum_generator.generate_curriculum(Random(0)):
        learner.observe(example)

    # shouldn't be able to describe "red" or "truck" alone
    assert not learner.describe(
        PerceptualRepresentation([BagOfFeaturesPerceptualRepresentationFrame(("red",))])
    )
    assert not learner.describe(
        PerceptualRepresentation([BagOfFeaturesPerceptualRepresentationFrame(("truck",))])
    )
    # but should be able to describe "red truck"
    red_truck_descriptions = learner.describe(
        PerceptualRepresentation(
            [BagOfFeaturesPerceptualRepresentationFrame(("red", "truck"))]
        )
    )
    assert len(red_truck_descriptions) == 1
    red_truck_description = only(red_truck_descriptions)
    assert red_truck_description.as_token_sequence() == ("red", "truck")
