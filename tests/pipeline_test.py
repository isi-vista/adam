from vistautils.iter_utils import only

from adam.language import TokenSequenceLinguisticDescription
from adam.learner import LearningExample, MemorizingLanguageLearner
from adam.perception import (
    BagOfFeaturesPerceptualRepresentationFrame,
    PerceptualRepresentation,
)


def test_pipeline():
    curriculum = [
        LearningExample(
            perception=PerceptualRepresentation(
                [BagOfFeaturesPerceptualRepresentationFrame(("red", "truck"))]
            ),
            linguistic_description=TokenSequenceLinguisticDescription(("red", "truck")),
        )
    ]

    learner: MemorizingLanguageLearner[
        BagOfFeaturesPerceptualRepresentationFrame, TokenSequenceLinguisticDescription
    ] = MemorizingLanguageLearner()

    for example in curriculum:
        learner.observe(example)

    # shouldn't be able to describe "red" or "truck" alone
    assert not learner.describe(
        PerceptualRepresentation([BagOfFeaturesPerceptualRepresentationFrame(("red",))])
    ).description_to_confidence

    assert not learner.describe(
        PerceptualRepresentation([BagOfFeaturesPerceptualRepresentationFrame(("truck",))])
    ).description_to_confidence
    # but should be able to describe "red truck"
    red_truck_descriptions = learner.describe(
        PerceptualRepresentation(
            [BagOfFeaturesPerceptualRepresentationFrame(("red", "truck"))]
        )
    ).description_to_confidence
    assert len(red_truck_descriptions) == 1
    red_truck_description = only(red_truck_descriptions)
    assert red_truck_description.as_token_sequence() == ("red", "truck")
