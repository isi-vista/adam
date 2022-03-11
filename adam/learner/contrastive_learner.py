from pathlib import Path
from typing import Protocol, NamedTuple

from attr import attrs, attrib
from attr.validators import instance_of

from adam.learner import ApprenticeLearner, LanguagePerceptionSemanticAlignment
from adam.learner.objects import SubsetObjectLearner


@attrs
class LanguagePerceptionSemanticContrast(NamedTuple):
    """
    Defines a contrasting pair of `LanguagePerceptionSemanticAlignment` s that a contrastive learner
    can learn from.

    By contrasting we mean the observations are of different *perceptions* and have different
    associated concepts or *language*.
    """

    first_alignment: LanguagePerceptionSemanticAlignment
    second_alignment: LanguagePerceptionSemanticAlignment


class ContrastiveLearner(Protocol):
    """
    A learner that can learn from a `LanguagePerceptionSemanticContrast`.

    Note that such learners are not expected to contribute to description at all. Rather they are
    meant to modify the behavior of other learners.
    """

    def learn_from(self, matching: LanguagePerceptionSemanticContrast) -> None:
        """
        Learn from the given pair of semantically-aligned inputs.
        """

    def log_hypotheses(self, log_output_path: Path) -> None:
        """
        Log some representation of the contrastive learner's hypotheses to the given log directory.
        """


@attrs
class TeachingContrastiveObjectLearner(Protocol):
    """
    A learner that learns contrasts between objects.

    This learner revolves around an apprentice. This learner expects to learn from the contrasting
    alignments produced by its apprenticed object learner. It learns in the sense that it pushes its
    understanding down into the apprentice.
    """

    apprentice: ApprenticeLearner = attrib(validator=instance_of(SubsetObjectLearner))

    def learn_from(self, matching: LanguagePerceptionSemanticContrast) -> None:
        """
        Learn from the given pair of semantically-aligned inputs.
        """
        raise NotImplementedError()

    def log_hypotheses(self, log_output_path: Path) -> None:
        """
        Log some representation of the contrastive learner's hypotheses to the given log directory.
        """
        raise NotImplementedError()
