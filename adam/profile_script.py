from adam.learner.language_mode import LanguageMode
from tests.learner.subset_verb_learner_test import (
    test_eat_simple,
    integrated_learner_factory,
)

if __name__ == "__main__":
    test_eat_simple(
        language_mode=LanguageMode.ENGLISH, learner=integrated_learner_factory
    )
