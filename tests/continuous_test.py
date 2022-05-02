import logging
from math import sqrt
from platform import python_implementation
from typing import Sequence

try:
    import numpy as np
except ImportError:
    from platform import python_implementation

    if python_implementation() == "CPython":
        raise
    else:
        logging.warning(
            "Ignoring missing NumPy requirement on non-CPython implementation %s. This means we "
            "can't run the continuous value matcher tests.",
            python_implementation(),
        )
        norm = None
import pytest

from adam.continuous import GaussianContinuousValueMatcher


def _matcher_from_values(values: Sequence[float]) -> GaussianContinuousValueMatcher:
    assert values
    matcher = GaussianContinuousValueMatcher(values[0])
    for value in values[1:]:
        matcher.update_on_observation(value)
    return matcher


@pytest.mark.skipif(python_implementation() != "CPython", reason="requires SciPy")
def test_gaussian_matcher_is_correct_after_construction():
    matcher = GaussianContinuousValueMatcher(0.0)
    assert np.isnan(matcher.match_score(0.0))


@pytest.mark.skipif(python_implementation() != "CPython", reason="requires SciPy")
def test_gaussian_matcher_parameters_are_correct_after_update():
    matcher = _matcher_from_values([-sqrt(1 / 2), sqrt(1 / 2)])
    assert matcher.mean == 0.0
    assert abs(matcher.sample_variance - 1.0) < 0.0005


@pytest.mark.skipif(python_implementation() != "CPython", reason="requires SciPy")
def test_gaussian_matcher_parameters_are_correct_for_annoying_sample():
    # Values sampled from numpy.random.standard_normal() with seed 103, 2022-04-14
    values = [
        -1.249,
        -0.260,
        0.384,
        -0.385,
        -1.085,
        2.327,
        0.431,
        0.432,
        -0.980,
        -0.632,
        0.577,
        -0.125,
        0.979,
        1.595,
        -1.202,
        -1.376,
        1.054,
        -0.039,
        0.680,
        1.330,
    ]
    matcher = _matcher_from_values(values)
    assert abs(matcher.mean - np.mean(values)) < 0.0005
    assert abs(matcher.sample_variance - np.var(values, ddof=1)) < 0.0005


@pytest.mark.skipif(python_implementation() != "CPython", reason="requires SciPy")
def test_gaussian_matcher_scores_are_correct_after_updates():
    # Values sampled from numpy.random.standard_normal() with seed 103, 2022-04-14
    values = [
        -1.249,
        -0.260,
        0.384,
        -0.385,
        -1.085,
        2.327,
        0.431,
        0.432,
        -0.980,
        -0.632,
        0.577,
        -0.125,
        0.979,
        1.595,
        -1.202,
        -1.376,
        1.054,
        -0.039,
        0.680,
        1.330,
    ]
    matcher = _matcher_from_values(values)
    assert abs(matcher.match_score(0.123) - 1.0) < 0.0005
    assert abs(matcher.match_score(-1.0) - 0.279) < 0.0005
    assert abs(matcher.match_score(1.0) - 0.398) < 0.0005
