from math import sqrt
from typing import Sequence

import numpy as np

from adam.continuous import GaussianContinuousValueMatcher


def _matcher_from_values(values: Sequence[float]) -> GaussianContinuousValueMatcher:
    assert values
    matcher = GaussianContinuousValueMatcher(values[0])
    for value in values[1:]:
        matcher.update_on_observation(value)
    return matcher


def test_gaussian_matcher_is_correct_after_construction():
    matcher = GaussianContinuousValueMatcher(0.0)
    assert matcher.match_score(0.0) == 1.0


def test_gaussian_matcher_parameters_are_correct_after_update():
    matcher = _matcher_from_values([-sqrt(1 / 2), sqrt(1 / 2)])
    assert matcher.mean == 0.0
    assert abs(matcher.sample_variance - 1.0) < 0.0005


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