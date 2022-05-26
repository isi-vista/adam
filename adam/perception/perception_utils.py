from math import sqrt
from typing import Iterable


def dist(list_1: Iterable, list_2: Iterable) -> float:
    """
    Computes Euclidean distance between two iterables.

    Taken from Python 3.8: https://docs.python.org/3/library/math.html#math.dist"""

    return sqrt(sum((px - qx) ** 2.0 for px, qx in zip(list_1, list_2)))
