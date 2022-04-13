"""
Code related to continuous feature matching.
"""
from abc import abstractmethod
from typing_extensions import Protocol

from attr import attrs, attrib
from attr.validators import instance_of


class ContinuousValueMatcher(Protocol):
    """
    Defines a soft way of matching continuous values and a way of integrating new observations.
    """

    @property
    @abstractmethod
    def n_observations(self) -> int:
        """
        Return the number of observations so far.
        """

    def match_score(self, value: float) -> float:
        """
        Return a score representing how closely the given value matches this distribution.

        This score should fall into the interval [0, 1] so that learners can threshold scores
        consistently across different matcher types.
        """

    def update_on_observation(self, value: float) -> None:
        """
        Update the matcher's distribution to account for the given value.
        """

    def merge(self, other: "ContinuousValueMatcher") -> None:
        """
        Update this matcher in place using information from the given other matcher.
        """


@attrs
class GaussianContinuousValueMatcher(ContinuousValueMatcher):
    """
    Implements soft value matching where we pretend values come from a Gaussian distribution.
    """
    _mean: float = attrib(validator=instance_of(float))
    _sum_of_squared_differences: float = attrib(validator=instance_of(float), init=False, default=0.0)
    """
    Also called M_{2,n} e.g. on the Wikipedia page.
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """
    _nobs: float = attrib(validator=instance_of(int), init=False, default=1)
    _min: float = attrib(validator=instance_of(float), init=False)
    _max: float = attrib(validator=instance_of(float), init=False)

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def sample_variance(self) -> float:
        if self._nobs < 2:
            return float("nan")
        else:
            return self._sum_of_squared_differences / (self._nobs - 1)

    @_min.default
    def _init_min(self) -> float:
        return self._mean

    @_max.default
    def _init_max(self) -> float:
        return self._mean

    @staticmethod
    def from_observation(value) -> "GaussianContinuousValueMatcher":
        """
        Return a new Gaussian continuous matcher created from the given single observation.

        This exists for clarity more than anything.
        """
        return GaussianContinuousValueMatcher(value)

    def match_score(self, value: float) -> float:
        """
        Return a score representing how closely the given value matches this distribution.

        This score should fall into the interval [0, 1] so that learners can threshold scores
        consistently across different matcher types.
        """
        raise NotImplementedError()

    def update_on_observation(self, value: float) -> None:
        """
        Update the matcher's distribution to account for the given value.
        """
        raise NotImplementedError()
