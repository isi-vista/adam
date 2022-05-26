"""
Code related to continuous feature matching.
"""
from abc import abstractmethod
import logging
from math import sqrt
from typing import Tuple, Iterable

from typing_extensions import Protocol

from adam.perception.perception_utils import dist
from attr import attrs, attrib
from attr.validators import instance_of

try:
    from scipy.stats import norm
except ImportError:
    from platform import python_implementation

    if python_implementation() == "CPython":
        raise
    else:
        logging.warning(
            "Ignoring missing scipy requirement on non-CPython implementation %s; make sure you're "
            "not using continuous distribution matching...",
            python_implementation(),
        )
        norm = None


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
    _sum_of_squared_differences: float = attrib(
        validator=instance_of(float), init=False, default=0.0
    )
    """
    Also called M_{2,n} e.g. on the Wikipedia page.
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """
    _n_observations: int = attrib(validator=instance_of(int), init=False, default=1)

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def sample_variance(self) -> float:
        if self._n_observations < 2:
            return float("nan")
        else:
            return self._sum_of_squared_differences / (self._n_observations - 1)

    @property
    def n_observations(self) -> int:
        return self._n_observations

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
        standard_deviation = sqrt(self.sample_variance)
        return (
            1.0
            if standard_deviation == 0 and value == self._mean
            else 2.0
            * norm.cdf(
                self._mean - abs(value - self._mean),
                loc=self._mean,
                scale=standard_deviation,
            )
        )

    @staticmethod
    def _calculate_new_values(
        mean: float,
        sum_of_squared_differences: float,
        n_observations: int,
        observation: float,
    ) -> Tuple[float, float]:
        new_mean = mean + (observation - mean) / (n_observations + 1)
        new_sum_squared = sum_of_squared_differences + (observation - mean) * (
            observation - new_mean
        )
        return new_mean, new_sum_squared

    def update_on_observation(self, value: float) -> None:
        """
        Update the matcher's distribution to account for the given value.
        """
        # With some help from Wikipedia. :)
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        new_mean, new_sum_squared = self._calculate_new_values(
            self._mean, self._sum_of_squared_differences, self._n_observations, value
        )

        self._mean = new_mean
        self._sum_of_squared_differences = new_sum_squared
        self._n_observations += 1

    def merge(self, other: "ContinuousValueMatcher") -> None:
        # pylint: disable=protected-access
        # Pylint doesn't realize the "client class" whose private members we're accessing is this
        # same class
        if isinstance(other, GaussianContinuousValueMatcher):
            if self._n_observations == 1:
                # Treat our own mean as a single observation (because it is) and calculate the new
                # mean and sum of squares from the other matcher's perspective.
                new_mean, new_sum_squared = self._calculate_new_values(
                    other._mean,
                    other._sum_of_squared_differences,
                    other._n_observations,
                    self._mean,
                )
                self._mean = new_mean
                self._sum_of_squared_differences = new_sum_squared
                self._n_observations += other._n_observations
            elif other._n_observations == 1:
                self.update_on_observation(other.mean)
            else:
                raise ValueError(
                    f"Cannot merge two matchers that both have multiple observations (self with "
                    f"{self._n_observations} and other with {other._n_observations})."
                )
        else:
            raise ValueError(
                f"Cannot merge {type(self)} with matcher of foreign type {type(other)}"
            )


@attrs
class GaussianContinuousValueMatcher(ContinuousValueMatcher):
    """
    Implements soft value matching where we pretend values come from a Gaussian distribution.
    """

    _mean: float = attrib(validator=instance_of(float))
    _sum_of_squared_differences: float = attrib(
        validator=instance_of(float), init=False, default=0.0
    )
    """
    Also called M_{2,n} e.g. on the Wikipedia page.
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """
    _n_observations: int = attrib(validator=instance_of(int), init=False, default=1)

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def sample_variance(self) -> float:
        if self._n_observations < 2:
            return float("nan")
        else:
            return self._sum_of_squared_differences / (self._n_observations - 1)

    @property
    def n_observations(self) -> int:
        return self._n_observations

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
        standard_deviation = sqrt(self.sample_variance)
        return (
            1.0
            if standard_deviation == 0 and value == self._mean
            else 2.0
            * norm.cdf(
                self._mean - abs(value - self._mean),
                loc=self._mean,
                scale=standard_deviation,
            )
        )

    @staticmethod
    def _calculate_new_values(
        mean: float,
        sum_of_squared_differences: float,
        n_observations: int,
        observation: float,
    ) -> Tuple[float, float]:
        new_mean = mean + (observation - mean) / (n_observations + 1)
        new_sum_squared = sum_of_squared_differences + (observation - mean) * (
            observation - new_mean
        )
        return new_mean, new_sum_squared

    def update_on_observation(self, value: float) -> None:
        """
        Update the matcher's distribution to account for the given value.
        """
        # With some help from Wikipedia. :)
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        new_mean, new_sum_squared = self._calculate_new_values(
            self._mean, self._sum_of_squared_differences, self._n_observations, value
        )

        self._mean = new_mean
        self._sum_of_squared_differences = new_sum_squared
        self._n_observations += 1

    def merge(self, other: "ContinuousValueMatcher") -> None:
        # pylint: disable=protected-access
        # Pylint doesn't realize the "client class" whose private members we're accessing is this
        # same class
        if isinstance(other, GaussianContinuousValueMatcher):
            if self._n_observations == 1:
                # Treat our own mean as a single observation (because it is) and calculate the new
                # mean and sum of squares from the other matcher's perspective.
                new_mean, new_sum_squared = self._calculate_new_values(
                    other._mean,
                    other._sum_of_squared_differences,
                    other._n_observations,
                    self._mean,
                )
                self._mean = new_mean
                self._sum_of_squared_differences = new_sum_squared
                self._n_observations += other._n_observations
            elif other._n_observations == 1:
                self.update_on_observation(other.mean)
            else:
                raise ValueError(
                    f"Cannot merge two matchers that both have multiple observations (self with "
                    f"{self._n_observations} and other with {other._n_observations})."
                )
        else:
            raise ValueError(
                f"Cannot merge {type(self)} with matcher of foreign type {type(other)}"
            )


@attrs
class MultidimensionalGaussianContinuousValueMatcher(GaussianContinuousValueMatcher):
    """
    Extend Gaussian continuous value matcher to >1 dimensional data
    """
    _root_coordinates: Iterable[float] = attrib(validator = instance_of(tuple), init=True)
    _mean: float = attrib(validator=instance_of(float), init=False, default=0.0)

    @staticmethod
    def from_observation(point: Iterable[float]) -> "MultidimensionalGaussianContinuousValueMatcher":
        """
        Return a new Multidimensional Gaussian continuous matcher created from the given single point.

        This exists for clarity more than anything.
        """
        return MultidimensionalGaussianContinuousValueMatcher(point)


    def match_score(self, point: Iterable[float]) -> float:
        """
        Return a score representing how closely the given value matches this distribution.

        This score should fall into the interval [0, 1] so that learners can threshold scores
        consistently across different matcher types.
        """
        standard_deviation = sqrt(self.sample_variance)
        value = dist(self._root_coordinates, point)
        return (
            1.0
            if standard_deviation == 0 and value == self._mean
            else 0.0 if standard_deviation == 0 else
            2.0 * norm.cdf(
                self._mean - abs(value - self._mean),
                loc=self._mean,
                scale=standard_deviation,
            )
        )

    def update_on_observation(self, point: Iterable[float]) -> None:
        """
        Update the matcher's distribution to account for the given value.
        """
        value = dist(self._root_coordinates, point)
        # With some help from Wikipedia. :)
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        new_mean, new_sum_squared = self._calculate_new_values(
            self._mean, self._sum_of_squared_differences, self._n_observations, value
        )

        self._mean = new_mean
        self._sum_of_squared_differences = new_sum_squared
        self._n_observations += 1

    def merge(self, other: "ContinuousValueMatcher") -> None:
        # pylint: disable=protected-access
        # Pylint doesn't realize the "client class" whose private members we're accessing is this
        # same class
        if isinstance(other, MultidimensionalGaussianContinuousValueMatcher):
            if self._n_observations == 1:
                # Treat our own mean as a single observation (because it is) and calculate the new
                # mean and sum of squares from the other matcher's perspective.
                new_mean, new_sum_squared = self._calculate_new_values(
                    other._mean,
                    other._sum_of_squared_differences,
                    other._n_observations,
                    dist(self._root_coordinates, other._root_coordinates),
                )
                self._mean = new_mean
                self._sum_of_squared_differences = new_sum_squared
                self._n_observations += other._n_observations
                self._root_coordinates = other._root_coordinates
            elif other._n_observations == 1:
                self.update_on_observation(other._root_coordinates)
            else:

                raise ValueError(
                    f"Cannot merge two matchers that both have multiple observations (self with "
                    f"{self._n_observations} and other with {other._n_observations})."
                )
        else:
            raise ValueError(
                f"Cannot merge {type(self)} with matcher of foreign type {type(other)}"
            )
