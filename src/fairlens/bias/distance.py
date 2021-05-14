from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from . import p_value as pv
from . import utils


@dataclass
class DistanceResult:
    distance: float
    p_value: Optional[float] = None


class DistanceMetric:
    """
    Base class for distance metrics that compare samples from two distributions.

    Computes the distance between the probability distributions of group1 and group2 with respect to the
    target attribute. If group1 is a dictionary and group2 is None then the distance is computed between
    group1 and the rest of the dataset.

    Subclasses must implement a distance method.
    """

    def __init__(self, column: pd.Series, group1: pd.Series, group2: pd.Series, **kwargs):
        """Initialize the distributions for computation.

        Args:
            column (pd.Series):
                The column of data with respect to which the distance will be computed.
            group1 (pd.Series):
                The data in the column representing the first group.
            group2 (pd.Series):
                The data in the column representing the second group.
            **kwargs:
                Keyword arguments for the distance metrics.
        """

        self.x = group1
        self.y = group2
        self.xy = column
        self.kwargs = kwargs

    def __call__(self, p_value: bool = False) -> DistanceResult:
        """Calculate the distance between two distributions.

        Args:
            p_value (bool, optional):
                If True, a p value is calculated. By default this uses a permutation test unless the derived class
                overrides the DistanceMetric.p_value method. Defaults to False.

        Returns:
            DistanceResult:
                The calculated result.
        """

        result = DistanceResult(self.distance)
        if p_value:
            result.p_value = self.p_value

        return result

    @property
    @abstractmethod
    def distance(self) -> float:
        """
        Derived classes must implement this.
        """
        ...

    @property
    def p_value(self) -> float:
        """
        Return a p-value for this metric using a permutation test. The null hypothesis
        is that both data samples are from the same distribution.

        Returns:
            The p-value under the null hypothesis.
        """

        return pv.permutation_test(self.x, self.y, self._distance_call)

    def _distance_call(self, x: pd.Series, y: pd.Series) -> float:
        cls = type(self)
        obj = cls(self.xy, x, y, **self.kwargs)
        return obj().distance

    @staticmethod
    @abstractmethod
    def id() -> str:
        """
        A string identifier for the method. Used by fairlens.metrics.stat_distance()
        """
        ...


class CategoricalDistanceMetric(DistanceMetric):
    """
    Base class for distance metrics that compare counts from two binned or categorical distributions.

    Subclasses must implement a distance method.
    """

    def __init__(self, column: pd.Series, group1: pd.Series, group2: pd.Series, **kwargs):
        super().__init__(column, group1, group2, **kwargs)

        # Compute pdfs of the data
        space = self.xy.unique()
        p, q, pq = utils.compute_probabilities(space, self.x, self.y, self.xy)

        self.p = p
        self.q = q
        self.pq = pq

    @property
    def p_value(self) -> float:
        """Returns a two-sided p-value for this metric using a bootstrapped distribution of the null hypothesis."""

        ts_distribution = pv.bootstrap_binned_statistic((self.x, self.y), self._distance_call, n_samples=1000)
        return pv.bootstrap_pvalue(self.distance, ts_distribution)
