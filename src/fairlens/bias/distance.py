from abc import abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from . import utils


class DistanceMetric:
    """
    Base class for distance metrics that compare samples from two distributions.

    Computes the distance between the probability distributions of group1 and group2 with respect to the
    target attribute. If group1 is a dictionary and group2 is None then the distance is computed between
    group1 and the rest of the dataset.

    Subclasses must implement a distance method.
    """

    def __init__(self, **kwargs):
        """Initialize distance metric.

        Args:
            kwargs:
                Keyword arguments passed down to distance function.
        """

        self.kwargs = kwargs

    def __call__(self, x: pd.Series, y: pd.Series) -> Optional[float]:
        """Calculate the distance between two distributions.

        Args:
            x (pd.Series):
                The data in the column representing the first group.
            y (pd.Series):
                The data in the column representing the second group.

        Returns:
            Optional[float]:
                The computed distance.
        """

        if not self.check_input(x, y):
            return None

        return self.distance(x, y)

    def check_input(self, x: pd.Series, y: pd.Series) -> bool:
        return x.dtype == y.dtype

    @abstractmethod
    def distance(self, x: pd.Series, y: pd.Series) -> float:
        """
        Derived classes must implement this.
        """
        ...

    def p_value(self, x: pd.Series, y: pd.Series) -> float:
        raise NotImplementedError

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

    The default p-value for this metric uses a bootstrapped distribution of the null hypothesis (two-sided).
    """

    def __init__(self, bin_edges: Optional[np.ndarray] = None, **kwargs):
        """Initialize categorical distance metric.

        Args:
            bin_edges (Optional[np.ndarray], optional):
                A list of bin edges used to bin continuous data by or to indicate bins of pre-binned data.
            kwargs:
                Keyword arguments passed down to distance function.
        """

        self.bin_edges = bin_edges
        self.kwargs = kwargs

    def __call__(self, x: pd.Series, y: pd.Series) -> Optional[float]:
        if not self.check_input(x, y):
            return None

        # Compute pdfs of the data, bin if continuous
        all = pd.concat((x, y))
        if utils.infer_distr_type(all).is_continuous():
            bin_edges = self.bin_edges or np.histogram_bin_edges(all, bins="auto")
            x, _ = np.histogram(x, bins=bin_edges)
            y, _ = np.histogram(y, bins=bin_edges)

            x = pd.Series(x / x.sum())
            y = pd.Series(y / y.sum())
            self.bin_edges = bin_edges
        else:
            space = all.unique()
            x, y = utils.compute_probabilities(space, x, y)

        return self.distance(x, y)

    def p_value(self, x: pd.Series, y: pd.Series) -> float:
        raise NotImplementedError
