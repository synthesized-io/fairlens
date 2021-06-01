from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from . import utils


class DistanceMetric(ABC):
    """
    Base class for distance metrics that compare samples from two distributions.

    Computes the distance between the probability distributions of x and y with respect to the
    target attribute.

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
        """Check whether the input is valid. Returns `False` if data isn't numeric by default.

        Args:
            x (pd.Series):
                The data in the column representing the first group.
            y (pd.Series):
                The data in the column representing the second group.

        Returns:
            bool:
                Whether or not the input is valid.
        """

        x_dtype = utils.infer_dtype(x).dtype
        y_dtype = utils.infer_dtype(y).dtype

        return x_dtype in ["int64", "float64"] and y_dtype in ["int64", "float64"]

    @abstractmethod
    def distance(self, x: pd.Series, y: pd.Series) -> float:
        """Distance between the distribution of numerical data in x and y. Derived classes must implement this.

        Args:
            x (pd.Series):
                Numerical data in a column.
            y (pd.Series):
                Numerical data in a column.

        Returns:
            float:
                The computed distance.
        """
        ...

    def p_value(self, x: pd.Series, y: pd.Series) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def id(self) -> str:
        """
        A string identifier for the method. Used by fairlens.metrics.stat_distance().
        Derived classes must implement this.
        """
        ...

    @classmethod
    def get_id(cls) -> str:
        return cls().id


class CategoricalDistanceMetric(DistanceMetric):
    """
    Base class for distance metrics on categorical data.

    Continuous data is automatically binned to create histograms, bin edges can be provided as an argument
    and will be used to bin continous data. If the data has been pre-binned and consists of pd.Intervals
    for instance, the histograms will be computed using the counts of each bin, and the bin_edges, if given,
    will be used in metrics such as EarthMoversDistanceCategorical to compute the distance space.

    Subclasses must implement a distance_pdf method.
    """

    def __init__(self, bin_edges: Optional[np.ndarray] = None, auto_bin: bool = True, **kwargs):
        """Initialize categorical distance metric.

        Args:
            bin_edges (Optional[np.ndarray], optional):
                A list of bin edges used to bin continuous data by or to indicate bins of pre-binned data.
            auto_bin (bool, optional):
                If set to False, forces continuous data to be treated as categorical and does not automatically
                bin using np.histogram(). Defaults to True.
            kwargs:
                Keyword arguments passed down to distance function.
        """

        self.bin_edges = bin_edges
        self.auto_bin = auto_bin
        self.kwargs = kwargs

    def check_input(self, x: pd.Series, y: pd.Series) -> bool:
        x_dtype = utils.infer_dtype(x).dtype
        y_dtype = utils.infer_dtype(y).dtype

        return x_dtype == y_dtype

    def distance(self, x: pd.Series, y: pd.Series) -> float:
        joint = pd.concat((x, y))
        bin_edges = None

        # Compute histograms of the data, bin if continuous and auto_bin set
        if utils.infer_distr_type(joint).is_continuous() and self.auto_bin:
            bin_edges = self.bin_edges or np.histogram_bin_edges(joint, bins="auto")
            p, _ = np.histogram(x, bins=bin_edges)
            q, _ = np.histogram(y, bins=bin_edges)

        else:
            space = joint.unique()
            x_counts = x.value_counts().to_dict()
            y_counts = y.value_counts().to_dict()

            p = np.zeros(len(space))
            q = np.zeros(len(space))
            for i, val in enumerate(space):
                p[i] = x_counts.get(val, 0)
                q[i] = y_counts.get(val, 0)

        # Normalize the histograms
        with np.errstate(divide="ignore", invalid="ignore"):
            p = pd.Series(np.nan_to_num(p / p.sum()))
            q = pd.Series(np.nan_to_num(q / q.sum()))

        return self.distance_pdf(p, q, bin_edges)

    @abstractmethod
    def distance_pdf(self, p: pd.Series, q: pd.Series, bin_edges: Optional[np.ndarray]) -> float:
        """Distance between 2 aligned normalized histograms. Derived classes must implement this.

        Args:
            p (pd.Series):
                A normalized histogram.
            q (pd.Series):
                A normalized histogram.
            bin_edges (Optional[np.ndarray]):
                bin_edges for binned continuous data. Used by metrics such as Earth Mover's Distance to compute the
                distance metric space.

        Returns:
            float:
                The computed distance.
        """
        ...

    def p_value(self, x: pd.Series, y: pd.Series) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def id(self) -> str:
        """
        A string identifier for the method. Used by fairlens.metrics.stat_distance()
        """
        ...
