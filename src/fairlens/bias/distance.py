import inspect
from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

import numpy as np
import pandas as pd

from . import p_value as pv
from . import utils


class DistanceMetric(ABC):
    """
    Base class for distance metrics that compare samples from two distributions.

    Computes the distance between the probability distributions of x and y with respect to the
    target attribute.

    Subclasses must implement a distance method.
    """

    class_dict: Dict[str, Type["DistanceMetric"]] = {}

    def __init__(self, **kwargs):
        ...

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if not inspect.isabstract(cls):
            cls_id = cls.get_id()
            if cls_id:
                cls.class_dict[cls_id] = cls
            else:
                cls.class_dict[cls.__name__] = cls

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

    @abstractmethod
    def check_input(self, x: pd.Series, y: pd.Series) -> bool:
        """Check whether the input is valid. Returns False if x and y have different dtypes by default.

        Args:
            x (pd.Series):
                The data in the column representing the first group.
            y (pd.Series):
                The data in the column representing the second group.

        Returns:
            bool:
                Whether or not the input is valid.
        """
        ...

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


class ContinuousDistanceMetric(DistanceMetric):
    """
    Base class for distance metrics on continuous data.

    Subclasses must implement a distance method.
    """

    def __init__(self, p_value_test="bootstrap"):
        """Initialize continuous distance metric.

        Args:
            p_value_test (str, optional):
                Choose which method of resampling will be used to compute the p-value. Overidden by metrics
                such as Kolmogrov Smirnov Distance.
                Defaults to "permutation".
        """

        self.p_value_test = p_value_test

    def check_input(self, x: pd.Series, y: pd.Series) -> bool:
        x_dtype = utils.infer_dtype(x).dtype
        y_dtype = utils.infer_dtype(y).dtype

        return x_dtype in ["int64", "float64"] and y_dtype in ["int64", "float64"]

    def p_value(self, x: pd.Series, y: pd.Series) -> float:
        if self.p_value_test == "permutation":
            ts_distribution = pv.permutation_statistic(x, y, self.distance, n_perm=100)
        elif self.p_value_test == "bootstrap":
            ts_distribution = pv.bootstrap_statistic(x, y, self.distance, n_samples=1000)
        else:
            raise ValueError('p_value_test must be one of ["permutation", "bootstrap"]')

        return pv.resampling_p_value(self.distance(x, y), ts_distribution)


class CategoricalDistanceMetric(DistanceMetric):
    """
    Base class for distance metrics on categorical data.

    Continuous data is automatically binned to create histograms, bin edges can be provided as an argument
    and will be used to bin continous data. If the data has been pre-binned and consists of pd.Intervals
    for instance, the histograms will be computed using the counts of each bin, and the bin_edges, if given,
    will be used in metrics such as EarthMoversDistanceCategorical to compute the distance space.

    Subclasses must implement a distance_pdf method.
    """

    def __init__(self, bin_edges: Optional[np.ndarray] = None):
        """Initialize categorical distance metric.

        Args:
            bin_edges (Optional[np.ndarray], optional):
                A numpy array of bin edges used to bin continuous data or to indicate bins of pre-binned data
                to metrics which take the distance space into account.
                i.e. For bins [0-5, 5-10, 10-15, 15-20], bin_edges would be [0, 5, 10, 15, 20].
                See numpy.histogram_bin_edges() for more information.
        """

        self.bin_edges = bin_edges

    def check_input(self, x: pd.Series, y: pd.Series) -> bool:
        x_dtype = utils.infer_dtype(x).dtype
        y_dtype = utils.infer_dtype(y).dtype

        return x_dtype == y_dtype

    def distance(self, x: pd.Series, y: pd.Series) -> float:
        (p, q), bin_edges = utils.zipped_hist((x, y), bin_edges=self.bin_edges, ret_bins=True)

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
        (h_x, h_y), bin_edges = utils.zipped_hist((x, y), bin_edges=self.bin_edges, normalize=False, ret_bins=True)

        def distance_call(h_x, h_y):
            with np.errstate(divide="ignore", invalid="ignore"):
                p = pd.Series(np.nan_to_num(h_x / h_x.sum()))
                q = pd.Series(np.nan_to_num(h_y / h_y.sum()))

            return self.distance_pdf(p, q, bin_edges)

        ts_distribution = pv.bootstrap_binned_statistic(h_x, h_y, distance_call, n_samples=100)

        return pv.resampling_p_value(distance_call(h_x, h_y), ts_distribution)
