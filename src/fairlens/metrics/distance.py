"""
Collection of metrics, tests that measure the distance, or similarity, between two univariate distributions.
"""

import inspect
from abc import ABC, abstractmethod
from typing import Dict, Optional, Type, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, kruskal, ks_2samp, wasserstein_distance

from .. import utils
from ..metrics import significance as pv


class DistanceMetric(ABC):
    """
    Base class for distance metrics that compare samples from two distributions.

    Computes the distance between the probability distributions of x and y with respect to the
    target attribute.

    Subclasses must implement a distance method.
    """

    _class_dict: Dict[str, Type["DistanceMetric"]] = {}

    def __init__(self, **kwargs):
        """Initialize distance metric."""
        ...

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if not inspect.isabstract(cls):
            cls_id = cls._get_id()
            if cls_id:
                cls._class_dict[cls_id] = cls
            else:
                cls._class_dict[cls.__name__] = cls

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
        """Returns a p-value for the test that x and y are sampled from the same distribution.

        Args:
            x (pd.Series):
                Numerical data in a column.
            y (pd.Series):
                Numerical data in a column.

        Returns:
            float:
                The computed p-value.
        """

        raise NotImplementedError()

    @property
    @abstractmethod
    def id(self) -> str:
        """
        A string identifier for the method. Used by fairlens.metrics.stat_distance().
        Derived classes must implement this.
        """
        ...

    @classmethod
    def _get_id(cls) -> str:
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


class MeanDistance(ContinuousDistanceMetric):
    """
    The difference between the means of the two distributions.
    """

    def distance(self, x: pd.Series, y: pd.Series) -> float:
        return abs(x.mean() - y.mean())

    @property
    def id(self) -> str:
        return "mean"


class BinomialDistance(ContinuousDistanceMetric):
    """
    Difference distance between two binary data samples.
    i.e p_x - p_y, where p_x, p_y are the probabilities of success in x and y, respectively.
    The p-value computed is for the null hypothesis is that the probability of success is p_y.
    Data is assumed to be a series of 1, 0 (success, failure) Bernoulli random variates.
    """

    def check_input(self, x: pd.Series, y: pd.Series) -> bool:
        return utils.infer_distr_type(pd.concat((x, y))).is_binary()

    def distance(self, x: pd.Series, y: pd.Series) -> float:
        return x.mean() - y.mean()

    def p_value(self, x: pd.Series, y: pd.Series) -> float:
        p_obs = x.mean()
        p_null = y.mean()
        n = len(x)

        return pv.binominal_proportion_p_value(p_obs, p_null, n)

    @property
    def id(self) -> str:
        return "binomial"


class KolmogorovSmirnovDistance(ContinuousDistanceMetric):
    """
    Kolmogorov-Smirnov (KS) distance between two data samples.
    """

    def distance(self, x: pd.Series, y: pd.Series) -> float:
        return ks_2samp(x, y)[0]

    def p_value(self, x: pd.Series, y: pd.Series) -> float:
        return ks_2samp(x, y)[1]

    @property
    def id(self) -> str:
        return "ks_distance"


class KruskalWallis(ContinuousDistanceMetric):
    def distance(self, x: pd.Series, y: pd.Series) -> float:
        return kruskal(x, y)[0]

    def p_value(self, x: pd.Series, y: pd.Series) -> float:
        return kruskal(x, y)[1]

    @property
    def id(self) -> str:
        return "kruskal"


class EarthMoversDistance(CategoricalDistanceMetric):
    """
    Earth movers distance (EMD), aka Wasserstein 1-distance, for categorical data.

    Using EarthMoversDistance on the raw data is faster and recommended.
    """

    def distance_pdf(self, p: pd.Series, q: pd.Series, bin_edges: Optional[np.ndarray]) -> float:
        p_sum = p.sum()
        q_sum = q.sum()

        if p_sum == 0 and q_sum == 0:
            return 0.0
        elif p_sum == 0 or q_sum == 0:
            return 1.0

        # normalise counts for consistency with scipy.stats.wasserstein
        with np.errstate(divide="ignore", invalid="ignore"):
            p_normalised = np.nan_to_num(p / p_sum).astype(np.float64)
            q_normalised = np.nan_to_num(q / q_sum).astype(np.float64)

        if bin_edges is None:
            # if bins not given, histograms are assumed to be counts of nominal categories,
            # and therefore distances betwen bins are meaningless. Set to all distances to
            # unity to model this.
            distance = 0.5 * np.sum(np.abs(p_normalised - q_normalised))
        else:
            # otherwise, use pair-wise euclidean distances between bin centers for scale data
            bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2.0
            distance = wasserstein_distance(bin_centers, bin_centers, u_weights=p_normalised, v_weights=q_normalised)

        return distance

    @property
    def id(self) -> str:
        return "emd"


class KullbackLeiblerDivergence(CategoricalDistanceMetric):
    """
    Kullback–Leibler Divergence or Relative Entropy between two probability distributions.
    """

    def distance_pdf(self, p: pd.Series, q: pd.Series, bin_edges: Optional[np.ndarray]) -> float:
        return entropy(np.array(p), np.array(q))

    @property
    def id(self) -> str:
        return "kl_divergence"


class JensenShannonDivergence(CategoricalDistanceMetric):
    """
    Jensen-Shannon Divergence between two probability distributions.
    """

    def distance_pdf(self, p: pd.Series, q: pd.Series, bin_edges: Optional[np.ndarray]) -> float:
        return jensenshannon(p, q)

    @property
    def id(self) -> str:
        return "js_divergence"


class Norm(CategoricalDistanceMetric):
    """
    LP Norm between two probability distributions.
    """

    def __init__(self, bin_edges: Optional[np.ndarray] = None, ord: Union[str, int] = 2):
        """
        Args:
            bin_edges (Optional[np.ndarray], optional):
                A list of bin edges used to bin continuous data by or to indicate bins of pre-binned data.
                Defaults to None.
            ord (Union[str, int], optional):
                The order of the norm. Possible values include positive numbers, 'fro', 'nuc'.
                See numpy.linalg.norm for more details. Defaults to 2.
        """

        super().__init__(bin_edges=bin_edges)
        self.ord = ord

    def distance_pdf(self, p: pd.Series, q: pd.Series, bin_edges: Optional[np.ndarray]) -> float:
        return np.linalg.norm(p - q, ord=self.ord)

    @property
    def id(self) -> str:
        return "norm"


class HellingerDistance(CategoricalDistanceMetric):
    """
    Hellinger distance between two probability distributions.
    """

    def distance_pdf(self, p: pd.Series, q: pd.Series, bin_edges: Optional[np.ndarray]) -> float:
        return np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2)

    @property
    def id(self) -> str:
        return "hellinger"
