"""
Collection of metrics, tests that measure the distance, or similarity, between two univariate distributions.
"""

import inspect
from abc import ABC, abstractmethod
from typing import Dict, Optional, Type, Union

import numpy as np
import pandas as pd
import pyemd
from synthesized_insight.metrics import HellingerDistance as HD
from synthesized_insight.metrics import JensenShannonDivergence as JSD
from synthesized_insight.metrics import KolmogorovSmirnovDistanceTest, KruskalWallisTest
from synthesized_insight.metrics import KullbackLeiblerDivergence as KLD

from .. import utils


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
                The data in the first sample.
            y (pd.Series):
                The data in the second sample.

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
                The data in the first sample.
            y (pd.Series):
                The data in the second sample.

        Returns:
            bool:
                Whether or not the input is valid.
        """
        ...

    @abstractmethod
    def distance(self, x: pd.Series, y: pd.Series) -> float:
        """Distance between the distributions in x and y. Derived classes must implement this.

        Args:
            x (pd.Series):
                The data in the first sample.
            y (pd.Series):
                The data in the second sample.

        Returns:
            float:
                The computed distance.
        """
        ...

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

    def check_input(self, x: pd.Series, y: pd.Series) -> bool:
        x_dtype = utils.infer_dtype(x).dtype
        y_dtype = utils.infer_dtype(y).dtype

        return x_dtype in ["int64", "float64"] and y_dtype in ["int64", "float64"]


class CategoricalDistanceMetric(DistanceMetric):
    """
    Base class for distance metrics on categorical data.

    Subclasses must implement a distance method.
    """

    def check_input(self, x: pd.Series, y: pd.Series) -> bool:
        x_dtype = utils.infer_dtype(x).dtype
        y_dtype = utils.infer_dtype(y).dtype

        return x_dtype == y_dtype


class BinaryDistanceMetric(DistanceMetric):
    """
    Base class for distance metrics on binary data.

    Subclasses must implement a distance method.
    """

    def check_input(self, x: pd.Series, y: pd.Series) -> bool:
        joint = pd.concat((x, y))
        return utils.infer_distr_type(joint).is_binary() and (np.sort(joint.unique()) == [0, 1]).all()


class MeanDistance(ContinuousDistanceMetric):
    """
    The difference between the means of the two distributions.
    """

    def distance(self, x: pd.Series, y: pd.Series) -> float:
        return abs(x.mean() - y.mean())

    @property
    def id(self) -> str:
        return "mean"


class BinomialDistance(BinaryDistanceMetric):
    """
    Difference distance between two binary data samples.
    i.e p_x - p_y, where p_x, p_y are the probabilities of success in x and y, respectively.

    Data is assumed to be a series of 1, 0 (success, failure) Bernoulli random variates.
    """

    def check_input(self, x: pd.Series, y: pd.Series) -> bool:
        return utils.infer_distr_type(pd.concat((x, y))).is_binary()

    def distance(self, x: pd.Series, y: pd.Series) -> float:
        return x.mean() - y.mean()

    @property
    def id(self) -> str:
        return "binomial"


class KolmogorovSmirnovDistance(ContinuousDistanceMetric):
    """
    Kolmogorov-Smirnov (KS) distance between two data samples.
    """

    def distance(self, x: pd.Series, y: pd.Series) -> float:
        return KolmogorovSmirnovDistanceTest()._compute_test(x, y)[0]

    @property
    def id(self) -> str:
        return "ks_distance"


class KruskalWallis(ContinuousDistanceMetric):
    """
    Kruskal Wallis H test between two data samples.
    """

    def distance(self, x: pd.Series, y: pd.Series) -> float:
        return KruskalWallisTest()._compute_test(x, y)[0]

    @property
    def id(self) -> str:
        return "kruskal"


class EarthMoversDistance(CategoricalDistanceMetric):
    """
    Earth movers distance (EMD), aka Wasserstein 1-distance, for categorical data.
    """

    def __init__(self, bin_edges: Optional[np.ndarray] = None):
        """
        Args:
            bin_edges (Optional[np.ndarray], optional):
                A list of bin edges used to bin continuous data by or to indicate bins of pre-binned data.
                Defaults to None.
        """

        self.bin_edges = bin_edges

    def distance(self, x: pd.Series, y: pd.Series) -> float:
        (p, q), bin_edges = utils.zipped_hist((x, y), bin_edges=self.bin_edges, ret_bins=True)

        distance_matrix = 1 - np.eye(len(p))

        if bin_edges is not None:
            # Use pair-wise euclidean distances between bin centers for scale data
            bin_centers = np.mean([bin_edges[:-1], bin_edges[1:]], axis=0)
            xx, yy = np.meshgrid(bin_centers, bin_centers)
            distance_matrix = np.abs(xx - yy)

        p = np.array(p).astype(np.float64)
        q = np.array(q).astype(np.float64)
        distance_matrix = distance_matrix.astype(np.float64)

        return pyemd.emd(p, q, distance_matrix)

    @property
    def id(self) -> str:
        return "emd"


class KullbackLeiblerDivergence(CategoricalDistanceMetric):
    """
    Kullback–Leibler Divergence or Relative Entropy between two probability distributions.
    """

    def distance(self, x: pd.Series, y: pd.Series) -> float:
        return KLD()._compute_metric(x, y)

    @property
    def id(self) -> str:
        return "kl_divergence"


class JensenShannonDivergence(CategoricalDistanceMetric):
    """
    Jensen-Shannon Divergence between two probability distributions.
    """

    def distance(self, x: pd.Series, y: pd.Series) -> float:
        return JSD()._compute_metric(x, y)

    @property
    def id(self) -> str:
        return "js_divergence"


class Norm(CategoricalDistanceMetric):
    """
    L-P Norm between two probability distributions.
    """

    def __init__(self, ord: Union[str, int] = 2):
        """
        Args:
            ord (Union[str, int], optional):
                The order of the norm. Possible values include positive numbers, 'fro', 'nuc'.
                See numpy.linalg.norm for more details. Defaults to 2.
        """

        self.ord = ord

    def distance(self, x: pd.Series, y: pd.Series) -> float:
        (p, q), _ = utils.zipped_hist((x, y), ret_bins=True)
        return np.linalg.norm(p - q, ord=self.ord)

    @property
    def id(self) -> str:
        return "norm"


class HellingerDistance(CategoricalDistanceMetric):
    """
    Hellinger distance between two probability distributions.
    """

    def distance(self, x: pd.Series, y: pd.Series) -> float:
        return HD()._compute_metric(x, y)

    @property
    def id(self) -> str:
        return "hellinger"
