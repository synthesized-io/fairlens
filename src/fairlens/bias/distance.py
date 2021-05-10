"""
Collection of Metrics that measure the distance, or similarity, between two datasets.
"""
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import pyemd
from scipy.stats import ks_2samp

from . import p_value as pv


@dataclass
class DistanceResult:
    distance: float
    p_value: Optional[float] = None


class DistanceMetric:
    """
    Base class for distance metrics that compare samples from two distributions.

    Subclasses must implement a distance method.
    """

    def __init__(self, x: pd.Series, y: pd.Series, **kwargs):
        self.x = x
        self.y = y

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
        return pv.permutation_test(self.x, self.y, lambda x, y: self._distance_call(x, y))

    def _distance_call(self, x: pd.Series, y: pd.Series) -> float:
        cls = type(self)
        kwargs = {k: v for k, v in self.__dict__.items() if k not in ("x", "y")}
        obj = cls(x, y, **kwargs)
        return obj().distance


class BinnedDistanceMetric(DistanceMetric):
    """
    Base class for distance metrics that compare counts from two binned distributions
    that have identical binning.

    Subclasses must implement a distance method.
    """

    def __init__(self, x: pd.Series, y: pd.Series, bins: Optional[Sequence[Union[float, int]]] = None):
        """
        Args:
            x: A series of histogram counts.
            y: A series of histogram counts.
            bins: Optional; If given, this must be an iterable of bin edges for x and y,
                i.e the output of np.histogram_bin_edges. If None, then it is assumed
                that the data represent counts of nominal categories, with no meaningful
                distance between bins.

        Raises:
            ValueError: x and y do not have the same number of bins.
        """
        super().__init__(x, y)
        self.bins = bins

        if self.x.shape != self.y.shape:
            raise ValueError("x and y must have the same number of bins")

        if self.bins is not None:
            if len(self.bins) != len(self.x) + 1:
                raise ValueError("'bins' does not match shape of input data.")

    @property
    def p_value(self) -> float:
        """
        Return a two-sided p-value for this metric using a bootstrapped distribution
        of the null hypothesis.

        Returns:
            The p-value under the null hypothesis.
        """
        ts_distribution = pv.bootstrap_binned_statistic((self.y, self.y), self._distance_call, n_samples=1000)
        return pv.bootstrap_pvalue(self.distance, ts_distribution)


class BinomialDistance(DistanceMetric):
    """
    Difference distance between two binomal data samples.
    """

    @property
    def distance(self) -> float:
        """
        Calculate the difference distance, i.e p_x - p_y, where p_x
        is the probability of success in sample x and p_y is the
        probablity of success in sample y.

        Data is assumed to be a series of 1, 0 (success, failure) Bernoulli
        random variates.

        Returns:
            Difference between p_x and p_y.
        """
        return self.x.mean() - self.y.mean()

    @property
    def p_value(self) -> float:
        """
        Calculate a p-value for the null hypothesis that the
        probability of success is p_y.

        Returns:
            The p-value under the null hypothesis.
        """
        p_obs = self.x.mean()
        p_null = self.y.mean()
        n = len(self.x)
        return pv.binominal_proportion_p_value(p_obs, p_null, n)


class KolmogorovSmirnovDistance(DistanceMetric):
    """
    Kolmogorov-Smirnov (KS) distance between two data samples.
    """

    @property
    def distance(self) -> float:
        """
        Calculate the KS distance.

        Returns:
            The KS distance.
        """
        return ks_2samp(self.x, self.y)[0]

    @property
    def p_value(self):
        return ks_2samp(self.x, self.y)[1]


class EarthMoversDistanceBinned(BinnedDistanceMetric):
    """
    Earth movers distance (EMD), aka Wasserstein 1-distance, between two histograms.

    The histograms can represent counts of nominal categories or counts on
    an ordinal range. If the latter, they must have equal binning.
    """

    @property
    def distance(self) -> float:
        """
        Calculate the EMD between two 1d histograms.

        Histograms must have an equal number of bins. They are not required to be normalised,
        and distances between bins are measured using a Euclidean metric.

        Returns:
            The earth mover's distance.
        """
        if self.bins is None:
            # if bins not given, histograms are assumed to be counts of nominal categories,
            # and therefore distances betwen bins are meaningless. Set to all distances to
            # unity to model this.
            distance_metric = 1 - np.eye(len(self.x))
        else:
            # otherwise, use pair-wise euclidean distances between bin centers for scale data
            bin_centers = 2 * (self.bins[:-1] + np.diff(self.bins) / 2.0,)
            mgrid = np.meshgrid(*bin_centers)
            distance_metric = np.abs(mgrid[0] - mgrid[1]).astype(np.float64)

        # normalise counts for consistency with scipy.stats.wasserstein
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.nan_to_num(self.x / self.x.sum())
            y = np.nan_to_num(self.y / self.y.sum())

        distance = pyemd.emd(x.astype(np.float64), y.astype(np.float64), distance_metric)
        return distance


class EarthMoversDistance(DistanceMetric):
    """
    Earth movers distance (EMD), aka Wasserstein 1-distance, between samples from two distributions.
    """

    def __init__(self, x: pd.Series, y: pd.Series, emd_kwargs=None):
        """
        Histograms do not have to be normalised, and distances between bins
        are measured using a Euclidean metric.

        Args:
            x: A series representing histogram counts.
            y: A series representing histogram counts.
            kwargs: optional keyword arguments to pyemd.emd_samples.
        """
        super().__init__(x, y)
        if emd_kwargs is None:
            emd_kwargs = {}
        self.emd_kwargs = emd_kwargs

    @property
    def distance(self) -> float:
        """
        Calculate the EMD between two 1d histograms.

        Returns:
            The earth mover's distance.
        """
        return pyemd.emd_samples(self.x, self.y, **self.emd_kwargs)


class HellingerDistance(DistanceMetric):
    """
    Hellinger distance between samples from two distributions.

    Samples are binned during the computation to approximate the pdfs P(x) and P(y).
    """

    def __init__(self, x: pd.Series, y: pd.Series, bins: Union[str, int, Sequence[Union[float, int]]] = "auto"):
        super().__init__(x, y)
        self.bins = bins

    @property
    def distance(self) -> float:
        y_hist, bins = np.histogram(self.y, bins=self.bins, density=False)
        x_hist, bins = np.histogram(self.x, bins=bins, density=False)

        return HellingerDistanceBinned(x_hist, y_hist, bins).distance


class HellingerDistanceBinned(BinnedDistanceMetric):
    """
    Hellinger distance between two histograms.
    """

    def __init__(self, x: pd.Series, y: pd.Series, bins: Sequence[Union[float, int]]):
        super().__init__(x, y, bins)
        self.bins = bins

    @property
    def distance(self) -> float:
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.nan_to_num(self.x / self.x.sum()) / np.diff(self.bins)
            y = np.nan_to_num(self.y / self.y.sum()) / np.diff(self.bins)

        return np.sqrt(min(1, abs(1 - np.sum(np.sqrt(x * y) * np.diff(self.bins)))))
