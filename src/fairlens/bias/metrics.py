"""
Collection of Metrics that measure the distance, or similarity, between two datasets.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyemd
from scipy.stats import entropy, ks_2samp

from . import p_value as pv
from . import utils
from .distance import CategoricalDistanceMetric, DistanceMetric
from .exceptions import IllegalArgumentException


def stat_distance(
    df: pd.DataFrame,
    target_attr: str,
    group1: Union[Dict[str, List[str]], pd.Series],
    group2: Union[Optional[Dict[str, List[str]]], pd.Series] = None,
    mode: str = "auto",
    p_value: bool = False,
    **kwargs,
) -> Union[float, Tuple[float, float]]:
    """Computes the statistical distance between two probability distributions ie. group 1 and group 2, with respect
    to the target attribute. The distance metric can be chosen through the mode parameter. If mode is set to "auto",
    the most suitable metric depending on the target attributes' distribution is chosen.

    If group1 is a dictionary and group2 is None then the distance is computed between group1 and the rest of the
    dataset.

    Args:
        df (pd.DataFrame):
            The input datafame.
        target_attr (str):
            The target attribute in the dataframe.
        group1 (Union[Dict[str, List[str]], pd.Series]):
            The first group of interest. Can be a dictionary mapping from attribute to values
            or the raw data in a pandas series.
        group2 (Union[Optional[Dict[str, List[str]]], pd.Series]], optional):
            The second group of interest. Can be a dictionary mapping from attribute to values
            or the raw data in a pandas series. Defaults to None.
        mode (str):
            Which distance metric to use. Can be the names of classes from fairlens.bias.metrics, or their
            __repr__() strings. If set to "auto", it automatically picks the best metric based on the
            distribution of the target attribute. Defaults to "auto".
        p_value (bool):
            Returns the a suitable p-value for the metric if it exists. Defaults to False.
        **kwargs:
            Keyword arguments for the distance metric.

    Returns:
        Union[float, Tuple[float, float]]:
            The distance as a float, and the p-value if p_value is set to True and can be computed.

    Examples:
        >>> df = pd.read_csv("datasets/compas.csv")
        >>> group1 = {"Ethnicity": ["African-American", "African-Am"]}
        >>> group2 = {"Ethnicity": ["Caucasian"]}
        >>> group3 = {"Ethnicity": ["Asian"]}
        >>> stat_distance(df, "RawScore", group1, group2, mode="auto")
        0.1133214633580949
        >>> stat_distance(df, "RawScore", group3, group2, mode="auto", p_value=True)
        (0.0816143577815524, 0.02693435054772131)
    """

    # Parse group arguments into pandas series'
    if isinstance(group1, dict) and (isinstance(group2, dict) or group2 is None):
        pred1, pred2 = utils.get_predicates(df, group1, group2)

        group1 = df[pred1][target_attr]
        group2 = df[pred2][target_attr]

    if not isinstance(group1, pd.Series) or not isinstance(group2, pd.Series):
        raise IllegalArgumentException()

    DistClass: Any = None

    # Choose statistical distance metric
    if mode == "auto":
        distr_type = utils.infer_distr_type(df[target_attr])
        if distr_type.is_continuous():
            DistClass = KolmogorovSmirnovDistance
        elif distr_type.is_binary():
            DistClass = BinomialDistance
        else:
            DistClass = EarthMoversDistanceCategorical

    else:
        class_map = {}
        valid_modes = []
        for cl in utils.get_all_subclasses(DistanceMetric):
            if cl.id():
                class_map[cl.id()] = cl
                valid_modes.append(cl.id())
            else:
                valid_modes.append(cl.__name__)

            # All mappings from class name to class kept for compatibility
            class_map[cl.__name__] = cl

        if mode not in class_map:
            raise ValueError(f"Invalid mode. Valid modes include:\n{valid_modes}")

        DistClass = class_map[mode]

    dist_metric = DistClass(df[target_attr], group1, group2, **kwargs)

    return dist_metric(p_value=p_value)


class ClassImbalance(DistanceMetric):
    """
    Class imbalance between the number samples in both distributions.
    """

    @property
    def distance(self) -> float:
        return (self.x.nunique() - self.y.nunique()) / self.xy.nunique()

    @staticmethod
    def id():
        return "class_imbalance"


class BinomialDistance(DistanceMetric):
    """
    Difference distance between two binomal data samples.
    i.e p_x - p_y, where p_x, p_y are the probabilities of success in x and y, respectively.
    The p-value computed is for the null hypothesis is that the probability of success is p_y.
    Data is assumed to be a series of 1, 0 (success, failure) Bernoulli random variates.
    """

    @property
    def distance(self) -> float:
        return self.x.mean() - self.y.mean()

    @property
    def p_value(self) -> float:
        p_obs = self.x.mean()
        p_null = self.y.mean()
        n = len(self.x)
        return pv.binominal_proportion_p_value(p_obs, p_null, n)

    @staticmethod
    def id():
        return "binomial"


class EarthMoversDistance(DistanceMetric):
    """
    Earth movers distance (EMD), aka Wasserstein 1-distance, for continous data.
    The samples are binned automatically by pyemd.

    Keyword arguments are passed to pyemd.emd_samples. ie. extra_mass_penalty, distance
    """

    @property
    def distance(self) -> float:
        return pyemd.emd_samples(self.x, self.y, **self.kwargs)

    @staticmethod
    def id():
        return "emd"


class EarthMoversDistanceCategorical(CategoricalDistanceMetric):
    """
    Earth movers distance (EMD), aka Wasserstein 1-distance, for categorical data.

    Bins can be included as a keyword argument 'bins' for pre-binned continous data, however
    using EarthMoversDistance on the raw data is faster and recommended.
    """

    @property
    def distance(self) -> float:
        distance_metric = 1 - np.eye(len(self.pq))

        if "bins" in self.kwargs:
            # Use pair-wise euclidean distances between bin centers for scale data
            bins = self.kwargs["bins"]
            bin_centers = 2 * (bins[:-1] + np.diff(bins) / 2.0,)

            xx, yy = np.meshgrid(*bin_centers)
            distance_metric = np.abs(xx - yy).astype(np.float64)

        return pyemd.emd(self.p, self.q, distance_metric)

    @staticmethod
    def id():
        return "emd_categorical"


class KolmogorovSmirnovDistance(DistanceMetric):
    """
    Kolmogorov-Smirnov (KS) distance between two data samples.

    Keyword arguments are passed to scipy.stats.ks_2amp. ie. alternative, mode
    """

    @property
    def distance(self) -> float:
        return ks_2samp(self.x, self.y, **self.kwargs)[0]

    @property
    def p_value(self):
        return ks_2samp(self.x, self.y, **self.kwargs)[1]

    @staticmethod
    def id():
        return "ks_distance"


class KullbackLeiblerDivergence(CategoricalDistanceMetric):
    """
    Kullback–Leibler Divergence or Relative Entropy between the probability distributions of the
    two groups with respect to the target attribute.
    """

    @property
    def distance(self) -> float:
        return entropy(self.p, self.q)

    @staticmethod
    def id():
        return "kl_divergence"


class JensenShannonDivergence(CategoricalDistanceMetric):
    """
    Jensen-Shannon Divergence or Relative Entropy between the probability distributions of the
    two groups with respect to the target attribute.
    """

    @property
    def distance(self) -> float:
        return (entropy(self.p, self.pq) + entropy(self.q, self.pq)) / 2

    @staticmethod
    def id():
        return "js_divergence"


class LNorm(CategoricalDistanceMetric):
    """
    LP Norm between the probability distributions of the two groups with respect to the target attribute.

    Keyword arguments are passed to np.linalg.norm. ie. ord
    """

    @property
    def distance(self) -> float:
        return np.linalg.norm(self.p - self.q)

    @staticmethod
    def id():
        return "norm"


class HellingerDistance(CategoricalDistanceMetric):
    """
    Hellinger distance between the two distributions.
    """

    @property
    def distance(self) -> float:
        return np.linalg.norm(np.sqrt(self.p) - np.sqrt(self.q)) / np.sqrt(2)

    @staticmethod
    def id():
        return "hellinger"
