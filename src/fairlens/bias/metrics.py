"""
Collection of Metrics that measure the distance, or similarity, between two datasets.
"""
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import pyemd
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, ks_2samp

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
            Keyword arguments for the distance metric. Passed to the __init__ function of distance metrics.

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
        if group2 is None:
            pred1 = utils.get_predicates_mult(df, [group1])[0]
            pred2 = ~pred1
        else:
            pred1, pred2 = tuple(utils.get_predicates_mult(df, [group1, group2]))

        group1 = df[pred1][target_attr]
        group2 = df[pred2][target_attr]

    if not isinstance(group1, pd.Series) or not isinstance(group2, pd.Series):
        raise IllegalArgumentException()

    DistClass: Optional[Type[DistanceMetric]] = None

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
            id = cl.get_id()
            if id:
                class_map[id] = cl
                valid_modes.append(id)
            else:
                valid_modes.append(cl.__name__)

            # All mappings from class name to class kept for compatibility
            class_map[cl.__name__] = cl

        if mode not in class_map:
            raise ValueError(f"Invalid mode. Valid modes include:\n{valid_modes}")

        DistClass = class_map[mode]

    assert DistClass is not None

    dist_metric = DistClass(**kwargs)

    d = dist_metric(group1, group2)

    if d is None:
        raise IllegalArgumentException("Incompatible data inside series'")

    return d


class BinomialDistance(DistanceMetric):
    """
    Difference distance between two binomal data samples.
    i.e p_x - p_y, where p_x, p_y are the probabilities of success in x and y, respectively.
    The p-value computed is for the null hypothesis is that the probability of success is p_y.
    Data is assumed to be a series of 1, 0 (success, failure) Bernoulli random variates.
    """

    def distance(self, x: pd.Series, y: pd.Series) -> float:

        if (not utils.infer_distr_type(x).is_binary()) or (not utils.infer_distr_type(y).is_binary()):
            raise IllegalArgumentException("BinomialDistance must be passed series' of 1, 0 Bernoulli random variates")

        return x.mean() - y.mean()

    @property
    def id(self) -> str:
        return "binomial"


class EarthMoversDistance(DistanceMetric):
    """
    Earth movers distance (EMD), aka Wasserstein 1-distance, for continous data.
    The samples are binned automatically by pyemd.

    Keyword arguments are passed to pyemd.emd_samples, i.e. extra_mass_penalty, distance, bins
    """

    def distance(self, x: pd.Series, y: pd.Series) -> float:
        return pyemd.emd_samples(x, y, **self.kwargs)

    @property
    def id(self) -> str:
        return "emd"


class KolmogorovSmirnovDistance(DistanceMetric):
    """
    Kolmogorov-Smirnov (KS) distance between two data samples.

    Keyword arguments are passed to scipy.stats.ks_2amp, i.e. alternative, mode
    """

    def distance(self, x: pd.Series, y: pd.Series) -> float:
        return ks_2samp(x, y, **self.kwargs)[0]

    def p_value(self, x: pd.Series, y: pd.Series) -> float:
        return ks_2samp(x, y, **self.kwargs)[1]

    @property
    def id(self) -> str:
        return "ks_distance"


class EarthMoversDistanceCategorical(CategoricalDistanceMetric):
    """
    Earth movers distance (EMD), aka Wasserstein 1-distance, for categorical data.

    Using EarthMoversDistance on the raw data is faster and recommended. Additional kwargs
    will be passed to pyemd.emd, i.e. extra_mass_penalty.
    """

    def distance_pdf(self, p: pd.Series, q: pd.Series, bin_edges: Optional[np.ndarray]) -> float:
        distance_metric = 1 - np.eye(len(p))

        if bin_edges is not None:
            # Use pair-wise euclidean distances between bin centers for scale data
            bin_centers = np.mean([bin_edges[:-1], bin_edges[1:]], axis=0)
            xx, yy = np.meshgrid(bin_centers, bin_centers)
            distance_metric = np.abs(xx - yy)

        p = np.array(p).astype(np.float64)
        q = np.array(q).astype(np.float64)
        distance_metric = distance_metric.astype(np.float64)

        return pyemd.emd(p, q, distance_metric, **self.kwargs)

    @property
    def id(self) -> str:
        return "emd_categorical"


class KullbackLeiblerDivergence(CategoricalDistanceMetric):
    """
    Kullback–Leibler Divergence or Relative Entropy between two probability distributions.

    Keyword arguments are passed to scipy.stats.entropy. ie. base
    """

    def distance_pdf(self, p: pd.Series, q: pd.Series, bin_edges: Optional[np.ndarray]) -> float:
        return entropy(np.array(p), np.array(q), **self.kwargs)

    @property
    def id(self) -> str:
        return "kl_divergence"


class JensenShannonDivergence(CategoricalDistanceMetric):
    """
    Jensen-Shannon Divergence between two probability distributions

    Keyword arguments are passed to scipy.spatial.distance. ie. base
    """

    def distance_pdf(self, p: pd.Series, q: pd.Series, bin_edges: Optional[np.ndarray]) -> float:
        return jensenshannon(p, q, **self.kwargs)

    @property
    def id(self) -> str:
        return "js_divergence"


class LNorm(CategoricalDistanceMetric):
    """
    LP Norm between two probability distributions.

    Keyword arguments are passed to np.linalg.norm. ie. ord
    """

    def distance_pdf(self, p: pd.Series, q: pd.Series, bin_edges: Optional[np.ndarray]) -> float:
        return np.linalg.norm(p - q, **self.kwargs)

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
