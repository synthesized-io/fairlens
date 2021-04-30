from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pyemd import emd as pemd
from scipy.stats import entropy, ks_2samp

from . import utils


def class_imbalance(
    df: pd.DataFrame, target_attr: str, group1: Dict[str, List[Any]], group2: Dict[str, List[str]]
) -> float:
    """Computes the class imbalance between group1 and group2 with respect to the target attribute.

    Args:
        df (pd.DataFrame):
            The input dataframe.
        target_attr (str):
            The target attribute in the dataframe.
        group1 (Dict[str, List[str]]):
            The first group of interest.
        group2 (Dict[str, List[str]]):
            The second group of interest.

    Returns:
        float:
            The class imbalance as a float.

    Examples:
        >>> df = pd.read_csv("datasets/compas.csv")
        >>> class_imbalance(df, 'RawScore', {'Ethnicity': ['African-American']}, {'Ethnicity': ['Caucasian']})
        0.021244309559939303
    """

    pred1, pred2 = utils.get_predicates(df, group1, group2)

    return (df[pred1][target_attr].nunique() - df[pred2][target_attr].nunique()) / df[target_attr].nunique()


def emd(
    df: pd.DataFrame,
    target_attr: str,
    group1: Dict[str, List[Any]],
    group2: Optional[Dict[str, List[str]]] = None,
    g1_counts: Optional[Dict[Any, int]] = None,
    g2_counts: Optional[Dict[Any, int]] = None,
) -> float:
    """Computes the Earth Mover's Distance between the probability distributions of group1 and group2 with
    respect to the target attribute. If group2 is None then the distance is computed between group1 and the
    rest of the dataset. Alternatively precomputed aggregated counts for each of the groups can be provided.

    Args:
        df (pd.DataFrame):
            The input datafame.
        target_attr (str):
            The target attribute in the dataframe.
        group1 (Dict[str, List[str]]):
            The first group of interest.
        group2 (Optional[Dict[str, List[str]]], optional):
            The second group of interest. Defaults to None.
        g1_counts (Optional[Dict[Any, int]], optional):
            Dictionary mapping from value to counts for group1. Defaults to None.
        g2_counts (Optional[Dict[Any, int]], optional):
            Dictionary mapping from value to counts for group2. Defaults to None.

    Returns:
        float:
            The Earth Mover's Distance as a float.

    Examples:
        >>> df = pd.read_csv("datasets/compas.csv")
        >>> emd(df, 'RawScore', {'Ethnicity': ['African-American']}, {'Ethnicity': ['Caucasian']})
        0.15406599999999984
        >>> metrics.emd(df, 'RawScore', {'Ethnicity': ['African-American']})
        0.16237499999999971
    """

    pred1, pred2 = utils.get_predicates(df, group1, group2)

    g1_counts = g1_counts or df[pred1].groupby(target_attr)[target_attr].aggregate(Count="count")["Count"].to_dict()
    g2_counts = g2_counts or df[pred2].groupby(target_attr)[target_attr].aggregate(Count="count")["Count"].to_dict()

    space = df[target_attr].unique()

    p = np.zeros(len(space))
    q = np.zeros(len(space))
    for i, val in enumerate(space):
        p[i] += g1_counts.get(val, 0)
        q[i] += g2_counts.get(val, 0)

    p /= p.sum()
    q /= q.sum()

    xx, yy = np.meshgrid(space, space)
    distance_space = np.abs(xx - yy)

    return pemd(p, q, distance_space)


def ks_distance(
    df: pd.DataFrame, target_attr: str, group1: Dict[str, List[str]], group2: Optional[Dict[str, List[str]]] = None
) -> Tuple[float, float]:
    """Performs the Kolmogorov–Smirnov test between the probability distributions of group1 and group2 with
    respect to the target attribute. If group2 is None then the distance is computed between group1 and the
    rest of the dataset. Returns the Kolmogorov–Smirnov statistical distance and the associated p-value.

    Args:
        df (pd.DataFrame):
            The input dataframe.
        target_attr (str):
            The target attribute in the dataframe.
        group1 (Dict[str, List[str]]):
            The first group of interest.
        group2 (Optional[Dict[str, List[str]]], optional):
            The second group of interest. Defaults to None.

    Returns:
        Tuple[float, float]:
            A tuple containing the Kolmogorov–Smirnov statistic

    Examples:
        >>> df = pd.read_csv("datasets/compas.csv")
        >>> ks_distance(df, 'RawScore', {'Sex': ['Male']}, {'Sex': ['Female']})
        (0.0574598854742705, 2.585181602569765e-30)
        >>> ks_distance(df, 'RawScore', {'FirstName': ['Stephanie']}, {'FirstName': ['Kevin']})
        (0.0744047619047619, 0.8522462138629425)
    """

    pred1, pred2 = utils.get_predicates(df, group1, group2)

    statistic, pval = ks_2samp(df[pred1][target_attr], df[pred2][target_attr])

    return statistic, pval


def kl_divergence(
    df: pd.DataFrame,
    target_attr: str,
    group1: Dict[str, List[str]],
    group2: Optional[Dict[str, List[str]]] = None,
    g1_counts: Optional[Dict[Any, int]] = None,
    g2_counts: Optional[Dict[Any, int]] = None,
) -> float:
    """Computes the Kullback–Leibler Divergence or Relative Entropy between the probability distributions of the
    two groups with respect to the target attribute. If group2 is None then the distance is computed between
    group1 and the rest of the dataset. Alternatively precomputed aggregated counts for each of the groups
    can be provided.

    Args:
        df (pd.DataFrame):
            The input dataframe.
        target_attr (str):
            The target attribute in the dataframe.
        group1 (Dict[str, List[str]]):
            The first group of interest.
        group2 (Optional[Dict[str, List[str]]], optional):
            The first group of interest. Defaults to None.
        g1_counts (Optional[Dict[Any, int]], optional):
            Dictionary mapping from value to counts for group1. Defaults to None.
        g2_counts (Optional[Dict[Any, int]], optional):
            Dictionary mapping from value to counts for group2. Defaults to None.

    Returns:
        float:
            The entropy as a float.
    """

    pred1, pred2 = utils.get_predicates(df, group1, group2)

    g1_counts = g1_counts or df[pred1].groupby(target_attr)[target_attr].aggregate(Count="count")["Count"].to_dict()
    g2_counts = g2_counts or df[pred2].groupby(target_attr)[target_attr].aggregate(Count="count")["Count"].to_dict()

    space = df[target_attr].unique()

    p = np.zeros(len(space))
    q = np.zeros(len(space))
    for i, val in enumerate(space):
        p[i] += g1_counts.get(val, 0)
        q[i] += g2_counts.get(val, 0)

    p /= p.sum()
    q /= q.sum()

    return entropy(p, q)


def js_divergence(
    df: pd.DataFrame,
    target_attr: str,
    group1: Dict[str, List[str]],
    group2: Optional[Dict[str, List[str]]] = None,
    g1_counts: Optional[Dict[Any, int]] = None,
    g2_counts: Optional[Dict[Any, int]] = None,
    total_counts: Optional[Dict[Any, int]] = None,
) -> float:
    """Computes the Jensen-Shannon Divergence between the probability distributions of the two groups with respect
    to the target attribute. If group2 is None then the distance is computed between group1 and the rest of the
    dataset. Alternatively precomputed aggregated counts for each of the groups can be provided.

    Args:
        df (pd.DataFrame):
            The input dataframe.
        target_attr (str):
            The target attribute in the dataframe.
        group1 (Dict[str, List[str]]):
            The first group of interest.
        group2 (Optional[Dict[str, List[str]]], optional):
            The second group of interest. Defaults to None.
        g1_counts (Optional[Dict[Any, int]], optional):
            Dictionary mapping from value to counts for group1. Defaults to None.
        g2_counts (Optional[Dict[Any, int]], optional):
            Dictionary mapping from value to counts for group2. Defaults to None.
        total_counts (Optional[Dict[Any, int]], optional):
            Dictionary mapping from value to counts for all unique values in the target column.
            Defaults to None.

    Returns:
        float:
            The entropy as a float.
    """

    pred1, pred2 = utils.get_predicates(df, group1, group2)

    g1_counts = g1_counts or df[pred1].groupby(target_attr)[target_attr].aggregate(Count="count")["Count"].to_dict()
    g2_counts = g2_counts or df[pred2].groupby(target_attr)[target_attr].aggregate(Count="count")["Count"].to_dict()
    total_counts = total_counts or df.groupby(target_attr)[target_attr].aggregate(Count="count")["Count"].to_dict()

    space = df[target_attr].unique()

    p = np.zeros(len(space))
    q = np.zeros(len(space))
    pq = np.zeros(len(space))
    for i, val in enumerate(space):
        p[i] += g1_counts.get(val, 0)
        q[i] += g2_counts.get(val, 0)
        pq[i] += total_counts.get(val, 0)

    p /= p.sum()
    q /= q.sum()
    pq /= pq.sum()

    return (entropy(p, pq) + entropy(q, pq)) / 2


def lp_norm(
    df: pd.DataFrame,
    target_attr: str,
    group1: Dict[str, List[str]],
    group2: Optional[Dict[str, List[str]]] = None,
    order: Union[int, str] = 2,
    g1_counts: Optional[Dict[Any, int]] = None,
    g2_counts: Optional[Dict[Any, int]] = None,
) -> float:
    """Computes the LP Norm between the probability distributions of the two groups with respect to the
    target attribute. If group2 is None then the distance is computed between group1 and the rest of the
    dataset. Alternatively precomputed aggregated counts for each of the groups can be provided.

    Args:
        df (pd.DataFrame):
            The input dataframe.
        target_attr (str):
            The target attribute in the dataframe.
        group1 (Dict[str, List[str]]):
            The first group of interest.
        group2 (Optional[Dict[str, List[str]]], optional):
            The second group of interest. Defaults to None.
        order (Union[int, str], optional):
            The order of the norm (p). Passed as 'ord' to numpy.linalg.norm. Defaults to 2.
        g1_counts (Optional[Dict[Any, int]], optional):
            Dictionary mapping from value to counts for group1. Defaults to None.
        g2_counts (Optional[Dict[Any, int]], optional):
            Dictionary mapping from value to counts for group2. Defaults to None.

    Returns:
        float:
            The norm as a float.
    """

    pred1, pred2 = utils.get_predicates(df, group1, group2)

    g1_counts = g1_counts or df[pred1].groupby(target_attr)[target_attr].aggregate(Count="count")["Count"].to_dict()
    g2_counts = g2_counts or df[pred2].groupby(target_attr)[target_attr].aggregate(Count="count")["Count"].to_dict()

    space = df[target_attr].unique()

    p = np.zeros(len(space))
    q = np.zeros(len(space))
    for i, val in enumerate(space):
        p[i] += g1_counts.get(val, 0)
        q[i] += g2_counts.get(val, 0)

    p /= p.sum()
    q /= q.sum()

    return np.linalg.norm(p - q, ord=order)
