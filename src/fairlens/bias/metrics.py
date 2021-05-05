from typing import Dict, Hashable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pyemd import emd as pemd
from scipy.stats import entropy, ks_2samp

from . import utils
from .exceptions import InsufficientParamError


def class_imbalance(
    df: pd.DataFrame, target_attr: str, group1: Dict[str, List[str]], group2: Dict[str, List[str]]
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
    group1: Optional[Dict[str, List[str]]] = None,
    group2: Optional[Dict[str, List[str]]] = None,
    counts: Optional[Tuple[Dict[Hashable, int], Dict[Hashable, int]]] = None,
) -> float:
    """Computes the Earth Mover's Distance between the probability distributions of group1 and group2 with
    respect to the target attribute. If group2 is None then the distance computed is between group1 and the
    remaining data points. Alternatively precomputed aggregated counts for each of the groups can be provided.

    Args:
        df (pd.DataFrame):
            The input datafame.
        target_attr (str):
            The target attribute in the dataframe.
        group1 (Optional[Dict[str, List[str]]]):
            The first group of interest. Defaults to None.
        group2 (Optional[Dict[str, List[str]]], optional):
            The second group of interest. Defaults to None.
        counts (Optional[Tuple[Dict[Hashable, int], Dict[Hashable, int]]], optional):
            A tuple containing the counts of the first and second group, each in a dictionary mapping
            from value to counts. These counts can be interpreted as the histograms between which the
            metric will be computed. Overrides group1 and group2. Defaults to None.

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

    if counts is None:
        if group1 is None:
            raise InsufficientParamError()

        # Find the predicates for the two groups
        pred1, pred2 = utils.get_predicates(df, group1, group2)

        # Compute the histogram / counts for each group
        g1_counts = df[pred1][target_attr].value_counts().to_dict()
        g2_counts = df[pred2][target_attr].value_counts().to_dict()

        counts = g1_counts, g2_counts

    space = df[target_attr].unique()

    p = utils.align_probabilities(counts[0], space)
    q = utils.align_probabilities(counts[1], space)

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
    group1: Optional[Dict[str, List[str]]] = None,
    group2: Optional[Dict[str, List[str]]] = None,
    counts: Optional[Tuple[Dict[Hashable, int], Dict[Hashable, int]]] = None,
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
        counts (Optional[Tuple[Dict[Hashable, int], Dict[Hashable, int]]], optional):
            A tuple containing the counts of the first and second group, each in a dictionary mapping
            from value to counts. These counts can be interpreted as the histograms between which the
            metric will be computed. Overrides group1 and group2. Defaults to None.

    Returns:
        float:
            The entropy as a float.
    """

    if counts is None:
        if group1 is None:
            raise InsufficientParamError()

        # Find the predicates for the two groups
        pred1, pred2 = utils.get_predicates(df, group1, group2)

        # Compute the histogram / counts for each group
        g1_counts = df[pred1][target_attr].value_counts().to_dict()
        g2_counts = df[pred2][target_attr].value_counts().to_dict()

        counts = g1_counts, g2_counts

    space = df[target_attr].unique()

    p = utils.align_probabilities(counts[0], space)
    q = utils.align_probabilities(counts[1], space)

    return entropy(p, q)


def js_divergence(
    df: pd.DataFrame,
    target_attr: str,
    group1: Optional[Dict[str, List[str]]] = None,
    group2: Optional[Dict[str, List[str]]] = None,
    counts: Optional[Tuple[Dict[Hashable, int], Dict[Hashable, int], Dict[Hashable, int]]] = None,
) -> float:
    """Computes the Jensen-Shannon Divergence between the probability distributions of the two groups with respect
    to the target attribute. If group2 is None then the distance is computed between group1 and the rest of the
    dataset. Alternatively precomputed aggregated counts for each of the groups can be provided.

    Args:
        df (pd.DataFrame):
            The input dataframe.
        target_attr (str):
            The target attribute in the dataframe.
        group1 (Optional[Dict[str, List[str]]], optional):
            The first group of interest. Defaults to None
        group2 (Optional[Dict[str, List[str]]], optional):
            The second group of interest. Defaults to None.
        counts (Optional[Tuple[Dict[Hashable, int], Dict[Hashable, int], Dict[Hashable, int]]], optional):
            A tuple containing the counts of the 2 groups and the entire dataset, each in a dictionary mapping
            from value to counts. These counts can be interpreted as the histograms between which the
            metric will be computed. Overrides group1 and group2. Defaults to None.

    Returns:
        float:
            The entropy as a float.
    """

    if counts is None:
        if group1 is None:
            raise InsufficientParamError()

        # Find the predicates for the two groups
        pred1, pred2 = utils.get_predicates(df, group1, group2)

        # Compute the histogram / counts for each group
        g1_counts = df[pred1][target_attr].value_counts().to_dict()
        g2_counts = df[pred2][target_attr].value_counts().to_dict()
        total_counts = df[target_attr].value_counts().to_dict()

        counts = g1_counts, g2_counts, total_counts

    space = df[target_attr].unique()

    p = utils.align_probabilities(counts[0], space)
    q = utils.align_probabilities(counts[1], space)
    pq = utils.align_probabilities(counts[2], space)

    return (entropy(p, pq) + entropy(q, pq)) / 2


def lp_norm(
    df: pd.DataFrame,
    target_attr: str,
    group1: Dict[str, List[str]],
    group2: Optional[Dict[str, List[str]]] = None,
    order: Union[int, str] = 2,
    counts: Optional[Tuple[Dict[Hashable, int], Dict[Hashable, int]]] = None,
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
        counts (Optional[Tuple[Dict[Hashable, int], Dict[Hashable, int]]], optional):
            A tuple containing the counts of the 2 groups and the entire dataset, each in a dictionary mapping
            from value to counts. These counts can be interpreted as the histograms between which the
            metric will be computed. Overrides group1 and group2. Defaults to None.

    Returns:
        float:
            The norm as a float.
    """

    if counts is None:
        if group1 is None:
            raise InsufficientParamError()

        # Find the predicates for the two groups
        pred1, pred2 = utils.get_predicates(df, group1, group2)

        # Compute the histogram / counts for each group
        g1_counts = df[pred1][target_attr].value_counts().to_dict()
        g2_counts = df[pred2][target_attr].value_counts().to_dict()

        counts = g1_counts, g2_counts

    space = df[target_attr].unique()

    p = utils.align_probabilities(counts[0], space)
    q = utils.align_probabilities(counts[1], space)

    return np.linalg.norm(p - q, ord=order)
