from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pyemd import emd as pemd
from scipy.stats import entropy, ks_2samp

from . import utils


def class_imbalance(
    df: pd.DataFrame,
    target_attr: str,
    group1: Union[Dict[str, List[str]], pd.Series],
    group2: Union[Dict[str, List[str]], pd.Series],
) -> float:
    """Computes the class imbalance between group1 and group2 with respect to the target attribute.

    Args:
        df (pd.DataFrame):
            The input dataframe.
        target_attr (str):
            The target attribute in the dataframe.
        group1 (Union[Dict[str, List[str]], pd.Series]):
            The first group of interest. Can be a dictionary mapping from attribute to values
            or the raw data in a pandas series.
        group2 (Union[Dict[str, List[str]], pd.Series]):
            The second group of interest. Can be a dictionary mapping from attribute to values
            or the raw data in a pandas series.

    Returns:
        float:
            The class imbalance as a float.

    Examples:
        >>> df = pd.read_csv("datasets/compas.csv")
        >>> class_imbalance(df, 'RawScore', {'Ethnicity': ['African-American']}, {'Ethnicity': ['Caucasian']})
        0.021244309559939303
    """

    g1, g2 = utils.parse_args(df, group1, group2)

    return (g1.nunique() - g2.nunique()) / df[target_attr].nunique()


def emd(
    df: pd.DataFrame,
    target_attr: str,
    group1: Union[Dict[str, List[str]], pd.Series],
    group2: Union[Optional[Dict[str, List[str]]], pd.Series] = None,
) -> float:
    """Computes the Earth Mover's Distance between the probability distributions of group1 and group2 with
    respect to the target attribute. If group1 is a dictionary and group2 is None then the distance is
    computed between group1 and the rest of the dataset.

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

    Returns:
        float:
            The Earth Mover's Distance as a float.

    Examples:
        >>> df = pd.read_csv("datasets/compas.csv")
        >>> emd(df, 'RawScore', {'Ethnicity': ['African-American']}, {'Ethnicity': ['Caucasian']})
        0.15406599999999984
        >>> emd(df, 'RawScore', {'Ethnicity': ['African-American']})
        0.16237499999999971
    """

    g1, g2 = utils.parse_args(df, group1, group2)

    space = df[target_attr].unique()

    p, q = tuple(utils.compute_probabilities(space, g1, g2))

    xx, yy = np.meshgrid(space, space)
    distance_space = np.abs(xx - yy)

    return pemd(p, q, distance_space)


def ks_distance(
    df: pd.DataFrame,
    target_attr: str,
    group1: Union[Dict[str, List[str]], pd.Series],
    group2: Union[Optional[Dict[str, List[str]]], pd.Series] = None,
) -> Tuple[float, float]:
    """Performs the Kolmogorov–Smirnov test between the probability distributions of group1 and group2 with
    respect to the target attribute. If group1 is a dictionary and group2 is None then the distance is
    computed between group1 and the rest of the dataset.

    Args:
        df (pd.DataFrame):
            The input dataframe.
        target_attr (str):
            The target attribute in the dataframe.
        group1 (Union[Dict[str, List[str]], pd.Series]):
            The first group of interest. Can be a dictionary mapping from attribute to values
            or the raw data in a pandas series.
        group2 (Union[Optional[Dict[str, List[str]]], pd.Series]], optional):
            The second group of interest. Can be a dictionary mapping from attribute to values
            or the raw data in a pandas series. Defaults to None.

    Returns:
        Tuple[float, float]:
            A tuple containing the Kolmogorov–Smirnov statistic and the associated p-value

    Examples:
        >>> df = pd.read_csv("datasets/compas.csv")
        >>> ks_distance(df, 'RawScore', {'Sex': ['Male']}, {'Sex': ['Female']})
        (0.0574598854742705, 2.585181602569765e-30)
        >>> ks_distance(df, 'RawScore', {'FirstName': ['Stephanie']}, {'FirstName': ['Kevin']})
        (0.0744047619047619, 0.8522462138629425)
    """

    g1, g2 = utils.parse_args(df, group1, group2)

    statistic, pval = ks_2samp(g1, g2)

    return statistic, pval


def kl_divergence(
    df: pd.DataFrame,
    target_attr: str,
    group1: Union[Dict[str, List[str]], pd.Series],
    group2: Union[Optional[Dict[str, List[str]]], pd.Series] = None,
) -> float:
    """Computes the Kullback–Leibler Divergence or Relative Entropy between the probability distributions of the
    two groups with respect to the target attribute. If group1 is a dictionary and group2 is None then the distance is
    computed between group1 and the rest of the dataset.

    Args:
        df (pd.DataFrame):
            The input dataframe.
        target_attr (str):
            The target attribute in the dataframe.
        group1 (Union[Dict[str, List[str]], pd.Series]):
            The first group of interest. Can be a dictionary mapping from attribute to values
            or the raw data in a pandas series.
        group2 (Union[Optional[Dict[str, List[str]]], pd.Series]], optional):
            The second group of interest. Can be a dictionary mapping from attribute to values
            or the raw data in a pandas series. Defaults to None.

    Returns:
        float:
            The entropy as a float.
    """

    g1, g2 = utils.parse_args(df, group1, group2)

    space = df[target_attr].unique()

    p, q = tuple(utils.compute_probabilities(space, g1, g2))

    return entropy(p, q)


def js_divergence(
    df: pd.DataFrame,
    target_attr: str,
    group1: Union[Dict[str, List[str]], pd.Series],
    group2: Union[Optional[Dict[str, List[str]]], pd.Series] = None,
) -> float:
    """Computes the Jensen-Shannon Divergence between the probability distributions of the two groups with
    respect to the target attribute. If group1 is a dictionary and group2 is None then the distance is
    computed between group1 and the rest of the dataset.

    Args:
        df (pd.DataFrame):
            The input dataframe.
        target_attr (str):
            The target attribute in the dataframe.
        group1 (Union[Dict[str, List[str]], pd.Series]):
            The first group of interest. Can be a dictionary mapping from attribute to values
            or the raw data in a pandas series.
        group2 (Union[Optional[Dict[str, List[str]]], pd.Series]], optional):
            The second group of interest. Can be a dictionary mapping from attribute to values
            or the raw data in a pandas series. Defaults to None.

    Returns:
        float:
            The entropy as a float.
    """

    g1, g2 = utils.parse_args(df, group1, group2)

    space = df[target_attr].unique()

    p, q, pq = tuple(utils.compute_probabilities(space, g1, g2, df[target_attr]))

    return (entropy(p, pq) + entropy(q, pq)) / 2


def lp_norm(
    df: pd.DataFrame,
    target_attr: str,
    group1: Union[Dict[str, List[str]], pd.Series],
    group2: Union[Optional[Dict[str, List[str]]], pd.Series] = None,
    order: Union[int, str] = 2,
) -> float:
    """Computes the LP Norm between the probability distributions of the two groups with respect to the
    target attribute. If group1 is a dictionary and group2 is None then the distance is computed between
    group1 and the rest of the dataset.

    Args:
        df (pd.DataFrame):
            The input dataframe.
        target_attr (str):
            The target attribute in the dataframe.
        group1 (Union[Dict[str, List[str]], pd.Series]):
            The first group of interest. Can be a dictionary mapping from attribute to values
            or the raw data in a pandas series.
        group2 (Union[Optional[Dict[str, List[str]]], pd.Series]], optional):
            The second group of interest. Can be a dictionary mapping from attribute to values
            or the raw data in a pandas series. Defaults to None.
        order (Union[int, str], optional):
            The order of the norm (p). Passed as 'ord' to numpy.linalg.norm. Defaults to 2.

    Returns:
        float:
            The norm as a float.
    """

    g1, g2 = utils.parse_args(df, group1, group2)

    space = df[target_attr].unique()

    p, q = tuple(utils.compute_probabilities(space, g1, g2))

    return np.linalg.norm(p - q, ord=order)
