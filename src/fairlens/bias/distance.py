"""
Collection of Metrics that measure the distance, or similarity, between two datasets.
"""
from typing import Dict, List, Tuple, Type, Union

import pandas as pd

import fairlens.metrics.distance_metrics as dist

from . import utils


def auto_distance(column: pd.Series) -> Type[dist.DistanceMetric]:
    """Return the best statistical distance metric based on the distribution of the data.

    Args:
        column (pd.Series):
            The input data in a pd.Series.

    Returns:
        Type[DistanceMetric]:
            The class of the distance metric.
    """

    distr_type = utils.infer_distr_type(column)
    if distr_type.is_continuous():
        return dist.KolmogorovSmirnovDistance
    elif distr_type.is_binary():
        return dist.BinomialDistance

    return dist.EarthMoversDistanceCategorical


def stat_distance(
    df: pd.DataFrame,
    target_attr: str,
    group1: Union[Dict[str, List[str]], pd.Series],
    group2: Union[Dict[str, List[str]], pd.Series],
    mode: str = "auto",
    p_value: bool = False,
    **kwargs,
) -> Tuple[float, ...]:
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
        Tuple[float, ...]:
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
    if isinstance(group1, dict) and isinstance(group2, dict):
        if target_attr not in df.columns:
            raise ValueError(f'"{target_attr}" is not a valid column name.')

        pred1, pred2 = tuple(utils.get_predicates_mult(df, [group1, group2]))

        group1 = df[pred1][target_attr]
        group2 = df[pred2][target_attr]

    if not isinstance(group1, pd.Series) or not isinstance(group2, pd.Series):
        raise TypeError("group1, group2 must be pd.Series or dictionaries")

    if target_attr in df.columns:
        column = df[target_attr]
    else:
        column = pd.concat((group1, group2))

    # Choose the distance metric
    if mode == "auto":
        dist_class = auto_distance(column)
    elif mode in dist.DistanceMetric.class_dict:
        dist_class = dist.DistanceMetric.class_dict[mode]
    else:
        raise ValueError(f"Invalid mode. Valid modes include:\n{dist.DistanceMetric.class_dict.keys()}")

    metric = dist_class(**kwargs)
    d = metric(group1, group2)

    if d is None:
        raise ValueError("Incompatible data inside both series")

    if p_value:
        p = metric.p_value(group1, group2)
        return (d, p)

    return (d,)
