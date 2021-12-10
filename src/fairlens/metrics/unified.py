"""
Collection of helper methods which can be used as to interface metrics.
"""

import multiprocessing as mp
from collections import namedtuple
from typing import Any, Callable, List, Mapping, Optional, Tuple, Union

import pandas as pd
from synthesized_insight.check import Check, ColumnCheck
from synthesized_insight.metrics.base import TwoColumnMetric
from synthesized_insight.metrics.statistical_tests import BootstrapTest, KolmogorovSmirnovDistanceTest, PermutationTest

from .. import utils
from .correlation import cramers_v, kruskal_wallis, pearson
from .distance import BinomialDistance, DistanceMetric, EarthMoversDistance, KolmogorovSmirnovDistance

DistanceResult = namedtuple("DistanceResult", ("distance", "p_value"))


class _OptimisticCheck(ColumnCheck):
    def continuous(self, sr: pd.Series) -> bool:
        return True

    def categorical(self, sr: pd.Series) -> bool:
        return True


check = _OptimisticCheck()


def stat_distance(
    df: pd.DataFrame,
    target_attr: str,
    group1: Union[Mapping[str, List[Any]], pd.Series],
    group2: Union[Mapping[str, List[Any]], pd.Series],
    mode: str = "auto",
    test: str = "default",
    alternative: str = "two-sided",
    **kwargs,
) -> Tuple[float, Optional[float]]:
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
        group1 (Union[Mapping[str, List[Any]], pd.Series]):
            The first group of interest. Each group can be a mapping / dict from attribute to value or
            a predicate itself, i.e. pandas series consisting of bools which can be used as a predicate
            to index a subgroup from the dataframe.
            Examples: {"Sex": ["Male"]}, df["Sex"] == "Female"
        group2 (Union[Mapping[str, List[Any]], pd.Series]):
            The second group of interest. Each group can be a mapping / dict from attribute to value or
            a predicate itself, i.e. pandas series consisting of bools which can be used as a predicate
            to index a subgroup from the dataframe.
            Examples: {"Sex": ["Male"]}, df["Sex"] == "Female"
        mode (str, optional):
            Which distance metric to use. Can be the names of classes from `fairlens.metrics`, or their
            id() strings. If set to "auto", the method automatically picks a suitable metric based on the
            distribution of the target attribute. Defaults to "auto".
        test (str, optional):
            The statistical test to use to compute the p-value. Can take values
            ["default", "bootstrap", "permutation"]. If set to "default", returns any p-value
            for the metric specified by `mode`. Defaults to "default"
        alternative (str, optional):
            The alternative hypothesis for the test. Can take values ["two-sided", "greater", "less"].
            Defaults to "two-sided".
        **kwargs:
            Keyword arguments for the distance metric. Passed to the __init__ function of distance metrics.

    Returns:
        Tuple[float, Optional[float]]:
            The distance as a float, the p-value as a float.

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

    if test not in ["default", "bootstrap", "permutation"]:
        raise ValueError('Invalid value for test. Valid values: ["default", "bootstrap", "permutation"]')

    if alternative not in ["two-sided", "greater", "less"]:
        raise ValueError('Invalid value for alternative. Valid values: ["two-sided", "greater", "less"]')

    # Parse group arguments into pandas series'
    pred1, pred2 = tuple(utils.get_predicates_mult(df, [group1, group2]))
    group1 = df[pred1][target_attr]
    group2 = df[pred2][target_attr]

    # Choose the metric
    if mode == "auto":
        mode = auto_distance(df[target_attr])

    if mode not in DistanceMetric._class_dict:
        raise ValueError(f"Invalid value for mode. Valid values:\n{DistanceMetric._class_dict.keys()}")

    dist_class = DistanceMetric._class_dict[mode]
    distance_metric = dist_class(**kwargs)
    metric = _MetricAdaptor(distance_metric)

    d, p = None, None

    if test == "bootstrap":
        d, p = BootstrapTest(metric, alternative=alternative, check=check)(group1, group2)
    elif test == "permutation":
        d, p = PermutationTest(metric, alternative=alternative, check=check)(group1, group2)
    elif test == "default" and mode == "ks_distance":
        d, p = KolmogorovSmirnovDistanceTest(alternative=alternative, check=check)(group1, group2)
    else:
        d = distance_metric(group1, group2)
        p = None

    if d is None:
        raise ValueError("Incompatible data inside both series")

    return DistanceResult(distance=d, p_value=p)


def auto_distance(column: pd.Series) -> str:
    """Return a suitable statistical distance metric based on the distribution of the data.

    Args:
        column (pd.Series):
            The input data in a pd.Series.

    Returns:
        str:
            The id of the distance metric.
    """

    distr_type = utils.infer_distr_type(column)
    if distr_type.is_continuous():
        return KolmogorovSmirnovDistance._get_id()
    elif distr_type.is_binary():
        return BinomialDistance._get_id()

    return EarthMoversDistance._get_id()


def correlation_matrix(
    df: pd.DataFrame,
    num_num_metric: Callable[[pd.Series, pd.Series], float] = pearson,
    cat_num_metric: Callable[[pd.Series, pd.Series], float] = kruskal_wallis,
    cat_cat_metric: Callable[[pd.Series, pd.Series], float] = cramers_v,
    columns_x: Optional[List[str]] = None,
    columns_y: Optional[List[str]] = None,
) -> pd.DataFrame:
    """This function creates a correlation matrix out of a dataframe, using a correlation metric for each
    possible type of pair of series (i.e. numerical-numerical, categorical-numerical, categorical-categorical).

    Args:
        df (pd.DataFrame):
            The dataframe that will be analyzed to produce correlation coefficients.
        num_num_metric (Callable[[pd.Series, pd.Series], float], optional):
            The correlation metric used for numerical-numerical series pairs. Defaults to Pearson's correlation
            coefficient.
        cat_num_metric (Callable[[pd.Series, pd.Series], float], optional):
            The correlation metric used for categorical-numerical series pairs. Defaults to Kruskal-Wallis' H Test.
        cat_cat_metric (Callable[[pd.Series, pd.Series], float], optional):
            The correlation metric used for categorical-categorical series pairs. Defaults to corrected Cramer's V
            statistic.
        columns_x (Optional[List[str]]):
            The column names that determine the rows of the matrix.
        columns_y (Optional[List[str]]):
            The column names that determine the columns of the matrix.

    Returns:
        pd.DataFrame:
            The correlation matrix to be used in heatmap generation.
    """

    if columns_x is None:
        columns_x = df.columns

    if columns_y is None:
        columns_y = df.columns

    pool = mp.Pool(mp.cpu_count())

    series_list = [
        pd.Series(
            pool.starmap(
                _correlation_matrix_helper,
                [(df[col_x], df[col_y], num_num_metric, cat_num_metric, cat_cat_metric) for col_x in columns_x],
            ),
            index=columns_x,
            name=col_y,
        )
        for col_y in columns_y
    ]

    pool.close()

    return pd.concat(series_list, axis=1, keys=[series.name for series in series_list])


def _correlation_matrix_helper(
    sr_a: pd.Series,
    sr_b: pd.Series,
    num_num_metric: Callable[[pd.Series, pd.Series], float] = pearson,
    cat_num_metric: Callable[[pd.Series, pd.Series], float] = kruskal_wallis,
    cat_cat_metric: Callable[[pd.Series, pd.Series], float] = cramers_v,
) -> float:

    a_type = utils.infer_distr_type(sr_a)
    b_type = utils.infer_distr_type(sr_b)

    if a_type.is_continuous() and b_type.is_continuous():
        return num_num_metric(sr_a, sr_b)

    elif b_type.is_continuous():
        return cat_num_metric(sr_a, sr_b)

    elif a_type.is_continuous():
        return cat_num_metric(sr_b, sr_a)

    else:
        return cat_cat_metric(sr_a, sr_b)


class _MetricAdaptor(TwoColumnMetric):
    def __init__(self, metric: DistanceMetric):
        self.metric = metric

    def __call__(self, sr_a: pd.Series, sr_b: pd.Series):
        return self.metric(sr_a, sr_b)

    def check_column_types(cls, sr_a: pd.Series, sr_b: pd.Series, check: Check):
        return True

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
        return self.metric.distance(sr_a, sr_b)
