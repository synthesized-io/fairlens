"""
This module contains statistical measures for analyzing target variable distributions across sensitive groups.
"""

from typing import Any, List, Mapping, Union

import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.stats import moment

from .. import utils


def _mean_numerical(x: pd.Series) -> float:
    return moment(x, moment=1, nan_policy="omit")


def _variance_numerical(x: pd.Series) -> float:
    return moment(x, moment=2, nan_policy="omit")


def _multinomial_means(x: pd.Series) -> pd.Series:
    return x.value_counts(normalize=True, sort=False)


def _multinomial_variances(x: pd.Series) -> pd.Series:
    n = x.size
    probs = x.value_counts(normalize=True, sort=False)
    vars = [n * prob * (1 - prob) for prob in probs]
    return vars


def sensitive_group_analysis(
    df: pd.DataFrame, target_attr: str, groups: List[Union[Mapping[str, List[Any]], pd.Series]]
) -> pd.DataFrame:
    preds = utils.get_predicates_mult(df, groups)
    distrs = [df[pred][target_attr] for pred in preds]

    means = [compute_distribution_mean(distr) for distr in distrs]
    vars = [compute_distribution_variance(distr) for distr in distrs]

    results = {"Means": means, "Variances": vars}

    return pd.DataFrame(results)


def compute_distribution_mean(x: pd.Series, categorical_mode: str = "multinomial") -> Union[float, pd.Series]:
    if is_numeric_dtype(x):
        return _mean_numerical(x)
    if categorical_mode == "multinomial":
        return _multinomial_means(x)
    else:
        return None


def compute_distribution_variance(x: pd.Series, categorical_mode: str = "multinomial") -> Union[float, pd.Series]:
    if is_numeric_dtype(x):
        return _variance_numerical(x)
    if categorical_mode == "multinomial":
        return _multinomial_variances(x)
    else:
        return None
