"""
This module contains statistical measures for analyzing target variable distributions across sensitive groups.
"""

from typing import Any, List, Mapping, Union

import pandas as pd
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


def compute_distribution_mean(x: pd.Series) -> Union[float, pd.Series]:
    pass


def compute_distribution_variance(x: pd.Series) -> Union[float, pd.Series]:
    pass
