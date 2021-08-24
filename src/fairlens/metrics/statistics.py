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


def sensitive_group_analysis(
    df: pd.DataFrame, target_attr: str, groups: List[Union[Mapping[str, List[Any]], pd.Series]]
) -> pd.DataFrame:
    preds = utils.get_predicates_mult(df, groups)
    distrs = [df[pred][target_attr] for pred in preds]

    means = [compute_distribution_mean(distr) for distr in distrs]
    vars = [compute_distribution_variance(distr) for distr in distrs]

    results = {"Means": means, "Variances": vars}

    return pd.DataFrame(results)


def compute_distribution_mean(x: pd.Series) -> float:
    pass


def compute_distribution_variance(x: pd.Series) -> float:
    pass
