"""Calculating p_values for using sampling techniques.

This module provides three functions to sample and generate distributions required for estimating p_values:
  - `binominal_proportion_p_value`
  - `permutation_statistic`
  - `bootstrap_statistic`

The final function, `resampling_p_value` is used for then calculating the p_values.
"""

from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.stats.morestats import binom_test


def binominal_proportion_p_value(p_obs: float, p_null: float, n: int, alternative: str = "two-sided") -> float:
    """Calculate an exact p-value for an observed binomial proportion of a sample.

    Args:
        p_obs (float):
            Observed proportion of successes.
        p_null (float):
            Expected proportion of sucesses under null hypothesis.
        n (int):
            Sample size.
        alternative (str, optional):
            Indicates the alternative hypothesis. One of "two-sided", "greater", "less".
            Defaults to "two-sided".

    Returns:
        float:
            The p-value under the null hypothesis.
    """

    k = np.ceil(p_obs * n)
    return binom_test(k, n, p_null, alternative)


def permutation_statistic(
    x: pd.Series,
    y: pd.Series,
    statistic: Callable[[pd.Series, pd.Series], float],
    n_perm: int = 100,
) -> np.ndarray:
    """
    Performs the sampling for a two sample permutation test.

    Args:
        x (pd.Series):
            First data sample.
        y (pd.Series):
            Second data sample.
        statistic (Callable[[pd.Series, pd.Series], float]):
            Function that computes the test statistic.
        n_perm (int):
            Number of permutations.

    Returns:
        np.ndarray:
            The distribution of the statistic on a n_perm permutations of samples.
    """

    joint = np.concatenate((x, y))
    t_null = np.empty(n_perm)

    for i in range(n_perm):
        perm = np.random.permutation(joint)
        x_sample = perm[: len(x)]
        y_sample = perm[len(x) :]
        t_null[i] = statistic(x_sample, y_sample)

    return t_null


def bootstrap_statistic(
    x: pd.Series,
    y: pd.Series,
    statistic: Callable[[pd.Series, pd.Series], float],
    n_samples: int = 100,
    sample_size: Optional[int] = None,
) -> np.ndarray:
    """Compute the samples of a statistic estimate using the bootstrap method.

    Args:
        x (pd.Series):
            First data sample.
        y (pd.Series):
            Second data sample.
        statistic (Callable[[pd.Series, pd.Series], float]):
            Function that computes the test statistic.
        n_samples (int, optional):
            Number of bootstrap samples to perform.
        sample_size (Optional[int], optional):
            Number of data samples in a bootstrap sample.

    Returns:
        np.ndarray:
            The bootstrap samples.
    """

    if sample_size is None:
        sample_size = min(len(x), len(y))

    statistic_samples = np.empty(n_samples)
    for i in range(n_samples):
        x_sample = x.sample(sample_size, replace=True)
        y_sample = y.sample(sample_size, replace=True)
        statistic_samples[i] = statistic(x_sample, y_sample)

    return statistic_samples


def bootstrap_binned_statistic(
    h_x: pd.Series, h_y: pd.Series, statistic: Callable[[pd.Series, pd.Series], float], n_samples: int = 1000
) -> np.ndarray:
    """Compute the samples of a binned statistic estimate using the bootstrap method.

    Args:
        h_x (pd.Series):
            First histogram.
        h_y (pd.Series):
            Second histogram.
        statistic (Callable[[pd.Series, pd.Series], float]):
            Function that computes the statistic.
        n_samples (int, optional):
            Number of bootstrap samples to perform.

    Returns:
        np.ndarray:
            The bootstrap samples.
    """

    statistic_samples = np.empty(n_samples)

    n_x = h_x.sum()
    n_y = h_y.sum()

    with np.errstate(divide="ignore", invalid="ignore"):
        p_x = np.nan_to_num(h_x / n_x)
        p_y = np.nan_to_num(h_y / n_y)

    x_samples = np.random.multinomial(n_x, p_x, size=n_samples)
    y_samples = np.random.multinomial(n_y, p_y, size=n_samples)

    for i in range(n_samples):
        statistic_samples[i] = statistic(x_samples[i], y_samples[i])

    return statistic_samples


def resampling_pvalue(t_obs: float, t_distribution: pd.Series, alternative: str = "two-sided") -> float:
    """Calculate a p-value using a resampled test statistic distribution.

    Args:
        t_obs (float):
            Observed value of the test statistic.
        t_distribution (pd.Series):
            Samples of test statistic distribution under the null hypothesis.
        alternative (str, optional):
            Indicates the alternative hypothesis. One of "two-sided", "greater", "less".
            Defaults to "two-sided".

    Returns:
        float:
            The p-value under the null hypothesis.
    """

    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("'alternative' argument must be one of 'two-sided', 'greater', 'less'")

    n_samples = len(t_distribution)

    if alternative == "two-sided":
        p = np.sum(np.abs(t_distribution) >= np.abs(t_obs)) / n_samples

    elif alternative == "greater":
        p = np.sum(t_distribution >= t_obs) / n_samples

    else:
        p = np.sum(t_distribution < t_obs) / n_samples

    return p
