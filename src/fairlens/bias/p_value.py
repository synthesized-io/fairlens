from typing import Callable, Tuple

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


def bootstrap_pvalue(t_obs: float, t_distribution: pd.Series, alternative: str = "two-sided") -> float:
    """Calculate a p-value using a bootstrapped test statistic distribution.

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


def bootstrap_statistic(
    data: Tuple[pd.Series, ...],
    statistic: Callable[[pd.Series, pd.Series], float],
    n_samples: int = 1000,
    sample_size=None,
) -> np.ndarray:
    """Compute the samples of a statistic estimate using the bootstrap method.

    Args:
        data (Tuple[pd.Series, ...]):
            Data on which to compute the statistic in a tuple.
        statistic (Callable[[pd.Series, pd.Series], float]):
            Function that computes the statistic.
        n_samples (int, optional):
            Number of bootstrap samples to perform.
    Returns:
        np.ndarray:
            The bootstrap samples.
    """

    if sample_size is None:
        sample_size = max((len(x) for x in data))

    def get_sample_idx(x):
        return np.random.randint(0, len(x), min(len(x), sample_size))

    statistic_samples = np.empty(n_samples)
    for i in range(n_samples):
        sample_idxs = [get_sample_idx(x) for x in data]
        statistic_samples[i] = statistic(*[x[idx] for x, idx in zip(data, sample_idxs)])

    return statistic_samples


def bootstrap_binned_statistic(
    data: Tuple[pd.Series, ...], statistic: Callable[[pd.Series, pd.Series], float], n_samples: int = 1000
) -> np.ndarray:
    """Compute the samples of a binned statistic estimate using the bootstrap method.
    
    Args:
        data (Tuple[pd.Series, ...]):
            Data on which to compute the statistic in a tuple.
        statistic (Callable[[pd.Series, pd.Series], float]):
            Function that computes the statistic.
        n_samples (int, optional):
            Number of bootstrap samples to perform.
    Returns:
        np.ndarray:
            The bootstrap samples.
    """

    statistic_samples = np.empty(n_samples)

    with np.errstate(divide="ignore", invalid="ignore"):
        p_x = np.nan_to_num(data[0] / data[0].sum())
        p_y = np.nan_to_num(data[1] / data[1].sum())

    n_x = data[0].sum()
    n_y = data[1].sum()

    x_samples = np.random.multinomial(n_x, p_x, size=n_samples)
    y_samples = np.random.multinomial(n_y, p_y, size=n_samples)

    for i in range(n_samples):
        statistic_samples[i] = statistic(x_samples[i], y_samples[i])

    return statistic_samples


def permutation_test(
    x: pd.Series,
    y: pd.Series,
    t: Callable[[pd.Series, pd.Series], float],
    n_perm: int = 100,
    alternative: str = "two-sided",
) -> float:
    """
    Perform a two sample permutation test.
    Determines the probability of observing t(x, y) or greater under the null hypothesis that x
    and y are from the same distribution.

    Args:
        x (pd.Series):
            First data sample.
        y (pd.Series):
            Second data sample.
        t (Callable[[pd.Series, pd.Series], float]):
            Callable that returns the test statistic.
        n_perm (int):
            Number of permutations.
        alternative: Optional;
            Indicates the alternative hypothesis. One of "two-sided', "greater", "less".
            Defaults to "two-sided".
    Returns:
        float:
            The p-value of t_obs under the null hypothesis.
    """

    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("'alternative' argument must be one of 'two-sided', 'greater', 'less'")

    t_obs = t(x, y)
    pooled_data = np.concatenate((x, y))
    t_null = np.empty(n_perm)

    for i in range(n_perm):
        perm = np.random.permutation(pooled_data)
        x_sample = perm[:len(x)]
        y_sample = perm[len(x):]
        t_null[i] = t(x_sample, y_sample)

    if alternative == "two-sided":
        p = np.sum(np.abs(t_null) >= np.abs(t_obs)) / n_perm

    elif alternative == "greater":
        p = np.sum(t_null >= t_obs) / n_perm

    else:
        p = np.sum(t_null < t_obs) / n_perm

    return p
