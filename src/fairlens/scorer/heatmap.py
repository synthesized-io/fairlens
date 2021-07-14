from typing import Callable

import dcor as dcor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns

from fairlens.bias import utils


def two_column_heatmap(
    df: pd.DataFrame,
    num_num_metric: Callable[[pd.Series, pd.Series], float] = None,
    cat_num_metric: Callable[[pd.Series, pd.Series], float] = None,
    cat_cat_metric: Callable[[pd.Series, pd.Series], float] = None,
):
    """This function creates a correlation heatmap out of a dataframe, using user provided or default correlation
    metrics for all possible types of pairs of series (i.e. numerical-numerical, categorical-numerical,
    categorical-categorical).

    Args:
        df (pd.DataFrame):
            The dataframe used for computing correlations and producing a heatmap.
        num_num_metric (Callable[[pd.Series, pd.Series], float], optional):
            The correlation metric used for numerical-numerical series pairs. Defaults to Pearson's correlation
            coefficient.
        cat_num_metric (Callable[[pd.Series, pd.Series], float], optional):
            The correlation metric used for categorical-numerical series pairs. Defaults to Kruskal-Wallis' H Test.
        cat_cat_metric (Callable[[pd.Series, pd.Series], float], optional):
            The correlation metric used for categorical-categorical series pairs. Defaults to corrected Cramer's V
            statistic.
    """
    num_num_metric = num_num_metric or _pearson
    cat_num_metric = cat_num_metric or _kruskal_wallis
    cat_cat_metric = cat_cat_metric or _cramers_v

    corr_matrix = compute_correlation_matrix(df, num_num_metric, cat_num_metric, cat_cat_metric).round(2)

    plt.figure(figsize=(16, 6))
    heatmap = sns.heatmap(corr_matrix, vmin=0, vmax=1, annot=True)
    heatmap.set_title("Correlation Heatmap", fontdict={"fontsize": 12}, pad=12)


def compute_correlation_matrix(
    df: pd.DataFrame,
    num_num_metric: Callable[[pd.Series, pd.Series], float] = None,
    cat_num_metric: Callable[[pd.Series, pd.Series], float] = None,
    cat_cat_metric: Callable[[pd.Series, pd.Series], float] = None,
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

    Returns:
        pd.DataFrame:
            The correlation matrix to be used in heatmap generation.
    """
    nn_metric = num_num_metric or _pearson
    cn_metric = cat_num_metric or _kruskal_wallis
    cc_metric = cat_cat_metric or _cramers_v

    def corr_wrapper(a: np.ndarray, b: np.ndarray):
        sr_a = pd.Series(a)
        sr_b = pd.Series(b)
        a_type = utils.infer_distr_type(sr_a)
        b_type = utils.infer_distr_type(sr_b)

        if a_type.is_continuous() and b_type.is_continuous():
            return nn_metric(sr_a, sr_b)

        elif a_type.is_continuous():
            return cn_metric(sr_b, sr_a)

        elif b_type.is_continuous():
            return cn_metric(sr_a, sr_b)

        return cc_metric(sr_a, sr_b)

    return df.corr(method=corr_wrapper)


def _cramers_v(sr_a: pd.Series, sr_b: pd.Series) -> float:
    """Metric that calculates the corrected Cramer's V statistic for categorical-categorical
    correlations, used in heatmap generation.

    Args:
        sr_a (pd.Series): First categorical series to analyze.
        sr_b (pd.Series): Second categorical series to analyze.

    Returns:
        float: Value of the statistic.
    """

    if len(sr_a.value_counts()) == 1:
        return 0
    if len(sr_b.value_counts()) == 1:
        return 0
    else:
        confusion_matrix = pd.crosstab(sr_a, sr_b)

        if confusion_matrix.shape[0] == 2:
            correct = False
        else:
            correct = True

        chi2 = ss.chi2_contingency(confusion_matrix, correction=correct)[0]
        n = sum(confusion_matrix.sum())
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def _pearson(sr_a: pd.Series, sr_b: pd.Series) -> float:
    """Metric that calculates Pearson's correlation coefficent for numerical-numerical
    pairs of series, used in heatmap generation.

    Args:
        sr_a (pd.Series): First numerical series to analyze.
        sr_b (pd.Series): Second numerical series to analyze.

    Returns:
        float: Value of the coefficient.
    """
    return abs(sr_a.corr(sr_b))


def _kruskal_wallis(sr_a: pd.Series, sr_b: pd.Series) -> float:
    """Metric that uses the Kruskal-Wallis H Test to obtain a p-value indicating the possibility
    that a categorical and numerical series are not correlated, used in heatmap generation.

    Args:
        sr_a (pd.Series): The categorical series to analyze, used for grouping the numerical one.
        sr_b (pd.Series): The numerical series to analyze.

    Returns:
        float: The correlation coefficient, calculating by subtracting the p-value from 1, as the
        p-value is the probability that the two columns are not correlated.
    """

    sr_a = sr_a.astype("category").cat.codes
    groups = sr_b.groupby(sr_a)
    arrays = [groups.get_group(category) for category in sr_a.unique()]

    args = [group.array for group in arrays]
    try:
        _, p_val = ss.kruskal(*args, nan_policy="omit")
    except ValueError:
        return 0

    return 1 - p_val


def _distance_nn_correlation(sr_a: pd.Series, sr_b: pd.Series) -> float:
    """Metric that uses non-linear correlation distance to obtain a correlation coefficient for
    numerical-numerical column pairs.

    Args:
        sr_a (pd.Series): First numerical series to analyze.
        sr_b (pd.Series): Second numerical series to analyze.

    Returns:
        float: The correlation coefficient.
    """
    if sr_a.size < sr_b.size:
        sr_a = sr_a.append(pd.Series(sr_a.mean()).repeat(sr_b.size - sr_a.size), ignore_index=True)
    elif sr_a.size > sr_b.size:
        sr_b = sr_b.append(pd.Series(sr_b.mean()).repeat(sr_a.size - sr_b.size), ignore_index=True)

    return dcor.distance_correlation(sr_a, sr_b)


def _distance_cn_correlation(sr_a: pd.Series, sr_b: pd.Series) -> float:
    """Metric that uses non-linear correlation distance to obtain a correlation coefficient for
    categorical-numerical column pairs.

    Args:
        sr_a (pd.Series): The categorical series to analyze, used for grouping the numerical one.
        sr_b (pd.Series): The numerical series to analyze.

    Returns:
        float: The correlation coefficient.
    """
    sr_a = sr_a.astype("category").cat.codes
    groups = sr_b.groupby(sr_a)
    arrays = [groups.get_group(category) for category in sr_a.unique()]

    total = 0.0
    n = len(arrays)

    for i in range(0, n):
        for j in range(i + 1, n):
            sr_i = arrays[i]
            sr_j = arrays[j]

            # Handle groups with a different number of elements.
            if sr_i.size < sr_j.size:
                sr_i = sr_i.append(sr_i.sample(sr_j.size - sr_i.size), ignore_index=True)
            elif sr_i.size > sr_j.size:
                sr_j = sr_j.append(sr_j.sample(sr_i.size - sr_j.size), ignore_index=True)
            total += dcor.distance_correlation(sr_i, sr_j)

    total /= n * (n - 1) / 2

    if total is None:
        return 0.0

    return total
