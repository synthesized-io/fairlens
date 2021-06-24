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
    num_num_metric = num_num_metric or _pearson
    cat_num_metric = cat_num_metric or _kruskal_wallis
    cat_cat_metric = cat_cat_metric or _cramers_v

    corr_matrix = compute_correlation_matrix(df, num_num_metric, cat_num_metric, cat_cat_metric)

    plt.figure(figsize=(16, 6))
    heatmap = sns.heatmap(corr_matrix, vmin=0, vmax=1, annot=True)
    heatmap.set_title("Correlation Heatmap", fontdict={"fontsize": 12}, pad=12)


def compute_correlation_matrix(
    df: pd.DataFrame,
    num_num_metric: Callable[[pd.Series, pd.Series], float] = None,
    cat_num_metric: Callable[[pd.Series, pd.Series], float] = None,
    cat_cat_metric: Callable[[pd.Series, pd.Series], float] = None,
) -> pd.DataFrame:
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
    confusion_matrix = pd.crosstab(sr_a, sr_b)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = len(sr_a)
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.nan_to_num(np.sqrt(phi2corr / np.minimum(kcorr - 1, rcorr - 1)))


def _pearson(sr_a: pd.Series, sr_b: pd.Series) -> float:
    return abs(sr_a.corr(sr_b))


def _kruskal_wallis(sr_a: pd.Series, sr_b: pd.Series) -> float:
    # TODO: replace with distance metric.

    sr_a = sr_a.astype("category").cat.codes
    groups = sr_b.groupby(sr_a)
    arrays = [groups.get_group(category) for category in sr_a.unique()]

    args = [group.array for group in arrays]
    try:
        _, p_val = ss.kruskal(*args, nan_policy="omit")
    except ValueError:
        return 1

    return 1 - p_val


def _distance_nn_correlation(sr_a: pd.Series, sr_b: pd.Series) -> float:
    if sr_a.size < sr_b.size:
        sr_a = sr_a.append(pd.Series(sr_a.mean()).repeat(sr_b.size - sr_a.size), ignore_index=True)
    elif sr_a.size > sr_b.size:
        sr_b = sr_b.append(pd.Series(sr_b.mean()).repeat(sr_a.size - sr_b.size), ignore_index=True)

    return dcor.distance_correlation(sr_a, sr_b)


def _distance_cn_correlation(sr_a: pd.Series, sr_b: pd.Series) -> float:
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
        return 1.0

    return total
