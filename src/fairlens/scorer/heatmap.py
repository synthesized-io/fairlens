from typing import Callable

import matplotlib as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns


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
    num_num_metric = num_num_metric or _pearson
    cat_num_metric = cat_num_metric or _kruskal_wallis
    cat_cat_metric = cat_cat_metric or _cramers_v

    series_list = list()
    for sr_a in df.columns:
        coeffs = list()
        a_categorical = df[sr_a].map(type).eq(str).all()
        for sr_b in df.columns:
            if sr_a == sr_b:
                coeffs.append(1.0)
                continue
            b_categorical = df[sr_b].map(type).eq(str).all()
            if a_categorical and b_categorical:
                coeffs.append(cat_cat_metric(df[sr_a], df[sr_b]))
            elif a_categorical and not b_categorical:
                coeffs.append(cat_num_metric(df[sr_a], df[sr_b]))
            elif not a_categorical and b_categorical:
                coeffs.append(cat_num_metric(df[sr_a], df[sr_b]))
            else:
                coeffs.append(num_num_metric(df[sr_a], df[sr_b]))
        series_list.append(pd.Series(coeffs, index=df.columns, name=sr_a))
    return pd.concat(series_list, axis=1, keys=[series.name for series in series_list])


def _cramers_v(sr_a: pd.Series, sr_b: pd.Series) -> float:
    confusion_matrix = pd.crosstab(sr_a, sr_b)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = len(sr_a)
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def _pearson(sr_a: pd.Series, sr_b: pd.Series) -> float:
    if not np.issubdtype(sr_a.dtype, np.number) or not np.issubdtype(sr_b.dtype, np.number):
        return 0
    return abs(sr_a.corr(sr_b))


def _kruskal_wallis(sr_a: pd.Series, sr_b: pd.Series) -> float:
    if not sr_a.map(type).eq(str).all() and not sr_b.str.isnumeric().all():
        return 0

    sr_a = sr_a.astype("category").cat.codes
    groups = sr_b.groupby(sr_a)
    arrays = [groups.get_group(category) for category in sr_a.unique()]

    args = [group.array for group in arrays]
    try:
        _, p_val = ss.kruskal(*args, nan_policy="omit")
    except ValueError:
        return 1

    return 1 - p_val
