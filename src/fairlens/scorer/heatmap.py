from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from fairlens.bias import utils
from fairlens.metrics import correlation_metrics


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
    num_num_metric = num_num_metric or correlation_metrics.pearson
    cat_num_metric = cat_num_metric or correlation_metrics.kruskal_wallis
    cat_cat_metric = cat_cat_metric or correlation_metrics.cramers_v

    corr_matrix = compute_correlation_matrix(df, num_num_metric, cat_num_metric, cat_cat_metric).round(2)

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr_matrix, vmin=0, vmax=1, annot=True)


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
    nn_metric = num_num_metric or correlation_metrics.pearson
    cn_metric = cat_num_metric or correlation_metrics.kruskal_wallis
    cc_metric = cat_cat_metric or correlation_metrics.cramers_v

    series_list = list()
    for sr_a in df.columns:
        coeffs = list()
        a_type = utils.infer_distr_type(df[sr_a])

        for sr_b in df.columns:
            if sr_a == sr_b:
                coeffs.append(1.0)
                continue

            b_type = utils.infer_distr_type(df[sr_b])
            if a_type.is_continuous() and b_type.is_continuous():
                coeffs.append(nn_metric(df[sr_a], df[sr_b]))

            elif a_type.is_continuous():
                coeffs.append(cn_metric(df[sr_b], df[sr_a]))

            elif b_type.is_continuous():
                coeffs.append(cn_metric(df[sr_a], df[sr_b]))

            else:
                coeffs.append(cc_metric(df[sr_a], df[sr_b]))

        series_list.append(pd.Series(coeffs, index=df.columns, name=sr_a))

    return pd.concat(series_list, axis=1, keys=[series.name for series in series_list])
