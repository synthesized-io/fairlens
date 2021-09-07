"""
Plot correlation heatmaps for datasets.
"""

from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..metrics import correlation, unified


def heatmap(
    df: pd.DataFrame,
    num_num_metric: Callable[[pd.Series, pd.Series], float] = correlation.pearson,
    cat_num_metric: Callable[[pd.Series, pd.Series], float] = correlation.kruskal_wallis,
    cat_cat_metric: Callable[[pd.Series, pd.Series], float] = correlation.cramers_v,
    **kwargs
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
        kwargs:
            Key word arguments for sns.heatmap.
    """

    corr_matrix = unified.correlation_matrix(df, num_num_metric, cat_num_metric, cat_cat_metric)

    if "cmap" not in kwargs:
        kwargs["cmap"] = sns.cubehelix_palette(start=0.2, rot=-0.2, dark=0.3, as_cmap=True)

    if "linewidth" not in kwargs:
        kwargs["linewidth"] = 0.5

    sns.heatmap(corr_matrix, vmin=0, vmax=1, square=True, **kwargs)
    plt.tight_layout()
