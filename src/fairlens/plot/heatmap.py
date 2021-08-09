"""
Plot correlation heatmaps for datasets.
"""

from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..metrics import correlation, unified


def two_column_heatmap(
    df: pd.DataFrame,
    num_num_metric: Callable[[pd.Series, pd.Series], float] = correlation.pearson,
    cat_num_metric: Callable[[pd.Series, pd.Series], float] = correlation.kruskal_wallis,
    cat_cat_metric: Callable[[pd.Series, pd.Series], float] = correlation.cramers_v,
    columns_x: Optional[List[str]] = None,
    columns_y: Optional[List[str]] = None,
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
        columns_x (Optional[List[str]]):
            The sensitive dataframe column names that will be used in generating the correlation heatmap.
        columns_y (Optional[List[str]]):
            The non-sensitive dataframe column names that will be used in generating the correlation heatmap.
    """

    if columns_x is None:
        columns_x = df.columns

    if columns_y is None:
        columns_y = df.columns

    corr_matrix = unified.correlation_matrix(
        df, num_num_metric, cat_num_metric, cat_cat_metric, columns_x, columns_y
    ).round(2)

    fig_width = 20.0
    margin_top = 0.8
    margin_bot = 0.8
    margin_left = 0.8
    margin_right = 0.8

    cell_size = (fig_width - margin_left - margin_right) / float(len(columns_y))
    fig_height = cell_size * len(columns_x) + margin_bot + margin_top

    plt.figure(figsize=(fig_width, fig_height), tight_layout=True)
    plt.subplots_adjust(
        bottom=margin_bot / fig_height,
        top=1.0 - margin_top / fig_height,
        left=margin_left / fig_width,
        right=1.0 - margin_right / fig_width,
    )

    g = sns.heatmap(
        corr_matrix,
        vmin=0,
        vmax=1,
        annot=True,
        annot_kws={"size": 35 / np.sqrt(len(corr_matrix))},
        square=True,
        cbar=True,
    )

    g.set_xticklabels(g.get_xticklabels(), rotation=90, horizontalalignment="right", fontdict={"fontsize": 14})
    g.set_yticklabels(g.get_yticklabels(), rotation=0, horizontalalignment="right", fontdict={"fontsize": 14})
