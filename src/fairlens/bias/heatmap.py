from typing import Callable, List, Optional

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
    """

    corr_matrix = unified.correlation_matrix(
        df, num_num_metric, cat_num_metric, cat_cat_metric, columns_x, columns_y
    ).round(2)
    sns.heatmap(corr_matrix, vmin=0, vmax=1, annot=True)
