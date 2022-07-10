"""
Plot correlation heatmaps for datasets.
"""

from typing import Callable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from ..metrics import correlation, unified


def heatmap(
    df: pd.DataFrame,
    num_num_metric: Callable[[pd.Series, pd.Series], float] = correlation.pearson,
    cat_num_metric: Callable[[pd.Series, pd.Series], float] = correlation.kruskal_wallis,
    cat_cat_metric: Callable[[pd.Series, pd.Series], float] = correlation.cramers_v,
    cmap: Optional[Sequence[Tuple[float, float, float]]] = None,
    annotate: bool = False,
) -> Axes:
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
        cmap (Optional[Sequence[Tuple[float, float, float]]], optional):
            A sequence of RGB tuples used to colour the histograms. If None seaborn's default pallete
            will be used. Defaults to None.
        annotate (bool, optional):
            Annotate the heatmap.

    Returns:
        matplotlib.axes.Axes:
            The matplotlib axis containing the plot.

    Examples:
        >>> df = pd.read_csv("datasets/german_credit_data.csv")
        >>> heatmap(df)
        >>> plt.show()

        .. image:: ../../savefig/corr_heatmap_1.png
    """

    corr_matrix = unified.correlation_matrix(df, num_num_metric, cat_num_metric, cat_cat_metric)

    cmap = cmap or sns.cubehelix_palette(start=0.2, rot=-0.2, dark=0.3, as_cmap=True)
    annot = annotate or None

    ax = sns.heatmap(corr_matrix, vmin=0, vmax=1, square=True, cmap=cmap, linewidth=0.5, annot=annot, fmt=".1f")
    plt.tight_layout()

    return ax
