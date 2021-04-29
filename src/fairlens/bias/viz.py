from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from . import utils


def plt_group_dist(
    df: pd.DataFrame,
    target_attr: str,
    group1: Dict[str, List[Any]],
    group2: Dict[str, List[Any]],
    title=False,
    legend=False,
):
    """Plot the distribution of the 2 groups with respect to the target attribute.

    Args:
        df (pd.DataFrame):
            The input dataframe.
        target_attr (str):
            The target attribute.
        group1 (Dict[str, List[Any]]):
            The first group of interest.
        group2 (Dict[str, List[Any]]):
            The second group of interest.
        title (bool, optional):
            Adds a title to the plot. Defaults to False.
        legend (bool, optional):
            Adds a legend to the plot. Defaults to False.
    """

    df = utils.infer_dtype(df, target_attr)
    distr_type = utils.infer_distr_type(df[target_attr])

    preds1, preds2 = utils.get_predicates(df, group1, group2)

    if distr_type.is_continuous() or str(df[target_attr].dtype) in ["float64", "int64"]:
        bins = utils.fd_opt_bins(df[target_attr])
        plt.hist(df[preds1][target_attr], bins=bins, alpha=0.5)
        plt.hist(df[preds2][target_attr], bins=bins, alpha=0.5)
    else:
        plt.hist(df[preds1][target_attr])
        plt.hist(df[preds2][target_attr])

    if title:
        plt.title(target_attr)

    if legend:
        plt.legend(
            [
                ",".join([",".join(vals) for vals in group1.values()]),
                ",".join([",".join(vals) for vals in group2.values()]),
            ]
        )