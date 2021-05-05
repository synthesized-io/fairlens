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
    fade=True,
    **kwargs
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
        fade (bool, optional):
            Adds a fade to the histograms. Set false if alpha passed as kwarg. Defaults to True.
        **kwargs:
            Additional keyword arguments passed to plt.hist

    Examples:
        >>> df = pd.read_csv("datasets/compas.csv")
        >>> plt_group_dist(df, 'RawScore', {'Ethnicity': ['African-American']}, {'Ethnicity': ['Caucasian']})
        >>> plt.show()
    """

    plt_group_dist_mult(df, target_attr, [group1, group2], title=title, legend=legend, fade=fade, **kwargs)


def plt_group_dist_mult(
    df: pd.DataFrame,
    target_attr: str,
    groups: List[Dict[str, List[Any]]],
    title=False,
    legend=False,
    fade=True,
    **kwargs
):
    """Plot the distribution of the groups with respect to the target attribute.

    Args:
        df (pd.DataFrame):
            The input dataframe.
        target_attr (str):
            The target attribute.
        groups (List[Dict[str, List[Any]]]):
            Groups of interest.
        title (bool, optional):
            Adds a title to the plot. Defaults to False.
        legend (bool, optional):
            Adds a legend to the plot. Defaults to False.
        fade (bool, optional):
            Adds a fade to the histograms. Set false if alpha passed as kwarg. Defaults to True.
        **kwargs:
            Additional keyword arguments passed to plt.hist

    Examples:
        >>> df = pd.read_csv("datasets/compas.csv")
        >>> g1 = {'Ethnicity': ['African-American']}
        >>> g2 = {'Ethnicity': ['Caucasian']}
        >>> g3 = {'Ethnicity': ['Asian']}}
        >>> plt_group_dist_mult(df, 'RawScore', [g1, g2, g3])
        >>> plt.show()
    """

    distr_type = utils.infer_distr_type(df[target_attr])

    preds = utils.get_predicates_mult(df, groups)

    if fade:
        kwargs["alpha"] = 0.5

    bins = None
    if distr_type.is_continuous() or str(df[target_attr].dtype) in ["float64", "int64"]:
        bins = utils.fd_opt_bins(df[target_attr])
        df["Binned"] = utils.bin(df["RawScore"], n_bins=bins, mean_bins=True)

    # Plot the histograms
    for pred in preds:
        plt.hist(df[pred][target_attr], bins=bins, **kwargs)

    plt.gca().set_prop_cycle(None)

    # Plot the curves
    for pred in preds:
        g_counts = df[pred].groupby(target_attr)[target_attr].aggregate(Count="count")["Count"].to_dict()

        plt.plot(g_counts.keys(), g_counts.values())

    if title:
        plt.title(target_attr)

    if legend:
        plt.legend([",".join([",".join(vals) for vals in group.values()]) for group in groups])
