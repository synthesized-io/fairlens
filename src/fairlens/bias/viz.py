from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from . import utils


def plt_group_dist(
    df: pd.DataFrame,
    target_attr: str,
    group1: Dict[str, List[Any]],
    group2: Dict[str, List[Any]],
    normalize: bool = True,
    show_hist: bool = True,
    show_curve: bool = True,
    title: bool = False,
    legend: bool = False,
    fade: bool = True,
    **kwargs
):
    """Plot the pdf of the 2 groups with respect to the target attribute.

    Args:
        df (pd.DataFrame):
            The input dataframe.
        target_attr (str):
            The target attribute.
        group1 (Dict[str, List[Any]]):
            The first group of interest.
        group2 (Dict[str, List[Any]]):
            The second group of interest.
        normalize (bool, optional):
            Converts the distribution into a pdf if True. If not the y-axis indicates the raw counts. Defaults to True.
        show_hist (bool, optional):
            Shows the histogram if True. Defaults to True.
        show_curve (bool, optional):
            Shows the KDE curve if True. Defaults to True.
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
        >>> plt_group_dist(df, "RawScore", {"Ethnicity": ["African-American"]}, {"Ethnicity": ["Caucasian"]})
        >>> plt.show()
    """

    plt_group_dist_mult(
        df, target_attr, [group1, group2], normalize, show_hist, show_curve, title, legend, fade, **kwargs
    )


def plt_group_dist_mult(
    df: pd.DataFrame,
    target_attr: str,
    groups: List[Dict[str, List[Any]]],
    normalize: bool = True,
    show_hist: bool = False,
    show_curve: bool = True,
    title: bool = False,
    legend: bool = False,
    fade: bool = False,
    **kwargs
):
    """Plot the pdf of the groups with respect to the target attribute.

    Args:
        df (pd.DataFrame):
            The input dataframe.
        target_attr (str):
            The target attribute.
        groups (List[Dict[str, List[Any]]]):
            Groups of interest.
        normalize (bool, optional):
            Converts the distribution into a pdf if True. If not the y-axis indicates the raw counts. Defaults to True.
        show_hist (bool, optional):
            Shows the histogram if True. Defaults to False.
        show_curve (bool, optional):
            Shows the KDE curve if True. `normalize` must be True for this to work. Defaults to True.
        title (bool, optional):
            Adds a title to the plot. Defaults to False.
        legend (bool, optional):
            Adds a legend to the plot. Defaults to False.
        fade (bool, optional):
            Adds a fade to the histograms. Set false if alpha passed as kwarg. Defaults to False.
        **kwargs:
            Additional keyword arguments passed to plt.hist()

    Examples:
        >>> df = pd.read_csv("datasets/compas.csv")
        >>> g1 = {"Ethnicity": ["African-American"]}
        >>> g2 = {"Ethnicity": ["Caucasian"]}
        >>> g3 = {"Ethnicity": ["Asian"]}}
        >>> plt_group_dist_mult(df, "RawScore", [g1, g2, g3])
        >>> plt.show()
    """

    distr_type = utils.infer_distr_type(df[target_attr])

    preds = utils.get_predicates_mult(df, groups)

    if normalize:
        kwargs["density"] = True

    if fade:
        kwargs["alpha"] = 0.5

    bins = None
    if distr_type.is_continuous() or str(df[target_attr].dtype) in ["float64", "int64"]:
        bins = utils.fd_opt_bins(df[target_attr])

    # Plot the histograms
    if show_hist:
        for pred in preds:
            df[pred][target_attr].plot.hist(bins=bins, **kwargs)

    plt.gca().set_prop_cycle(None)

    # Plot the curves
    if normalize and show_curve:
        for pred in preds:
            df[pred][target_attr].plot.kde()

    plt.xlabel(target_attr)

    if title:
        plt.title(target_attr)

    if legend:
        plt.legend([",".join([",".join(vals) for vals in group.values()]) for group in groups])


def plt_attr_dist(df: pd.DataFrame, target_attr: str, attr: str, separate: bool = False, **kwargs):
    """Plot the pdf of the all values in the column `attr` with respect to the target attribute.

    Args:
        df (pd.DataFrame):
            The input dataframe.
        target_attr (str):
            The target attribute.
        attr (str):
            The attribute whose values' distributions are to be plotted.
        separate (bool):
            Separate into multiple plots (subplot). Defaults to False.
        **kwargs:
            Additional keyword arguments passed to plt_group_dist_mult()

    Examples:
        >>> df = pd.read_csv("datasets/compas.csv")
        >>> plt_attr_dist(df, "RawScore", "Ethnicity")
        >>> plt.show()
    """

    if "fade" not in kwargs and "alpha" not in kwargs:
        kwargs["fade"] = True

    if "normalize" not in kwargs:
        kwargs["normalize"] = False

    if "show_hist" not in kwargs and not kwargs["normalize"]:
        kwargs["show_hist"] = True

    legend = ["All"]
    groups = [{attr: df[attr].unique()}]
    for val in df[attr].unique():
        groups.append({attr: [val]})
        legend.append(str(val))

    if separate:
        raise NotImplementedError()

    else:
        plt_group_dist_mult(df, target_attr, groups, **kwargs)
        plt.legend(legend)
