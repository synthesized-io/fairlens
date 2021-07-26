import itertools
from math import ceil
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from . import utils


def use_style():
    """Set the default seaborn style to a predefined style that works well with the package."""

    sns.reset_defaults()
    sns.set_style("darkgrid")
    sns.set(font="Verdana")
    sns.set_context("paper")


def reset_style():
    """Restore the seaborn style to its defaults."""

    sns.reset_defaults()


def distr_plot(
    df: pd.DataFrame,
    target_attr: str,
    groups: Sequence[Union[Mapping[str, List[Any]], pd.Series]],
    distr_type: Optional[str] = None,
    show_hist: bool = True,
    show_curve: bool = True,
    normalize: bool = False,
    cmap: Optional[Sequence[Tuple[float, float, float]]] = None,
    labels: Optional[Sequence[str]] = None,
    ax: Optional[Axes] = None,
    **kwargs,
) -> Axes:
    """Plot the distribution of the groups with respect to the target attribute.

    Args:
        df (pd.DataFrame):
            The input dataframe.
        target_attr (str):
            The target attribute.
        groups (Sequence[Union[Mapping[str, List[Any]], pd.Series]]):
            A list of groups of interest. Each group can be a mapping / dict from attribute to value or
            a predicate itself, i.e. pandas series consisting of bools which can be used as a predicate
            to index a subgroup from the dataframe.
            Examples: {"Sex": ["Male"]}, df["Sex"] == "Female"
        distr_type (Optional[str]):
            The type of distribution of the target attribute. Can take values from
            ["categorical", "continuous", "binary", "datetime"]. If None, the type of
            distribution is inferred based on the data in the column. Defaults to None.
        show_curve (bool, optional):
            Shows the histogram if True. Defaults to True.
        show_curve (bool, optional):
            Shows a KDE if True. Defaults to True.
        normalize (bool, optional):
            Normalizes the counts so the sum of the bar heights is 1. Defaults to False.
        cmap (Optional[Sequence[Tuple[float, float, float]]], optional):
            A sequence of RGB tuples used to colour the histograms. If None seaborn's default pallete
            will be used. Defaults to None.
        labels (Optional[Sequence[str]], optional):
            A list of labels for each of the groups which will be used for the legend.
        ax (Optional[matplotlib.axes.Axes], optional):
            An axis to plot the figure on. Defaults to plt.gca().
        **kwargs:
            Additional keyword arguments passed to seaborn.histplot().

    Returns:
        matplotlib.axes.Axes:
            The matplotlib axis containing the plot.

    Examples:
        >>> df = pd.read_csv("datasets/compas.csv")
        >>> g1 = {"Ethnicity": ["African-American"]}
        >>> g2 = {"Ethnicity": ["Caucasian"]}
        >>> g3 = {"Ethnicity": ["Asian"]}}
        >>> distr_plot(df, "RawScore", [g1, g2, g3])
        >>> plt.show()
    """

    if ax is None:
        ax = plt.gca()

    preds = utils.get_predicates_mult(df, groups)

    cmap = cmap or sns.color_palette("deep")
    palette = itertools.cycle(cmap)

    column = utils.infer_dtype(df[target_attr])

    if distr_type is None:
        distr_type = utils.infer_distr_type(column).value

    kde = show_curve
    if "kde" in kwargs:
        kde = kwargs.pop("kde")

    shrink = 1 if show_hist else 0
    if "shrink" in kwargs:
        shrink = kwargs.pop("shrink")

    stat = "probability" if normalize else "count"
    if "stat" in kwargs:
        stat = kwargs.pop("stat")

    if "bins" in kwargs:
        bins = kwargs.pop("bins")
    elif distr_type == "datetime":
        bins = utils.fd_opt_bins(column)  # TODO: Look at seaborn log scaling in more detail
    elif distr_type == "continuous":
        _, bins = utils.zipped_hist((df[target_attr],), ret_bins=True, distr_type=distr_type)
    elif column.dtype == "int64":
        bins = sorted(column.unique())
    else:
        bins = "auto"

    for pred in preds:
        sns.histplot(column[pred], bins=bins, color=next(palette), kde=kde, shrink=shrink, stat=stat, ax=ax, **kwargs)

    if labels is not None:
        plt.legend(labels)

    plt.xlabel(target_attr)

    return ax


def attr_distr_plot(
    df: pd.DataFrame,
    target_attr: str,
    attr: str,
    distr_type: Optional[str] = None,
    attr_distr_type: Optional[str] = None,
    max_bins=10,
    separate: bool = False,
    figsize: Optional[Tuple[int, int]] = None,
    max_width: int = 3,
    ax: Optional[Axes] = None,
    **kwargs,
) -> Union[Axes, Figure]:
    """Plot the distribution of the target attribute with respect to all the unique values in the column `attr`.

    Args:
        df (pd.DataFrame):
            The input dataframe.
        target_attr (str):
            The target attribute.
        attr (str):
            The attribute whose values' distributions are to be plotted.
        distr_type (Optional[str]):
            The type of distribution of the target attribute. Can take values from
            ["categorical", "continuous", "binary", "datetime"]. If None, the type of
            distribution is inferred based on the data in the column. Defaults to None.
        attr_distr_type (Optional[str], optional):
            The type of distribution of attr. Can be "categorical" or "continuous".
            If None the type of distribution is inferred based on the data in the column.
            Defaults to None.
        max_bins (int, optional):
            The maximum amount of bins to use for continuous data. Defaults to 10.
        separate (bool, optional):
            Separate into multiple plots (subplot). Defaults to False.
        figsize (Optional[Tuple[int, int]], optional):
            The size of each figure if `separate` is True. Defaults to (6, 4).
        max_width (int, optional):
            The maximum amount of figures in a row if `separate` is True. Defaults to 3.
        ax (Optional[matplotlib.Axes], optional):
            An axis to plot the figure on. Defaults to plt.gca().
        **kwargs:
            Additional keyword arguments passed to distr_plot() or sns.histplot().

    Returns:
        Union[matplotlib.axes.Axes, matplotlib.figure.Figure]:
            The matplotlib axes containing the plot if `separate` is False, otherwise the
            matplotlib figure containing the set of plots.

    Examples:
        >>> df = pd.read_csv("datasets/compas.csv")
        >>> attr_distr_plot(df, "RawScore", "Ethnicity")
        >>> plt.show()
    """

    if target_attr == attr:
        raise ValueError("'target_attr' and 'attr' cannot be the same.")

    df_ = df[[attr, target_attr]].copy()

    col = utils.infer_dtype(df_[attr])

    if attr_distr_type is None:
        attr_distr_type = utils.infer_distr_type(col).value

    # Bin data
    if attr_distr_type == "continuous":
        quantiles = min(max_bins, utils.fd_opt_bins(df_[attr]))
        df_.loc[:, attr] = pd.qcut(df_[attr], quantiles).apply(
            lambda iv: "[" + "{:.2f}".format(iv.left) + ", " + "{:.2f}".format(iv.right) + "]"
        )

    elif attr_distr_type == "datetime":
        df_.loc[:, attr] = utils.quantize_date(col)

    # Values ordered by counts in order for overlay to work well.
    unique_values = df_[attr].dropna().value_counts().keys()

    labels = ["All"] + [str(val) for val in unique_values]
    groups = [pd.Series([True] * len(df_))] + [(df_[attr] == val) for val in unique_values]

    if separate:
        if figsize is None:
            figsize = 6, 5

        n = len(groups)
        r = ceil(n / max_width)
        c = min(n, max_width)
        fig = plt.figure(figsize=(figsize[0] * c, figsize[1] * r))
        fig.tight_layout()
        plt.subplots_adjust(hspace=0.3)

        for i, (group, title) in enumerate(zip(groups, labels)):
            ax_ = fig.add_subplot(r, c, i + 1)
            distr_plot(df_, target_attr, [group], distr_type=distr_type, ax=ax_, **kwargs)
            plt.title(title)

        return fig

    if ax is None:
        ax = plt.gca()

    distr_plot(df_, target_attr, groups, distr_type=distr_type, legend=False, labels=labels, ax=ax, **kwargs)
    plt.title(attr)

    return ax


def mult_distr_plot(
    df: pd.DataFrame,
    target_attr: str,
    attrs: Sequence[str],
    figsize: Optional[Tuple[int, int]] = None,
    max_width: int = 3,
    **kwargs,
) -> Figure:
    """Plot the pdf of the all values for each of the unique values in the column `attr`
    with respect to the target attribute.

    Args:
        df (pd.DataFrame):
            The input dataframe.
        target_attr (str):
            The target attribute.
        attrs (Sequence[str]):
            The attributes whose value distributions are to be plotted.
        figsize (Optional[Tuple[int, int]], optional):
            The size of each figure if `separate` is True. Defaults to (6, 4).
        max_width (int, optional):
            The maximum amount of figures in a row if `separate` is True. Defaults to 3.
        **kwargs:
            Additional keywords passed down to attr_distr_plot().

    Returns:
        matplotlib.figure.Figure:
            A matplotlib figure containing the plots.

    Examples:
        >>> df = pd.read_csv("datasets/compas.csv")
        >>> mult_distr_plot(df, "RawScore", ['Sex', 'MaritalStatus', 'Ethnicity', 'Language', 'DateOfBirth'])
        >>> plt.show()
    """

    if figsize is None:
        figsize = 6, 4

    n = len(attrs)
    r = ceil(n / max_width)
    c = min(n, max_width)
    fig = plt.figure(figsize=(figsize[0] * c, figsize[1] * r))
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.3)

    for i, attr in enumerate(attrs):
        ax_ = fig.add_subplot(r, c, i + 1)
        attr_distr_plot(df, target_attr, attr, ax=ax_, **kwargs)

    return fig
