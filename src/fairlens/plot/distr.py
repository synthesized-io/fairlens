"""
Visualize distributions of data.
"""

import itertools
from math import ceil
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from .. import utils


def distr_plot(
    df: pd.DataFrame,
    target_attr: str,
    groups: Sequence[Union[Mapping[str, List[Any]], pd.Series]],
    distr_type: Optional[str] = None,
    show_hist: Optional[bool] = None,
    show_curve: Optional[bool] = None,
    shade: bool = True,
    normalize: bool = False,
    cmap: Optional[Sequence[Tuple[float, float, float]]] = None,
    ax: Optional[Axes] = None,
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
        show_hist (Optional[bool], optional):
            Shows the histogram if True. Defaults to True if the data is categorical or binary.
        show_curve (Optional[bool], optional):
            Shows a KDE if True. Defaults to True if the data is continuous or a date.
        shade (bool, optional):
            Shades the curve if True. Defaults to True.
        normalize (bool, optional):
            Normalizes the counts so the sum of the bar heights is 1. Defaults to False.
        cmap (Optional[Sequence[Tuple[float, float, float]]], optional):
            A sequence of RGB tuples used to colour the histograms. If None seaborn's default pallete
            will be used. Defaults to None.
        ax (Optional[matplotlib.axes.Axes], optional):
            An axis to plot the figure on. Defaults to plt.gca(). Defaults to None.

    Returns:
        matplotlib.axes.Axes:
            The matplotlib axis containing the plot.

    Examples:
        >>> df = pd.read_csv("datasets/compas.csv")
        >>> g1 = {"Ethnicity": ["African-American"]}
        >>> g2 = {"Ethnicity": ["Caucasian"]}
        >>> distr_plot(df, "RawScore", [g1, g2])
        >>> plt.show()

        .. image:: ../../savefig/distr_plot.png
    """

    if ax is None:
        ax = plt.gca()

    preds = utils.get_predicates_mult(df, groups)

    cmap = cmap or sns.color_palette("deep")
    palette = itertools.cycle(cmap)

    column = utils.infer_dtype(df[target_attr])

    if distr_type is None:
        distr_type = utils.infer_distr_type(column).value

    if show_hist is None:
        show_hist = distr_type in ["categorical", "binary"]

    if show_curve is None:
        show_curve = distr_type in ["continuous", "datetime"]

    shrink = int(show_hist)
    stat = "probability" if normalize else "count"

    if distr_type == "continuous":
        _, bins = utils.zipped_hist((df[target_attr],), ret_bins=True, distr_type=distr_type)
    elif distr_type == "datetime":
        bins = utils.fd_opt_bins(column)  # TODO: Look at seaborn log scaling in more detail
    elif column.dtype in ["int64", "float64"]:
        bins = np.arange(0, column.max() + 1.5) - 0.5
        ax.set_xticks(bins + 0.5)
    else:
        bins = "auto"

    plt.xlabel(target_attr)

    for pred in preds:
        sns.histplot(column[pred], bins=bins, color=next(palette), kde=show_curve, shrink=shrink, stat=stat, ax=ax)

    if shade and not show_hist:
        _shade_area(ax, cmap, alpha=0.3)

    return ax


def attr_distr_plot(
    df: pd.DataFrame,
    target_attr: str,
    attr: str,
    distr_type: Optional[str] = None,
    attr_distr_type: Optional[str] = None,
    max_quantiles: int = 8,
    separate: bool = False,
    show_hist: Optional[bool] = None,
    show_curve: Optional[bool] = None,
    shade: bool = True,
    normalize: bool = False,
    cmap: Optional[Sequence[Tuple[float, float, float]]] = None,
    ax: Optional[Axes] = None,
) -> Optional[Axes]:
    """Plot the distribution of the target attribute with respect to all the unique values in the column `attr`.

    Args:
        df (pd.DataFrame):
            The input dataframe.
        target_attr (str):
            The target attribute.
        attr (str):
            The attribute whose values' distributions are to be plotted.
        distr_type (Optional[str], optional):
            The type of distribution of the target attribute. Can take values from
            ["categorical", "continuous", "binary", "datetime"]. If None, the type of
            distribution is inferred based on the data in the column. Defaults to None.
        attr_distr_type (Optional[str], optional):
            The type of distribution of `attr`. Can be "categorical", "continuous" or "datetime".
            If None the type of distribution is inferred based on the data in the column.
            Defaults to None.
        max_quantiles (int, optional):
            The maximum amount of quantiles to use for continuous data. Defaults to 8.
        separate (bool, optional):
            Separate into multiple plots (subplot). Defaults to False.
        show_hist (Optional[bool], optional):
            Shows the histogram if True. Defaults to True if the data is categorical or binary.
        show_curve (Optional[bool], optional):
            Shows a KDE if True. Defaults to True if the data is continuous or a date.
        shade (bool, optional):
            Shades the curve if True. Defaults to True.
        normalize (bool, optional):
            Normalizes the counts so the sum of the bar heights is 1. Defaults to False.
        cmap (Optional[Sequence[Tuple[float, float, float]]], optional):
            A sequence of RGB tuples used to colour the histograms. If None seaborn's default pallete
            will be used. Defaults to None.
        ax (Optional[matplotlib.axes.Axes], optional):
            An axis to plot the figure on. Defaults to plt.gca(). Defaults to None.

    Returns:
        Optional[matplotlib.axes.Axes]:
            The matplotlib axes containing the plot if `separate` is False, otherwise None.

    Examples:
        >>> df = pd.read_csv("datasets/compas.csv")
        >>> attr_distr_plot(df, "RawScore", "Ethnicity")
        >>> plt.show()

        .. image:: ../../savefig/attr_distr_plot.png
    """

    if target_attr == attr:
        raise ValueError("'target_attr' and 'attr' cannot be the same.")

    df_ = df[[attr, target_attr]].copy()

    col = utils.infer_dtype(df_[attr])

    if attr_distr_type is None:
        attr_distr_type = utils.infer_distr_type(col).value

    # Bin data
    if attr_distr_type == "continuous" or attr_distr_type == "datetime":
        df_.loc[:, attr] = utils._bin_as_string(col, attr_distr_type, max_bins=max_quantiles)

    # Values ordered by counts in order for overlay to work well.
    unique_values = df_[attr].dropna().value_counts().keys()

    labels = ["All"] + [str(val) for val in unique_values]
    groups = [pd.Series([True] * len(df_))] + [(df_[attr] == val) for val in unique_values]

    if separate:
        figsize = 6, 5

        n = len(groups)
        r = ceil(n / 3)
        c = min(n, 3)
        fig = plt.figure(figsize=(figsize[0] * c, figsize[1] * r))
        fig.tight_layout()
        plt.subplots_adjust(hspace=0.3)

        for i, (group, title) in enumerate(zip(groups, labels)):
            ax_ = fig.add_subplot(r, c, i + 1)
            distr_plot(
                df_,
                target_attr,
                [group],
                distr_type=distr_type,
                show_hist=show_hist,
                show_curve=show_curve,
                shade=shade,
                normalize=normalize,
                cmap=cmap,
                ax=ax_,
            )
            plt.title(title)

        return None

    distr_plot(
        df_,
        target_attr,
        groups,
        distr_type=distr_type,
        show_hist=show_hist,
        show_curve=show_curve,
        shade=shade,
        normalize=normalize,
        cmap=cmap,
        ax=ax,
    )

    plt.legend(labels)
    plt.title(attr)

    return ax


def mult_distr_plot(
    df: pd.DataFrame,
    target_attr: str,
    attrs: Sequence[str],
    figsize: Optional[Tuple[int, int]] = None,
    max_width: int = 3,
    distr_type: Optional[str] = None,
    attr_distr_types: Optional[Mapping[str, str]] = None,
    max_quantiles: int = 8,
    show_hist: Optional[bool] = None,
    show_curve: Optional[bool] = None,
    shade: bool = True,
    normalize: bool = False,
    cmap: Optional[Sequence[Tuple[float, float, float]]] = None,
):
    """Plot the distribution of the all values for each of the unique values in the column `attr`
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
            The maximum amount of figures. Defaults to 3.
        distr_type (Optional[str], optional):
            The type of distribution of the target attribute. Can take values from
            ["categorical", "continuous", "binary", "datetime"]. If None, the type of
            distribution is inferred based on the data in the column. Defaults to None.
        attr_distr_types (Optional[Mapping[str, str]], optional):
            The types of distribution of the attributes in `attrs`. Passed as a mapping
            from attribute name to corresponding distribution type.
            Can take values from ["categorical", "continuous", "binary", "datetime"].
            If None, the type of distribution of all sensitive attributes are inferred
            based on the data in the respective columns. Defaults to None.
        max_quantiles (int, optional):
            The maximum amount of quantiles to use for continuous data. Defaults to 8.
        show_hist (Optional[bool], optional):
            Shows the histogram if True. Defaults to True if the data is categorical or binary.
        show_curve (Optional[bool], optional):
            Shows a KDE if True. Defaults to True if the data is continuous or a date.
        shade (bool, optional):
            Shades the curve if True. Defaults to True.
        normalize (bool, optional):
            Normalizes the counts so the sum of the bar heights is 1. Defaults to False.
        cmap (Optional[Sequence[Tuple[float, float, float]]], optional):
            A sequence of RGB tuples used to colour the histograms. If None seaborn's default pallete
            will be used. Defaults to None.

    Examples:
        >>> df = pd.read_csv("datasets/compas.csv")
        >>> mult_distr_plot(df, "RawScore", ["Ethnicity", "Sex", "MaritalStatus", "Language", "DateOfBirth"])
        >>> plt.show()

    .. image:: ../../savefig/mult_distr_plot.png
    """

    attr_distr_types = attr_distr_types or {}

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
        attr_distr_type = attr_distr_types[attr] if attr in attr_distr_types else None
        attr_distr_plot(
            df,
            target_attr,
            attr,
            distr_type=distr_type,
            attr_distr_type=attr_distr_type,
            max_quantiles=max_quantiles,
            show_hist=show_hist,
            show_curve=show_curve,
            shade=shade,
            normalize=normalize,
            cmap=cmap,
            ax=ax_,
        )


def _shade_area(ax: Axes, cmap: Sequence[Tuple[float, float, float]], alpha: float = 0.3):
    """Shade area under all lines in axes."""

    palette = itertools.cycle(cmap)
    for line in ax.lines:
        xy = line.get_xydata()
        ax.fill_between(xy[:, 0], xy[:, 1], color=next(palette), alpha=alpha)
