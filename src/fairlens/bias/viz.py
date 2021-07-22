import itertools
from math import ceil
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from . import utils

sns.reset_defaults()
sns.set_style("darkgrid")
sns.set(font="Verdana")
sns.set_context("paper", font_scale=0.8)


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
    **kwargs,
):
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
        **kwargs:
            Additional keyword arguments passed to seaborn.histplot().

    Examples:
        >>> df = pd.read_csv("datasets/compas.csv")
        >>> g1 = {"Ethnicity": ["African-American"]}
        >>> g2 = {"Ethnicity": ["Caucasian"]}
        >>> g3 = {"Ethnicity": ["Asian"]}}
        >>> distr_plot(df, "RawScore", [g1, g2, g3])
        >>> plt.show()
    """

    preds = utils.get_predicates_mult(df, groups)

    palette = itertools.cycle(sns.color_palette("deep") if cmap is None else cmap)

    column = utils.infer_dtype(df[target_attr])

    if distr_type is None:
        distr_type = utils.infer_distr_type(column).value

    if "color" in kwargs:
        raise ValueError("Colors cannot be passed directly as kwargs. Use the cmap argument instead")

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
    elif distr_type == "datetime64[ns]":
        bins = utils.fd_opt_bins(column)  # TODO: Look at seaborn log scaling in more detail
    elif distr_type == "continuous":
        _, bins = utils.zipped_hist((df[target_attr],), ret_bins=True, distr_type=distr_type)
    elif column.dtype == "int64":
        bins = sorted(column.unique())
    else:
        bins = "auto"

    for pred in preds:
        sns.histplot(column[pred], bins=bins, color=next(palette), kde=kde, shrink=shrink, stat=stat, **kwargs)

    if labels is not None:
        plt.legend(labels)

    plt.xlabel(target_attr)


def attr_distr_plot(
    df: pd.DataFrame,
    target_attr: str,
    attr: str,
    distr_type: Optional[str] = None,
    attr_distr_type: Optional[str] = None,
    max_bins=10,
    separate: bool = False,
    figure: Optional[Figure] = None,
    max_width: int = 3,
    **kwargs,
):
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
        figure (Optional[Figure], optional):
            A matplotlib figure to plot on if `separate` is True. Defaults to matplotlib.pyplot.figure().
        max_width (int, optional):
            The maximum amount of figures in a row if `separate` is True. Defaults to 3.
        **kwargs:
            Additional keyword arguments passed to distr_plot().

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
        n = len(groups)
        r = ceil(n / max_width)
        c = min(n, max_width)
        fig = figure or plt.figure(figsize=(6 * c, 4 * r))

        for i, (group, title) in enumerate(zip(groups, labels)):
            fig.add_subplot(r, c, i + 1)
            distr_plot(df_, target_attr, [group], distr_type=distr_type, **kwargs)
            plt.title(title)

    else:
        distr_plot(df_, target_attr, groups, distr_type=distr_type, legend=False, labels=labels, **kwargs)
        plt.title(attr)


def mult_distr_plot(
    df: pd.DataFrame,
    target_attr: str,
    attrs: Sequence[str],
    figure: Optional[Figure] = None,
    max_width: int = 3,
    **kwargs,
):
    """Plot the pdf of the all values for each of the unique values in the column `attr`
    with respect to the target attribute.

    Args:
        df (pd.DataFrame):
            The input dataframe.
        target_attr (str):
            The target attribute.
        attrs (Sequence[str]):
            The attributes whose value distributions are to be plotted.
        figure (Optional[Figure], optional):
            A matplotlib figure to plot on. Defaults to matplotlib.pyplot.figure().
        max_width (int, optional):
            The maximum amount of figures in a row. Defaults to 3.
        **kwargs:
            Additional keywords passed down to attr_distr_plot().

    Examples:
        >>> df = pd.read_csv("datasets/compas.csv")
        >>> mult_distr_plot(df, "RawScore", ['Sex', 'MaritalStatus', 'Ethnicity', 'Language', 'DateOfBirth'])
        >>> plt.show()
    """

    n = len(attrs)
    r = ceil(n / max_width)
    c = min(n, max_width)
    fig = figure or plt.figure(figsize=(6 * c, 4 * r))

    for i, attr in enumerate(attrs):
        ax = fig.add_subplot(r, c, i + 1)
        attr_distr_plot(df, target_attr, attr, ax=ax, **kwargs)
