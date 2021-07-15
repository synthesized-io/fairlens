import itertools
from math import ceil
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from . import utils
from .metrics import auto_distance, stat_distance

sns.reset_defaults()
sns.set_style("darkgrid")
sns.set(font="Verdana")
sns.set_context("paper", font_scale=0.8)


def distr_pair_plot(
    df: pd.DataFrame,
    target_attr: str,
    group1: Union[Mapping[str, List[Any]], pd.Series],
    group2: Union[Mapping[str, List[Any]], pd.Series],
    distr_type: Optional[str] = None,
    show_curve: bool = True,
    show_metric: bool = True,
    cmap: Optional[Sequence[Tuple[float, float, float]]] = None,
    labels: Optional[Sequence[str]] = None,
    **kwargs,
):
    """Plot the distribution of the 2 groups with respect to the target attribute.

    Args:
        df (pd.DataFrame):
            The input dataframe.
        target_attr (str):
            The target attribute.
        group1 (Union[Mapping[str, List[Any]], pd.Series]):
            First group of interest. Each group can be a mapping / dict from attribute to value or
            a predicate itself, i.e. pandas series consisting of bools which can be used as a predicate
            to index a subgroup from the dataframe.
        group2 (Union[Mapping[str, List[Any]], pd.Series]):
            Second group of interest. Each group can be a mapping / dict from attribute to value or
            a predicate itself, i.e. pandas series consisting of bools which can be used as a predicate
            to index a subgroup from the dataframe.
        distr_type (Optional[str]):
            The type of distribution of the target attribute. Can be "categorical" or "continuous".
            If None the type of distribution is inferred based on the data in the column.
            Defaults to None.
        show_curve (bool, optional):
            Shows a KDE if True. Defaults to True.
        show_metric (bool, optional):
            Show suitable metric if True. Defaults to False.
        cmap (Optional[Sequence[Tuple[float, float, float]]], optional):
            A sequence of RGB tuples used to colour the histograms. If None seaborn's default pallete
            will be used. Defaults to None.
        labels (Optional[Sequence[str]], optional):
            A list of labels for each of the groups which will be used for the legend.
        **kwargs:
            Additional keyword arguments passed to seaborn.histplot().

    Examples:
        >>> df = pd.read_csv("datasets/compas.csv")
        >>> distr_pair_plot(df, "RawScore", {"Ethnicity": ["African-American"]}, {"Ethnicity": ["Caucasian"]})
        >>> plt.show()
    """

    distr_plot(df, target_attr, [group1, group2], distr_type, show_curve, cmap, **kwargs)

    if show_metric:
        metric = auto_distance(df[target_attr])
        mode = metric.get_id()
        distance, p_value = stat_distance(df, target_attr, group1, group2, mode=mode, p_value=True)
        plt.xlabel(f"{mode}={distance: .3g}, p-value={p_value: .3g}")


def distr_plot(
    df: pd.DataFrame,
    target_attr: str,
    groups: Sequence[Union[Mapping[str, List[str]], pd.Series]],
    distr_type: Optional[str] = None,
    show_curve: bool = True,
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
        groups (Sequence[Union[Mapping[str, List[str]], pd.Series]]):
            A list of groups of interest. Each group can be a mapping / dict from attribute to value or
            a predicate itself, i.e. pandas series consisting of bools which can be used as a predicate
            to index a subgroup from the dataframe.
        distr_type (Optional[str]):
            The type of distribution of the target attribute. Can be "categorical" or "continuous".
            If None the type of distribution is inferred based on the data in the column.
            Defaults to None.
        show_curve (bool, optional):
            Shows a KDE if True. Defaults to True.
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

    palette = itertools.cycle(sns.color_palette() if cmap is None else cmap)

    if "kde" in kwargs:
        show_curve = kwargs.pop("kde")

    col = df[target_attr]
    inferred_col = utils.infer_dtype(col)

    if "bins" in kwargs:
        bins = kwargs.pop("bins")
    elif inferred_col.dtype == "datetime64[ns]":
        bins = utils.fd_opt_bins(inferred_col)  # TODO: Look at seaborn log scaling in more detail
        col = inferred_col
    else:
        _, bins = utils.zipped_hist((df[target_attr],), ret_bins=True, distr_type=distr_type)

    for pred in preds:
        sns.histplot(col[pred], bins=bins, color=next(palette), kde=show_curve, **kwargs)

    if labels is not None:
        plt.legend(labels)

    plt.xlabel(target_attr)


def attr_distr_plot(
    df: pd.DataFrame,
    target_attr: str,
    attr: str,
    distr_type: Optional[str] = None,
    separate: bool = False,
    figure: Optional[Figure] = None,
    max_width: int = 3,
    **kwargs,
):
    """Plot the pdf of the target attribute with respect to all the unique values in the column `attr`.

    Args:
        df (pd.DataFrame):
            The input dataframe.
        target_attr (str):
            The target attribute.
        attr (str):
            The attribute whose values' distributions are to be plotted.
        distr_type (Optional[str]):
            The type of distribution of the target attribute. Can be "categorical" or "continuous".
            If None the type of distribution is inferred based on the data in the column.
            Defaults to None.
        separate (bool):
            Separate into multiple plots (subplot). Defaults to False.
        figure (Optional[Figure], optional):
            A matplotlib figure to plot on if `separate` is True. Defaults to matplotlib.pyplot.figure().
        max_width (int, optional):
            The maximum amount of figures in a row if `separate` is True. Defaults to 3.

        **kwargs:
            Additional keyword arguments passed to seaborn.histplot().

    Examples:
        >>> df = pd.read_csv("datasets/compas.csv")
        >>> attr_distr_plot(df, "RawScore", "Ethnicity")
        >>> plt.show()
    """

    if utils.infer_distr_type(df[attr]).is_continuous():
        # TODO: need separate plot for continous data
        raise NotImplementedError()

    origin_attr = attr

    inferred_col = utils.infer_dtype(df[attr])
    binned_attr = attr + "__BINNED__"
    if inferred_col.dtype == "datetime64[ns]":
        df.loc[:, binned_attr] = ((inferred_col.dt.year // 10) * 10).apply(lambda y: str(y) + "-" + str(y + 10))
        attr = binned_attr

    unique = df[attr].dropna().value_counts().keys()

    labels = ["All"]
    groups = [df[attr] == df[attr]]

    for val in unique:
        labels.append(str(val))
        groups.append({attr: [val]})

    if separate:
        max_width = 3
        n = len(groups)
        r = ceil(n / max_width)
        c = min(n, 3)
        fig = figure or plt.figure(figsize=(6 * c, 5 * r))

        for i, (group, title) in enumerate(zip(groups, labels)):
            fig.add_subplot(r, c, i + 1)
            distr_plot(df, target_attr, [group], distr_type=distr_type, legend=False, **kwargs)
            plt.title(title)

    else:
        distr_plot(df, target_attr, groups, distr_type=distr_type, legend=False, labels=labels, **kwargs)
        plt.title(origin_attr)

    if binned_attr in df.columns:
        df.drop(columns=[binned_attr])


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
            Additional keywords passed down to seaborn.histplot().

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
