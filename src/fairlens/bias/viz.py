from math import ceil
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from . import utils
from .metrics import auto_distance, stat_distance


def plt_group_dist(
    df: pd.DataFrame,
    target_attr: str,
    group1: Dict[str, List[Any]],
    group2: Dict[str, List[Any]],
    distr_type: Optional[str] = None,
    normalize: bool = True,
    show_hist: bool = True,
    show_curve: bool = True,
    title: bool = False,
    legend: bool = True,
    fade: bool = True,
    show_metric: bool = False,
    **kwargs,
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
        distr_type (Optional[str]):
            The type of distribution of the target attribute. Can be "categorical" or "continuous".
            If None the type of distribution is inferred based on the data in the column.
            Defaults to None.
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
        show_metric (bool, optional):
            Show suitable metric if True. Defaults to False.
        **kwargs:
            Additional keyword arguments passed to plt.hist

    Examples:
        >>> df = pd.read_csv("datasets/compas.csv")
        >>> plt_group_dist(df, "RawScore", {"Ethnicity": ["African-American"]}, {"Ethnicity": ["Caucasian"]})
        >>> plt.show()
    """

    plt_group_dist_mult(
        df, target_attr, [group1, group2], distr_type, normalize, show_hist, show_curve, title, legend, fade, **kwargs
    )

    if show_metric:
        metric = auto_distance(df[target_attr])
        mode = metric.get_id()
        distance, p_value = stat_distance(df, target_attr, group1, group2, mode=mode, p_value=True)
        plt.xlabel(f"{mode}={distance: .3g}, p-value={p_value: .3g}")


def plt_group_dist_mult(
    df: pd.DataFrame,
    target_attr: str,
    groups: Sequence[Dict[str, List[Any]]],
    distr_type: Optional[str] = None,
    normalize: bool = False,
    show_hist: bool = True,
    show_curve: bool = False,
    title: bool = False,
    legend: bool = False,
    fade: bool = True,
    **kwargs,
):
    """Plot the distribution of the groups with respect to the target attribute.

    Args:
        df (pd.DataFrame):
            The input dataframe.
        target_attr (str):
            The target attribute.
        groups (Sequence[Dict[str, List[Any]]]):
            Groups of interest.
        distr_type (Optional[str]):
            The type of distribution of the target attribute. Can be "categorical" or "continuous".
            If None the type of distribution is inferred based on the data in the column.
            Defaults to None.
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

    inferred_distr_type = utils.infer_distr_type(df[target_attr])
    is_continuous = distr_type == "continuous" if distr_type is not None else inferred_distr_type.is_continuous()

    preds = utils.get_predicates_mult(df, groups)

    if normalize:
        kwargs["density"] = True

    if fade:
        kwargs["alpha"] = 0.5

    bins = None
    if is_continuous:
        if "bins" in kwargs:
            bins = kwargs["bins"]
        else:
            _, bins = utils.histogram((df[target_attr],), ret_bins=True, distr_type=distr_type)

    # Plot the histograms
    if show_hist:
        for pred in preds:
            import seaborn

            seaborn.kdeplot
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


def plt_series_dist_mult(
    groups: Sequence[pd.Series],
    distr_type: Optional[str] = None,
    normalize: bool = True,
    show_hist: bool = False,
    show_curve: bool = True,
    fade: bool = True,
    **kwargs,
):
    """Plot the distribution of the series'.

    Args:
        groups (Sequence[pd.Series]):
            Series' of interest. The dtype of all series' must be the same.
        distr_type (Optional[str]):
            The type of distribution of the target attribute. Can be "categorical" or "continuous".
            If None the type of distribution is inferred based on the data in the column.
            Defaults to None.
        normalize (bool, optional):
            Converts the distribution into a pdf if True. If not the y-axis indicates the raw counts. Defaults to True.
        show_hist (bool, optional):
            Shows the histogram if True. Defaults to False.
        show_curve (bool, optional):
            Shows the KDE curve if True. `normalize` must be True for this to work. Defaults to True.
        fade (bool, optional):
            Adds a fade to the histograms. Set false if alpha passed as kwarg. Defaults to False.
        **kwargs:
            Additional keyword arguments passed to plt.hist()

    Examples:
        >>> df = pd.read_csv("datasets/compas.csv")
        >>> plt_series_dist_mult([df["RawScore"], df[df["Ethnicity"] == "African-American"]["RawScore"]])
        >>> plt.show()
    """

    joint = pd.concat(groups)
    inferred_distr_type = utils.infer_distr_type(joint)
    is_continuous = distr_type == "continuous" if distr_type is not None else inferred_distr_type.is_continuous()

    if normalize:
        kwargs["density"] = True

    if fade:
        kwargs["alpha"] = 0.5

    bins = None
    if is_continuous:
        bins = utils.fd_opt_bins(joint)

    # Plot the histograms
    if show_hist:
        for series in groups:
            series.plot.hist(bins=bins, **kwargs)

    plt.gca().set_prop_cycle(None)

    # Plot the curves
    if normalize and show_curve:
        for series in groups:
            series.plot.kde()


def plt_attr_dist(
    df: pd.DataFrame,
    target_attr: str,
    attr: str,
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
        separate (bool):
            Separate into multiple plots (subplot). Defaults to False.
        figure (Optional[Figure], optional):
            A matplotlib figure to plot on if `separate` is True. Defaults to matplotlib.pyplot.figure().
        max_width (int, optional):
            The maximum amount of figures in a row if `separate` is True. Defaults to 3.

        **kwargs:
            Additional keyword arguments passed to plt_group_dist_mult().

    Examples:
        >>> df = pd.read_csv("datasets/compas.csv")
        >>> plt_attr_dist(df, "RawScore", "Ethnicity")
        >>> plt.show()
    """

    unique = df[attr].dropna().value_counts().keys()
    groups = [{attr: unique}]

    legend = ["All"]
    for val in unique:
        groups.append({attr: [val]})
        legend.append(str(val))

    if separate:
        max_width = 3
        n = len(groups)
        fig = figure or plt.figure(figsize=(20, 10))
        r = ceil(n / max_width)
        c = min(n, 3)

        for i, (group, title) in enumerate(zip(groups, legend)):
            fig.add_subplot(r, c, i + 1)
            plt_group_dist_mult(df, target_attr, [group], **kwargs)
            plt.title(title)

    else:
        plt_group_dist_mult(df, target_attr, groups, **kwargs)
        plt.title(attr)
        plt.legend(legend)


def plt_attr_dist_mult(
    df: pd.DataFrame,
    target_attr: str,
    attrs: Sequence[str],
    figure: Optional[Figure] = None,
    max_width: int = 3,
    **kwargs,
):
    """Plot the pdf of the all values for each of the  the column `attr` with respect to the target attribute.

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
            Additional keywords passed down to plt_group_dist_mult().

    Examples:
        >>> df = pd.read_csv("datasets/compas.csv")
        >>> plt_all_attr_dist(df, "RawScore", ['Sex', 'MaritalStatus', 'Ethnicity', 'Language', 'DateOfBirth'])
        >>> plt.show()
    """

    n = len(attrs)
    fig = figure or plt.figure(figsize=(20, 10))
    r = ceil(n / max_width)
    c = min(n, max_width)

    for i, attr in enumerate(attrs):
        fig.add_subplot(r, c, i + 1)
        plt_attr_dist(df, target_attr, attr, **kwargs)
